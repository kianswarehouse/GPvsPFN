import torch
import json
import numpy as np
from pathlib import Path
import sys
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
import cProfile
import pstats
from datetime import datetime
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics, format_metric_value
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from tabpfn import TabPFNRegressor
import torch.nn.functional as F
# from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor

# Ensure this folder is on sys.path so local imports (e.g. load_experimental_data.py) work
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from load_experimental_data import generate_mf_wing_data
import defaults

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configure gpplus logger to show INFO messages
gpplus.config.configure_logger(level=logging.INFO)
# SEEK kernel (the only substantive change vs Kian baseline)
from gpplus.kernels import SEEKKernel, GaussianKernel, PowerExponentialKernel, LogScaleKernel, SEEKKernelTrunkHead

def wing_SF_GPvsPFN(
    num_folds: int = defaults.NUM_FOLDS,
    num_test: int = 5000,
    train_size: int = 10,  # total training size is train_size * input_dim
    num_runs: int = defaults.TRAINER_NUM_RUNS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float = defaults.TRAINER_LR,
    convergence_patience: int = defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_loss_change: float = defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    save_path: str | None = "./results/wing",
    title: str | None = None,
    standardize_X: bool = True,
    standardize_y: bool = True,
    x_standardize_method: int = 2,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    pfn_dtype: torch.dtype = defaults.DTYPE_PFN,
    trainer_info: bool = True,
    run_models: str | None = None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
    kernel_type: str | None = None,  # None=default, 'Gaussian', 'PowerExponential', 'Matern'
    *,
    seek_use_bias: bool = True,
    seek_weight_trunk_layer_config: dict | None = None,
    seek_weight_head_config: dict | None = None,
    seek_bias_trunk_layer_config: dict | None = None,
    seek_bias_head_config: dict | None = None,
    seek_share_bias_trunk: bool = False,
):
    """Single-fidelity Wing: TabPFN vs GPPlus with SEEKKernel.

    Notes:
    - Data is generated from the MF wing generator but we drop the source column (s0 only).
    - The GP covariance is SEEKKernel with an ensemble of continuous base kernels.
    """

    if run_models == "pfn":
        # If only running PFN, skip GP restarts entirely.
        num_runs = 0

    if title is None:
        title = f"wing_SF_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"wing_SF_{title}_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"

    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device)
    regressor = None
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Calculate total samples needed
    train_per_fold = train_size * 10
    total_train = num_folds * train_per_fold
    total_samples = num_test + total_train
    
    # Generate all unique Sobol samples at once
    print(f"Generating {total_samples} unique Sobol samples\n\tTest samples: {num_test} / Train samples: {total_train}")
    X_train_all, y_train_all, X_test_all, y_test_all = generate_mf_wing_data(
        train_samples_per_source=[total_train, 0, 0, 0], 
        test_samples_per_source=[num_test, 0, 0, 0], 
        train_noise=noise_train, 
        test_noise=noise_test, 
        noise_type=noise_type,
        seed=seed,
    )
    # Drop the 11th (source) column since SF uses only s0
    if X_train_all.shape[1] == 11:
        X_train_all = X_train_all[:, :10]
    if X_test_all.shape[1] == 11:
        X_test_all = X_test_all[:, :10]

    input_dim = int(X_train_all.shape[1])
    train_per_fold = train_size * input_dim
    total_train = num_folds * train_per_fold

    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    X_enc_train_all, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    # Normalize to plain Python lists for SEEK
    cont_cols = list(cont_cols) if cont_cols is not None else []
    cat_cols = cat_cols or []
    source_cols = source_cols or []

    # If everything looks continuous, encode_qual_data can return empty cont_cols in some edge cases.
    if not cont_cols:
        cont_cols = list(range(input_dim))

    TabPFN_metrics: list[dict] = []
    GPPlus_metrics: list[dict] = []
    GPTrainer_info: list[dict] = []

    # Randomize across the single source (s0), then split across folds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)
        
    total_start_time = time.time()
    for i in range(num_folds):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]

        X_train = X_train_all[fold_train_indices]
        # X_enc_train = X_enc_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section (SEEK)
        # =============================================================================
        if run_models in [None, "gp"]:
            print(f"\n--- {title} GP(SEEK) Training ---")

            # Reuse PFN split, convert to torch
            X_train = X_train.detach().clone().to(dtype=gp_dtype)
            X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
            y_train = y_train.detach().clone().to(dtype=gp_dtype)
            y_test = y_test_all.detach().clone().to(dtype=gp_dtype)

            # Determine X scaling type
            if standardize_X:
                if x_standardize_method == 0:
                    Xscaler = gpplus.utils.StandardScaler()
                    X_scaling_type = "StandardScaler (Gaussian)"
                elif x_standardize_method == 1:
                    Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                    X_scaling_type = "UniformScaler [0, 1]"
                elif x_standardize_method == 2:
                    Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                    X_scaling_type = "UniformScaler [-1, 1]"
                else:
                    raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
                Xscaler.fit(X_train[:, cont_cols])
                X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
                X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
            else:
                X_scaling_type = "None"

            # Normalize y
            Yscaler = gpplus.utils.StandardScaler()
            Yscaler.fit(y_train)
            y_train_mean = Yscaler.mean 
            y_train_std = Yscaler.std
            y_train_normal = Yscaler.transform(y_train)

            # --- kernel configuration ---
            cont_cols_seek = cont_cols
            if cont_cols_seek is None or len(cont_cols_seek) == 0:
                cont_cols_seek = list(range(X_train.shape[1]))
            cont_dim = len(cont_cols_seek)

            # --- SEEK kernel definition (uses gpplus/kernels/seek_kernel.py trunk/head API) ---
            # Choose base continuous kernels to ensemble inside SEEK.
            if kernel_type == "PowerExponential":
                continuous_kernels = [gpplus.kernels.PowerExponentialKernel(ard_num_dims=cont_dim)]
            elif kernel_type == "Matern":
                continuous_kernels = [gpplus.kernels.MaternKernel(nu=2.5, ard_num_dims=cont_dim)]
            else:
                # Default (and also kernel_type == "Gaussian" or None): Gaussian base kernel
                # continuous_kernels = [GaussianKernel(ard_num_dims=cont_dim), PowerExponentialKernel(ard_num_dims=cont_dim)]
                continuous_kernels = [GaussianKernel(ard_num_dims=cont_dim)]
            # Default SEEK trunk/head configs (can be overridden via function args)
            # Note: With the new shared weight network architecture, weight_head_config["dims"]
            # should match the number of base kernels. If not specified, it auto-defaults.
            act = torch.nn.Softplus
            # if seek_weight_trunk_layer_config is None:
            #     seek_weight_trunk_layer_config = {
            #         0: {"dims": input_dim, "activation": act},
            #         1: {"dims": 2, "activation": act},
            #     }
            # if seek_weight_head_config is None:
            #     # Don't set dims - let SEEKKernel auto-default to len(continuous_kernels)
            #     seek_weight_head_config = {"activation": torch.nn.Identity}
            # elif "dims" in seek_weight_head_config:
            #     # Warn if user explicitly sets dims that don't match number of base kernels
            #     num_base_kernels = len(continuous_kernels)
            #     if seek_weight_head_config["dims"] != num_base_kernels:
            #         import warnings
            #         warnings.warn(
            #             f"weight_head_config['dims']={seek_weight_head_config['dims']} does not match "
            #             f"number of base kernels ({num_base_kernels}). "
            #             f"Setting dims to {num_base_kernels} for proper behavior.",
            #             UserWarning
            #         )
            #         seek_weight_head_config["dims"] = num_base_kernels
            # if seek_bias_trunk_layer_config is None:
            #     seek_bias_trunk_layer_config = {
            #         0: {"dims": input_dim, "activation": act},
            #         1: {"dims": 2, "activation": act},
            #     }
            # if seek_bias_head_config is None:
            #     seek_bias_head_config = {"dims": 1, "activation": torch.nn.Identity}

            # kernel_mod = SEEKKernel(
            #     cont_cols=cont_cols_seek,
            #     cat_cols=cat_cols,
            #     source_cols=source_cols,
            #     continuous_kernels=continuous_kernels,
            #     use_bias=seek_use_bias,
            #     use_exponential_wrapper=True,  # Set to True to enable exponential wrapper (but may cause outputscale collapse)
            #     weight_layer_config={
            #         0: {"dims": 2, "activation": act},
            #         1: {"dims": 1, "activation": act},
            #     },
            #     bias_layer_config={
            #         0: {"dims": 2, "activation": act},
            #         1: {"dims": 1, "activation": act},
            #     },
            # )

            kernel_mod = SEEKKernelTrunkHead(
                cont_cols=cont_cols_seek,
                cat_cols=cat_cols,
                source_cols=source_cols,
                continuous_kernels=continuous_kernels,
                use_bias=seek_use_bias,
                use_log_scale_kernel=True,
                use_exponential_wrapper=True,
                normalize=True,  # Keep L2 normalization ON for numerical stability and faster convergence
                act=act,
                share_bias_trunk=seek_share_bias_trunk,
                trunk_layer_config={
                    0: {"dims": 1, "activation": act},
                    # 1: {"dims": 2, "activation": act},
                },
                bias_trunk_layer_config={
                    0: {"dims": 1, "activation": act},
                    # 1: {"dims": 2, "activation": act},
                },
                weight_head_configs=[
                    {"dims": 1, "activation": torch.nn.Identity},
                    # {"dims": 1, "activation": torch.nn.Identity},
                ],
                bias_head_config={"dims": 1, "activation": torch.nn.Identity},
                # trunk_layer_config={
                #     0: {"dims": input_dim, "activation": act},
                #     1: {"dims": 2, "activation": act},
                # },
                # weight_head_configs=[
                #     {"dims": 2, "activation": act},
                # ],
                # bias_head_config={"dims": 2, "activation": act},
            )

            # kernel_mod=LogScaleKernel(GaussianKernel(ard_num_dims=input_dim))

            # class TrunkHeadNet(torch.nn.Module):
            #     def __init__(self, trunk: torch.nn.Module, input_head_dim: int, head_config: dict):
            #         """
            #         trunk: an nn.Module mapping inputs to some feature tensor
            #         head_config: {
            #             "dims": int,              # output dimension of the final Linear
            #             "activation": nn.Module   # class of activation, e.g. nn.ReLU or nn.Tanh
            #         }
            #         """
            #         super().__init__()
            #         self.trunk = trunk
            #         self.head = torch.nn.Sequential(
            #             torch.nn.Flatten(start_dim=1),               # flatten (batch, …) → (batch, features)
            #             # torch.nn.LazyLinear(head_config["dims"]),    # infer in_features on first forward
            #             torch.nn.Linear(input_head_dim, head_config["dims"]),
            #             head_config["activation"]()            # e.g. nn.Tanh()
            #         )

            #     def forward(self, x):
            #         x = self.trunk(x)
            #         x = self.head(x)
            #         x = F.normalize(x, p=2, dim=-1, eps=1e-6)
            #         return x
                
                
            # act  = torch.nn.Softplus
            # layer_cfg = {
            #     0: {"dims": 2, "activation": act},
            #     1: {"dims": 4, "activation": act},
            #     # 2: {"dims": 64, "activation": act},
            #     # 2: {"dims": 4,  "activation": torch.nn.Identity},
            # }
            # head_cfg = {"dims": 4,  "activation": torch.nn.Identity}

            # trunk_transform =  gpplus.utils.InputTransformNet(input_dim=input_dim, layer_config=layer_cfg)

            # kernel_mod = gpplus.kernels.ExponentialKernel(
            #     base_kernel = gpplus.kernels.CompositeScaleKernel(input_transform=TrunkHeadNet(trunk=trunk_transform, input_head_dim=4, head_config=head_cfg))
            #                 * LogScaleKernel(GaussianKernel(ard_num_dims=cont_dim))
            #                 # * GaussianKernel(ard_num_dims=cont_dim)
            #                 + gpplus.kernels.CompositeScaleKernel(input_transform=TrunkHeadNet(trunk=trunk_transform, input_head_dim=4, head_config=head_cfg))
            # )
            # Create GP model
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kernel_mod,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_folds - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                print(model)
                
                # Print SEEK kernel weights before training
                if isinstance(kernel_mod, gpplus.kernels.SEEKKernel):
                    print("\n" + "="*80)
                    print("SEEK Kernel Neural Network Weights (BEFORE TRAINING)")
                    print("="*80)
                    seek_kernel = kernel_mod
                    
                    # Print weight kernels
                    for w_idx, weight_kernel in enumerate(seek_kernel.weight_kernels):
                        print(f"\nWeight Kernel {w_idx}:")
                        if hasattr(weight_kernel, 'input_transform'):
                            net = weight_kernel.input_transform.network
                            for layer_idx, layer in enumerate(net):
                                if isinstance(layer, torch.nn.Linear):
                                    print(f"  Layer {layer_idx} (Linear {layer.in_features}->{layer.out_features}):")
                                    w_data = layer.weight.data
                                    print(f"    Weight: min={w_data.min().item():.6f}, max={w_data.max().item():.6f}, "
                                          f"mean={w_data.mean().item():.6f}, std={w_data.std().item():.6f}")
                                    if layer.bias is not None:
                                        b_data = layer.bias.data
                                        print(f"    Bias: min={b_data.min().item():.6f}, max={b_data.max().item():.6f}, "
                                              f"mean={b_data.mean().item():.6f}")
                    
                    # Print bias kernel
                    if seek_kernel.bias_kernel is not None:
                        print(f"\nBias Kernel:")
                        if hasattr(seek_kernel.bias_kernel, 'input_transform'):
                            net = seek_kernel.bias_kernel.input_transform.network
                            for layer_idx, layer in enumerate(net):
                                if isinstance(layer, torch.nn.Linear):
                                    print(f"  Layer {layer_idx} (Linear {layer.in_features}->{layer.out_features}):")
                                    w_data = layer.weight.data
                                    print(f"    Weight: min={w_data.min().item():.6f}, max={w_data.max().item():.6f}, "
                                          f"mean={w_data.mean().item():.6f}, std={w_data.std().item():.6f}")
                                    if layer.bias is not None:
                                        b_data = layer.bias.data
                                        print(f"    Bias: min={b_data.min().item():.6f}, max={b_data.max().item():.6f}, "
                                              f"mean={b_data.mean().item():.6f}")
                    print("="*80 + "\n")

            # Create trainer (train_eval_gp always returns 4 values)
            gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
                model,
                X_test,
                y_test,
                num_epochs=num_epochs,
                seed=fold_seed,
                num_runs=num_runs,
                lr=lr,
                convergence_patience=convergence_patience,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                initializer_class=initializer_class,
                device=gp_device,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,  # Source column is at index 10 (single int = not encoded)
                trainer_info=trainer_info,
            )

            # -----------------------------------------------------------------
            # Print per-kernel parameters when SEEK is used (after training)
            # Works for both SEEKKernel and SEEKKernelTrunkHead, and for any
            # underlying continuous kernel types (Gaussian, Matern, etc.).
            # -----------------------------------------------------------------
            def _print_seek_kernel_params(seek_kernel):
                """Utility to print parameters of each base kernel inside SEEK."""
                from gpplus.kernels.log_scale_kernel import LogScaleKernel
                from gpplus.kernels.mvmf_kernel import MVMFKernel

                print("\nSEEK base kernel parameters (after training):")
                for idx, base in enumerate(seek_kernel.base_kernels):
                    print(f"\n  Base kernel {idx}:")
                    actual = base
                    # Unwrap LogScaleKernel → underlying kernel
                    if isinstance(actual, LogScaleKernel):
                        print("    Wrapper: LogScaleKernel")
                        actual = actual.base_kernel
                    # Unwrap MVMFKernel → continuous kernel if present
                    if isinstance(actual, MVMFKernel) and hasattr(actual, "cont_kernel") and actual.cont_kernel is not None:
                        print("    Wrapper: MVMFKernel (using cont_kernel)")
                        actual = actual.cont_kernel
                    print(f"    Type: {actual.__class__.__name__}")
                    # Lengthscale(s)
                    if hasattr(actual, "raw_lengthscale") and actual.raw_lengthscale is not None:
                        ls = actual.raw_lengthscale.detach().cpu().flatten().tolist()
                        print(f"    raw_lengthscale (log-space): {ls}")
                    # Power parameter (for PowerExponentialKernel)
                    if hasattr(actual, "raw_power") and actual.raw_power is not None:
                        power = actual.raw_power.detach().cpu().item()
                        # Also show transformed power if available
                        if hasattr(actual, "power"):
                            power_transformed = actual.power.detach().cpu().item()
                            print(f"    raw_power: {power:.6f} (transformed: {power_transformed:.6f})")
                        else:
                            print(f"    raw_power: {power:.6f}")
                    # Outputscale (for LogScaleKernel already handled above)
                    if hasattr(base, "raw_outputscale") and base.raw_outputscale is not None:
                        os = base.raw_outputscale.detach().cpu().item()
                        print(f"    raw_outputscale (log-space): {os}")

            if isinstance(kernel_mod, (gpplus.kernels.SEEKKernel, gpplus.kernels.SEEKKernelTrunkHead)):
                _print_seek_kernel_params(kernel_mod)

            # Print SEEK kernel weights AFTER training
            if (i == 0) or (i == num_folds - 1):
                if isinstance(kernel_mod, gpplus.kernels.SEEKKernel):
                    print("\n" + "="*80)
                    print("SEEK Kernel Neural Network Weights (AFTER TRAINING)")
                    print("="*80)
                    seek_kernel = kernel_mod
                    
                    # Print weight kernels
                    for w_idx, weight_kernel in enumerate(seek_kernel.weight_kernels):
                        print(f"\nWeight Kernel {w_idx}:")
                        if hasattr(weight_kernel, 'input_transform'):
                            net = weight_kernel.input_transform.network
                            for layer_idx, layer in enumerate(net):
                                if isinstance(layer, torch.nn.Linear):
                                    print(f"  Layer {layer_idx} (Linear {layer.in_features}->{layer.out_features}):")
                                    w_data = layer.weight.data
                                    print(f"    Weight: min={w_data.min().item():.6f}, max={w_data.max().item():.6f}, "
                                          f"mean={w_data.mean().item():.6f}, std={w_data.std().item():.6f}")
                                    if layer.bias is not None:
                                        b_data = layer.bias.data
                                        print(f"    Bias: min={b_data.min().item():.6f}, max={b_data.max().item():.6f}, "
                                              f"mean={b_data.mean().item():.6f}")
                    
                    # Print bias kernel
                    if seek_kernel.bias_kernel is not None:
                        print(f"\nBias Kernel:")
                        if hasattr(seek_kernel.bias_kernel, 'input_transform'):
                            net = seek_kernel.bias_kernel.input_transform.network
                            for layer_idx, layer in enumerate(net):
                                if isinstance(layer, torch.nn.Linear):
                                    print(f"  Layer {layer_idx} (Linear {layer.in_features}->{layer.out_features}):")
                                    w_data = layer.weight.data
                                    print(f"    Weight: min={w_data.min().item():.6f}, max={w_data.max().item():.6f}, "
                                          f"mean={w_data.mean().item():.6f}, std={w_data.std().item():.6f}")
                                    if layer.bias is not None:
                                        b_data = layer.bias.data
                                        print(f"    Bias: min={b_data.min().item():.6f}, max={b_data.max().item():.6f}, "
                                              f"mean={b_data.mean().item():.6f}")
                    
                    # Print weight kernel outputs on training data
                    print("\n--- Weight Kernel Outputs on Training Data ---")
                    model.eval()
                    with torch.no_grad():
                        if seek_kernel.use_mvmf:
                            x_encoded, _ = seek_kernel._encode_for_weights(X_train, seek_kernel.base_kernels[0])
                        else:
                            x_encoded = X_train[:, seek_kernel.cont_cols] if seek_kernel.cont_cols else X_train
                        
                        for w_idx, weight_kernel in enumerate(seek_kernel.weight_kernels):
                            k_weight = weight_kernel(x_encoded, x_encoded)
                            if hasattr(k_weight, 'to_dense'):
                                k_weight = k_weight.to_dense()
                            print(f"  Weight Kernel {w_idx} output: min={k_weight.min().item():.6f}, "
                                  f"max={k_weight.max().item():.6f}, mean={k_weight.mean().item():.6f}, "
                                  f"diag_mean={torch.diag(k_weight).mean().item():.6f}, "
                                  f"%negative={(k_weight < 0).float().mean().item() * 100:.1f}%")
                        
                        if seek_kernel.bias_kernel is not None:
                            k_bias = seek_kernel.bias_kernel(x_encoded, x_encoded)
                            if hasattr(k_bias, 'to_dense'):
                                k_bias = k_bias.to_dense()
                            print(f"  Bias Kernel output: min={k_bias.min().item():.6f}, "
                                  f"max={k_bias.max().item():.6f}, mean={k_bias.mean().item():.6f}, "
                                  f"diag_mean={torch.diag(k_bias).mean().item():.6f}, "
                                  f"%negative={(k_bias < 0).float().mean().item() * 100:.1f}%")
                    
                    print("="*80 + "\n")

            # Record and print GP metrics (inside the GP branch so gp_metric is always defined)
            GPPlus_metrics.append(gp_metric)
            if gp_trainer_info:
                gp_trainer_info["fold"] = i + 1
                gp_trainer_info["metrics"] = gp_metric
                GPTrainer_info.append(gp_trainer_info)

        # Always print GP results for this fold, including jitter and jitter_max if present
        print(f"\nGP Results (Fold {i+1}/{num_folds})")
        for k, v in gp_metric.items():
            if isinstance(v, (int, float, np.floating)):
                if np.isnan(v):
                    print(f"  {k}: NaN")
                else:
                    # Use scientific notation for jitter/jitter_max/noise via shared formatter
                    print(f"  {k}: {format_metric_value(str(k), float(v), precision=4)}")
            else:
                print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_models in [None, "pfn"]:
            print(f"\n--- {title} TabPFN Training ---")

            if regressor is None:
                raise RuntimeError(
                    "TabPFN requested (run_models=None/'pfn') but `regressor` is None. "
                    "Instantiate a TabPFN regressor (e.g. VanillaDirectTabPFNRegressor) and pass it in."
                )
            
            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
                X_train,
                X_test,
                y_train_normal if standardize_y else y_train,
                y_test,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,
            )
            
            TabPFN_metrics.append(tabpfn_metric)

            # Print results for this fold
            print(f"\nTabPFN Results (Fold {i+1}/{num_folds})")
            for k, v in tabpfn_metric.items():
                if isinstance(v, (int, float, np.floating)):
                    if np.isnan(v):
                        print(f"  {k}: NaN")
                    else:
                        print(f"  {k}: {format_metric_value(str(k), float(v), precision=4)}")
                else:
                    print(f"  {k}: {v}")
        
        # Collect model info from first fold
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            model_params_dict = {}
            if run_models in [None, "gp"] and GPPlus_metrics:
                # Extract model parameters from gp_metric (lengthscales, outputscale, noise, jitter, etc.)
                model_params_dict = {}
                # Include jitter_max so we can see the maximum jitter used in the best run
                param_keys = ["lengthscale", "outputscale", "noise", "jitter", "jitter_max", "raw_noise"]
                for key, value in gp_metric.items():
                    # Include any key that contains parameter names
                    if any(param_key in key.lower() for param_key in param_keys):
                        model_params_dict[key] = value
                    # Also include best_epoch if present
                    elif key in ["best_epoch", "best_loss"]:
                        model_params_dict[key] = value

            # Extract weights and biases from SEEKKernel neural networks
            def extract_nn_params(network, prefix=""):
                """Extract weights and biases from a Sequential network."""
                params = {}
                layer_idx = 0
                for module in network:
                    if isinstance(module, torch.nn.Linear):
                        weight_key = f"{prefix}layer_{layer_idx}_weight"
                        bias_key = f"{prefix}layer_{layer_idx}_bias"
                        params[weight_key] = module.weight.detach().cpu().numpy().tolist()
                        if module.bias is not None:
                            params[bias_key] = module.bias.detach().cpu().numpy().tolist()
                        layer_idx += 1
                return params

            # SEEK-specific parameter extraction (commented out since using different kernel)
            # seek_nn_params = {}
            # # Extract from weight kernels
            # for weight_idx, weight_kernel in enumerate(seek_kernel.weight_kernels):
            #     weight_prefix = f"weight_kernel_{weight_idx}_"
            #     weight_params = extract_nn_params(weight_kernel.input_transform.network, prefix=weight_prefix)
            #     seek_nn_params.update(weight_params)
            # 
            # # Extract from bias kernel if present
            # if seek_kernel.bias_kernel is not None:
            #     bias_prefix = "bias_kernel_"
            #     bias_params = extract_nn_params(seek_kernel.bias_kernel.input_transform.network, prefix=bias_prefix)
            #     seek_nn_params.update(bias_params)
            # 
            # # Print weights and biases summary
            # print(f"\n--- SEEK Neural Network Parameters (Fold {i+1}) ---")
            # for key, value in seek_nn_params.items():
            #     if isinstance(value, list) and len(value) > 0:
            #         arr = np.array(value)
            #         print(f"  {key}: shape {arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
            #     else:
            #         print(f"  {key}: {value}")

            gp_model_info = None
            tabpfn_model_info = None
            if run_models in [None, "gp"]:
                gp_model_info = {
                    "model_str": str(model),
                    "kernel_type": kernel_type,
                    "kernel": "SEEKKernel" if kernel_type is None else f"SEEKKernel_base={kernel_type}",
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "source_cols": source_cols,
                    "qual_dict": qual_dict,
                    "input_dim": X_train.shape[1],
                    "train_samples": int(train_per_fold),
                    "test_samples": num_test,
                    "standardize_X": standardize_X,
                    "standardize_y": standardize_y,
                    "x_standardize_method": x_standardize_method,
                    "X_scaling_type": X_scaling_type,
                    "dtype": str(gp_dtype),
                    "device": str(gp_device),
                    "num_epochs": num_epochs,
                    "num_runs": num_runs,
                    "lr": lr,
                    "optimizer": optimizer_class.__name__,
                    "convergence_patience": convergence_patience,
                    "min_loss_change": min_loss_change,
                    "initializer": initializer_class.__name__ if initializer_class else None,
                    **y_test_stats,
                    "num_folds": num_folds,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                    **model_params_dict,
                }

            if run_models in [None, "pfn"] and regressor is not None:
                tabpfn_model_info = {
                    "model_path": regressor.model_path,
                    "fit_mode": regressor.fit_mode,
                    "device": str(regressor.device_),
                    "inference_precision": regressor.inference_precision,
                    "random_state": regressor.random_state,
                    "use_autocast": regressor.use_autocast_,
                    "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
                }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_summary = (
        analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
        if run_models in [None, "pfn"]
        else None
    )
    GPPlus_summary = (
        analyze_metrics(GPPlus_metrics, print_summary=True, label="GP(SEEK)", title=title)
        if run_models in [None, "gp"]
        else None
    )

    if save_path is not None:
        if run_models is None:
            plot_metrics(
                TabPFN_metrics,
                GPPlus_metrics,
                labels=["TabPFN", "GP(SEEK)"],
                title=title,
                save_path=plot_save_path,
            )

        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            file_prefix = run_models if run_models is not None else "gpVpfn"

            combined_data: dict = {}
            if run_models in [None, "gp"]:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                }
                if trainer_info and GPTrainer_info:
                    combined_data["gp_trainer_info"] = GPTrainer_info

            if run_models in [None, "pfn"]:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info,
                }

            # Append defaults.py source for reproducibility (same folder as this script)
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")

            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("=" * 60)
    print(
        "Trainer details: "
        f"\n\tnumber of epochs: {num_epochs}"
        f"\n\tnumber of runs: {num_runs}"
        f"\n\tlearning rate: {lr}"
        f"\n\toptimizer: {optimizer_class}"
        f"\n\tconvergence patience: {convergence_patience}"
        f"\n\tdevice: {gp_device}"
        f"\n\tinitializer: {initializer_class}"
        f"\n\tcont_cols: {cont_cols}"
        f"\n\tcat_cols: {cat_cols}"
        f"\n\tsource_cols: {source_cols}"
        f"\n\tX_standardize: {standardize_X}"
        f"\n\tX_scaling_type: {X_scaling_type}"
        f"\n\ty_standardize: {standardize_y}"
    )
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Setup profiling
    profile_output_dir = Path("./profile_results")
    profile_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_file = profile_output_dir / f"seek_profile_{timestamp}.prof"
    
    print(f"Starting profiling... Profile will be saved to: {profile_file}")
    print("="*60)
    
    # Create profiler and run the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # wing_SF_GPvsPFN(
        #     num_folds=1,
        #     train_size=1,
        #     num_test=5,
        #     noise_train=0.0,
        #     noise_test=0.0,
        #     num_runs=1,
        #     num_epochs=1,
        #     save_path="./results/wing/temp_warmup",
        #     # seek_weight_trunk_layer_config=None,
        #     # seek_weight_head_config=None,
        #     # seek_bias_trunk_layer_config=None,
        #     # seek_bias_head_config=None,
        #     seek_use_bias=False,
        #     run_models="gp",
        #     # gp_device="cuda",
        # )
        wing_SF_GPvsPFN(
            num_folds=1,
            train_size=1,
            num_test=5000,
            noise_train=0.005,
            noise_test=0.005,
            num_runs=16,
            num_epochs=100,
            save_path="./results/wing/SEEK_Test/SEEK_tests",
            # seek_weight_trunk_layer_config=None,
            # seek_weight_head_config=None,
            # seek_bias_trunk_layer_config=None,
            # seek_bias_head_config=None,
            seek_use_bias=True,
            seek_share_bias_trunk=True,
            run_models="gp",
            # gp_device="cuda",
            # optimizer_class=torch.optim.Adam,
        )
    finally:
        profiler.disable()
        
        # # Save profile to file
        # profiler.dump_stats(str(profile_file))
        # print(f"\n{'='*60}")
        # print(f"Profile saved to: {profile_file}")
        # print(f"{'='*60}")
        
        # # Print top 30 functions by cumulative time
        # print("\nTop 30 functions by cumulative time:")
        # print("-" * 60)
        # stats = pstats.Stats(profiler)
        # stats.sort_stats('cumulative')
        # stats.print_stats(30)
        
        # # Print top 30 functions by total time
        # print("\nTop 30 functions by total time:")
        # print("-" * 60)
        # stats.sort_stats('tottime')
        # stats.print_stats(30)
        
        # print(f"\n{'='*60}")
        # print("To visualize with snakeviz, run:")
        # print(f"  snakeviz {profile_file}")
        # print(f"{'='*60}")
