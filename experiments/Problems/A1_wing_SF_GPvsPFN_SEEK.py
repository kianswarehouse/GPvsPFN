import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
import cProfile
import pstats
from datetime import datetime
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import generate_mf_wing_data
import defaults

# SEEK kernel (the only substantive change vs Kian baseline)
from gpplus.kernels import SEEKKernel, GaussianKernel, LogScaleKernel, ExponentialKernel

def wing_SF_GPvsPFN(
    num_folds: int = defaults.NUM_FOLDS,
    num_test: int = 5000,
    train_size: int = 10,  # total training size is train_size * input_dim
    num_runs: int = defaults.TRAINER_NUM_RUNS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float = defaults.TRAINER_LR,
    convergence_patience: int = defaults.TRAINER_CONVERGENCE_PATIENCE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    save_path: str | None = "./results/wing",
    title: str | None = None,
    standardize_X: bool = True,
    standardize_y: bool = True,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    pfn_dtype: torch.dtype = defaults.DTYPE_PFN,
    *,
    seek_use_bias: bool = True,
    seek_weight_layer_config: dict | None = None,
    seek_bias_layer_config: dict | None = None,
):
    """Single-fidelity Wing: TabPFN vs GPPlus with SEEKKernel.

    Notes:
    - Data is generated from the MF wing generator but we drop the source column (s0 only).
    - The GP covariance is SEEKKernel with an ensemble of continuous base kernels.
    """

    if title is None:
        title = f"wing_SF_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"wing_SF_{title}_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"

    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
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

    # Randomize across the single source (s0), then split across folds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)
        
    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]

        X_train = X_train_all[fold_train_indices]
        # X_enc_train = X_enc_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section (SEEK)
        # =============================================================================
        print(f"\n--- {title} GP(SEEK) Training ---")

        # Reuse PFN split, convert to torch
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        if standardize_X:
            Xscaler = gpplus.utils.StandardScaler()
            Xscaler.fit(X_train)
            X_train = Xscaler.transform(X_train)
            X_test = Xscaler.transform(X_test)


        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        # --- SEEK kernel definition (only substantive change vs Kian baseline) ---
        cont_cols_seek = cont_cols
        if cont_cols_seek is None or len(cont_cols_seek) == 0:
            cont_cols_seek = list(range(X_train.shape[1]))
        cont_dim = len(cont_cols_seek)

        # Pass unwrapped continuous kernels - LogScaleKernel will wrap each base kernel
        # continuous_kernels = [
        #     GaussianKernel(ard_num_dims=cont_dim),
        # ]
        
        # act_func = torch.nn.Tanh
        # # Default weight/bias layer configs if not provided
        # if seek_weight_layer_config is None:
        #     seek_weight_layer_config = {
        #         0: {"dims": 2*input_dim, "activation": act_func},
        #         1: {"dims": 2*input_dim, "activation": act_func},
        #         2: {"dims": 2, "activation": torch.nn.Identity},  # Output between 0 and 1
        #     }
        # if seek_bias_layer_config is None:
        #     seek_bias_layer_config = {
        #         0: {"dims": 2*input_dim, "activation": act_func},
        #         1: {"dims": 2*input_dim, "activation": act_func},
        #         2: {"dims": 2, "activation": torch.nn.Identity},  # Output between -1 and 1
        #     }
        
        # seek_kernel = SEEKKernel(
        #     cont_cols=cont_cols_seek,
        #     cat_cols=cat_cols,
        #     source_cols=source_cols,
        #     continuous_kernels=continuous_kernels,
        #     weight_layer_config=seek_weight_layer_config,
        #     bias_layer_config=seek_bias_layer_config,
        #     use_bias=seek_use_bias,
        #     activation=ExponentialKernel,  # Use ExponentialKernel from gpplus as activation
        # )
        
        class TrunkHeadNet(torch.nn.Module):
            def __init__(self, trunk: torch.nn.Module, input_head_dim: int, head_config: dict):
                """
                trunk: an nn.Module mapping inputs to some feature tensor
                head_config: {
                    "dims": int,              # output dimension of the final Linear
                    "activation": nn.Module   # class of activation, e.g. nn.ReLU or nn.Tanh
                }
                """
                super().__init__()
                self.trunk = trunk
                self.head = torch.nn.Sequential(
                    torch.nn.Flatten(start_dim=1),               # flatten (batch, …) → (batch, features)
                    # torch.nn.LazyLinear(head_config["dims"]),    # infer in_features on first forward
                    torch.nn.Linear(input_head_dim, head_config["dims"]),
                    head_config["activation"]()            # e.g. nn.Tanh()
                )

            def forward(self, x):
                x = self.trunk(x)
                x = self.head(x)
                x = F.normalize(x, p=2, dim=-1, eps=1e-6)
                return x
    
    
        act  = torch.nn.Softplus
        layer_cfg = {
            0: {"dims": input_dim, "activation": act},
            1: {"dims": 2, "activation": act},
            # 2: {"dims": 2, "activation": act},
            # 2: {"dims": 4,  "activation": torch.nn.Identity},
        }
        head_cfg = {"dims": 2,  "activation": torch.nn.Identity}

        trunk_transform =  gpplus.utils.InputTransformNet(input_dim=input_dim, layer_config=layer_cfg)

        kernel = gpplus.kernels.ExponentialKernel(
            base_kernel = gpplus.kernels.CompositeScaleKernel(input_transform=TrunkHeadNet(trunk=trunk_transform, input_head_dim=2, head_config=head_cfg))
                        * gpplus.kernels.LogScaleKernel(gpplus.kernels.GaussianKernel(ard_num_dims=cont_dim))
                        + gpplus.kernels.CompositeScaleKernel(input_transform=TrunkHeadNet(trunk=trunk_transform, input_head_dim=2, head_config=head_cfg))
        )
        # Create GP model
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=kernel,
            mean_module=defaults.SF_mean,
            likelihood=defaults.SF_likelihood,
        )
        if (i == 0) or (i == num_folds - 1):
            print(f"X_train: {X_train.shape}")
            print(f"X_test: {X_test.shape}")
            print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
            print(model)

        # Create trainer
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            seed=seed_trainer,
            num_runs=num_runs,
            lr=lr,
            convergence_patience=convergence_patience,
            min_loss_change=1e-7,
            optimizer_class=optimizer_class,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,  # Source column is at index 10 (single int = not encoded)
        )
        
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Fold {i+1}/{num_folds})")
        for k, v in gp_metric.items():
            if isinstance(v, (int, float, np.floating)):
                if np.isnan(v):
                    print(f"  {k}: NaN")
                else:
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")
        
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
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        # Collect model info from first fold
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            # Extract model parameters from gp_metric (lengthscales, outputscale, noise, jitter, etc.)
            model_params_dict = {}
            param_keys = ["lengthscale", "outputscale", "noise", "jitter", "raw_noise"]
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

            gp_model_info = {
                "model_str": str(model),
                "kernel": "ExponentialKernel_with_CompositeScaleKernel",  # Updated kernel name
                # "seek_use_bias": seek_use_bias,  # Not applicable for current kernel
                # "seek_num_base_kernels": len(continuous_kernels),  # Not applicable for current kernel
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": int(train_per_fold),
                "test_samples": num_test,
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                **y_test_stats,
                "num_folds": num_folds,
                "seed": seed,
                "seed_trainer": seed_trainer,
                **model_params_dict,  # Add all model parameter keys from gp_metric (lengthscales, outputscale, noise, etc.)
                # "seek_nn_params": seek_nn_params,  # Not applicable for current kernel
            }
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
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP(SEEK)", title=title)

    if save_path is not None:
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
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {
                "gp_data": {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info
                },
                "tabpfn_data": {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info
                },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
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
        wing_SF_GPvsPFN(
            num_folds=1,
            train_size=10,
            num_test=5000,
            noise_train=0.0,
            noise_test=0.0,
            num_runs=16,
            num_epochs=100,
            save_path="./results/wing/SEEKtemp",
            seek_weight_layer_config=None,
            seek_bias_layer_config=None,
            seek_use_bias=True,
            # gp_device="cuda",
        )
    finally:
        profiler.disable()
        
        # Save profile to file
        profiler.dump_stats(str(profile_file))
        print(f"\n{'='*60}")
        print(f"Profile saved to: {profile_file}")
        print(f"{'='*60}")
        
        # Print top 30 functions by cumulative time
        print("\nTop 30 functions by cumulative time:")
        print("-" * 60)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(30)
        
        # Print top 30 functions by total time
        print("\nTop 30 functions by total time:")
        print("-" * 60)
        stats.sort_stats('tottime')
        stats.print_stats(30)
        
        print(f"\n{'='*60}")
        print("To visualize with snakeviz, run:")
        print(f"  snakeviz {profile_file}")
        print(f"{'='*60}")
