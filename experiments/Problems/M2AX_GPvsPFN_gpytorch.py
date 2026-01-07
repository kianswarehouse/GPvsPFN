import torch
import os
import sys
# Add root directory to path to allow imports from any directory
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import gpytorch
import linear_operator
from linear_operator.utils.errors import NotPSDError, NanError
import time
import copy
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics, compute_metrics
from gpplus.utils import set_seed, train_eval_PFN
from gpplus.utils.relative_distance_encoder import RelativeDistanceEncoder
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from experiments.data.load_experimental_data import load_m2ax_data
import defaults_gpytorch as defaults
from gpytorch_train_eval import train_eval_gp_gpytorch_default
# import warnings
# warnings.filterwarnings("ignore")

class ExactGPModel(gpytorch.models.ExactGP):
    """Pure gpytorch ExactGP model for regression."""
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def M2AX_GPvsPFN(
        num_seeds=20,
        test_size=0.2, 
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        encode_PFN_data=True, # In this problem, 
        save_path='./results/M2AX',
        title=None,
        # standardize_X_gp=False, # Not supported for this problem, all categorical
        standardize_y_gp=True, # True gives best results for PFN
    ):

    if title is None:
        title = f"M2AX_gpytorch_{test_size}tsize_{num_epochs}epochs_{num_runs}runs_{lr}"
    else: 
        title = f"M2AX_gpytorch_{title}"
    # Load the data
    X, y = load_m2ax_data()
    
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None
        
    amp_dtype = torch.float32
    dtype = torch.float64
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)

    # Initialize results storage

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison (GPyTorch LBFGS)")
    print("="*10)
    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    # print(X.shape)
    qual_dict = learn_encodings(X[:,[0,1]])
    # qual_dict = {0: 10, 1:12}
    X_enc, cont_cols, cat_cols, source_cols = encode_qual_data(X[:,[0,1]], qual_dict=qual_dict)
    # print(X_enc.shape)
    # print(X_enc[:5])
    X_enc = torch.concatenate([X_enc, X[:,[2]]], axis=1)
    # print(X_enc.shape)
    # print(X_enc[:100])
    # print(qual_dict)
    # print(cat_cols)
    TabPFN_M2AX_metrics = []
    GPPlus_M2AX_metrics = []
    set_seed(0)  # Set global seed for reproducible seeds
    seeds = np.random.RandomState(0).choice(10**6, size=num_seeds, replace=False).tolist()
    
    total_start_time = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()
        
        X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=seed)
    
        X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        # =============================================================================
        # GP M2AX Section 
        # =============================================================================
        print(f"\n--- {title} GP Evaluation (GPyTorch LBFGS) ---")

        # Reuse PFN split, convert to torch
        X_gp_train = X_train_enc.detach().clone().to(dtype=dtype)
        X_gp_test = X_test_enc.detach().clone().to(dtype=dtype)
        y_gp_train = y_train.detach().clone().to(dtype=dtype)
        y_gp_test = y_test.detach().clone().to(dtype=dtype)

        # Normalize the GP data
        Y_scaler = gpplus.utils.StandardScaler()
        Y_scaler.fit(y_gp_train)
        # Squeeze to remove extra dimensions from keepdim=True in StandardScaler
        y_gp_train_mean = Y_scaler.mean.squeeze()
        y_gp_train_std = Y_scaler.std.squeeze()
        y_gp_train_normal = Y_scaler.transform(y_gp_train)
        y_gp_test_normal = Y_scaler.transform(y_gp_test)

        # Create GP model using pure gpytorch components
        # Use gpytorch defaults: GaussianLikelihood, ScaleKernel(RBFKernel), ConstantMean
        input_dim = X_gp_train.shape[-1]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )
        
        # Ensure components use the correct dtype
        likelihood = likelihood.to(dtype=dtype)
        mean_module = mean_module.to(dtype=dtype)
        covar_module = covar_module.to(dtype=dtype)
        
        model = ExactGPModel(
            X_gp_train,
            y_gp_train_normal if standardize_y_gp else y_gp_train,
            likelihood,
            mean_module,
            covar_module
        )
        
        if (i == 0) or (i == len(seeds) - 1):
            print(model)

        # Train with gpytorch defaults
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp_gpytorch_default(
            model,
            X_gp_test,
            y_gp_test,
            num_epochs=num_epochs,
            num_runs=num_runs,
            seed=seed,
            device=gp_device,
            y_train_mean=y_gp_train_mean if standardize_y_gp else None,
            y_train_std=y_gp_train_std if standardize_y_gp else None,
            convergence_patience=convergence_patience,
        )
        
        # Collect metrics in a list first (like wFolds version does) to ensure consistent processing
        seed_gp_metrics = [gp_metric]
        
        # Process metrics the same way as wFolds version (even though we only have one entry)
        # This ensures the same code path is used for lengthscale tracking
        processed_gp_metric = {}
        if seed_gp_metrics:
            for key in seed_gp_metrics[0].keys():
                processed_gp_metric[key] = seed_gp_metrics[0][key]
        
        GPPlus_M2AX_metrics.append(processed_gp_metric)

        print(f"\nGP Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        # Check if this seed had failures
        if processed_gp_metric.get("all_runs_failed", False):
            print(f"  ⚠️  WARNING: All training runs failed for this seed!")
        elif processed_gp_metric.get("all_metrics_nan", False):
            print(f"  ⚠️  WARNING: All metrics are NaN for this seed!")
        elif "evaluation_error" in processed_gp_metric:
            print(f"  ⚠️  WARNING: Evaluation error: {processed_gp_metric['evaluation_error']}")
        
        from gpplus.utils.metrics_functions import format_metric_value
        for k, v in processed_gp_metric.items():
            if k not in ["evaluation_error", "all_runs_failed", "all_metrics_nan"]:
                if isinstance(v, (int, float)):
                    if np.isnan(v):
                        print(f"  {k}: NaN")
                    else:
                        print(f"  {k}: {format_metric_value(k, v)}")
                else:
                    print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Evaluation ---")
        
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_train_enc if encode_PFN_data else X_train,
            X_test_enc if encode_PFN_data else X_test,
            y_gp_train_normal if standardize_y_gp else y_gp_train,
            y_gp_test_normal if standardize_y_gp else y_gp_test,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            regressor=regressor,
            # source_cols=source_cols,
        )
        TabPFN_M2AX_metrics.append(tabpfn_metric)

        # Print results for this seed
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        if i == 0:
            gp_model_info = {
                "model_str": str(model),
                "cont_cols": cont_cols,
                "cat_cols": cat_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_gp_train.shape[1],
                "train_samples": X_gp_train.shape[0],
                "test_samples": X_gp_test.shape[0],
                "y_train_mean": float(y_gp_train_mean.item()),
                "y_train_std": float(y_gp_train_std.item()),
                "standardize_X_gp": "All categorical",
                "standardize_y_gp": standardize_y_gp,
                "dtype": str(dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": 0.1,  # LBFGS learning rate
                "optimizer": "LBFGS",
                "convergence_patience": convergence_patience,
                "initializer": None,  # Not used in gpytorch default training
            }
            tabpfn_model_info = {
                "model_path": regressor.model_path,
                "fit_mode": regressor.fit_mode,
                "device": str(regressor.device_),
                "inference_precision": regressor.inference_precision,
                "random_state": regressor.random_state,
                "use_autocast": regressor.use_autocast_,
                "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
                "encoded_data": encode_PFN_data,
            }

    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Check for seeds with all NaN metrics
    failed_seeds = []
    for idx, metric in enumerate(GPPlus_M2AX_metrics):
        if metric.get("all_runs_failed", False) or metric.get("all_metrics_nan", False):
            failed_seeds.append(idx + 1)
    
    if failed_seeds:
        print(f"\n⚠️  WARNING: {len(failed_seeds)} seed(s) had complete failures: {failed_seeds}")
        print(f"   These seeds will have NaN metrics in the summary.")
        print(f"   Consider investigating data scaling, initialization, or increasing jitter.")

    # Summaries via analyze_metrics
    TabPFN_M2AX_summary = analyze_metrics(TabPFN_M2AX_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_M2AX_summary = analyze_metrics(GPPlus_M2AX_metrics, print_summary=True, label="GP", title=title)
    
    if save_path is not None:
        plot_metrics(TabPFN_M2AX_metrics, GPPlus_M2AX_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {
                "gp_data": {
                    "summary": GPPlus_M2AX_summary,
                    "metrics": GPPlus_M2AX_metrics,
                    "gp_model_info": gp_model_info
                },
                "tabpfn_data": {
                    "summary": TabPFN_M2AX_summary,
                    "metrics": TabPFN_M2AX_metrics,
                    "pfn_model_info": tabpfn_model_info
                },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {len(seeds)} seeds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details (GPyTorch LBFGS): \n\tnumber of epochs: {num_epochs}\n\tlearning rate: 0.1 (LBFGS)\n\toptimizer: LBFGS\n\tdevice: {gp_device}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\ty_standardize: {standardize_y_gp}")
    print(f"Experiment details: \n\ttest size: {test_size} ({len(X_test)} test samples, {len(X_train)} train samples)\n\tseeds: {seeds}")

    return GPPlus_M2AX_metrics, TabPFN_M2AX_metrics

if __name__ == "__main__":
    # M2AX_GPvsPFN()
    M2AX_GPvsPFN(num_seeds=1, num_runs=4, test_size=0.2, num_epochs=100, save_path='./results/M2AX/TestStandardizeY', gp_device='cpu', amp_device='cpu')
    # M2AX_GPvsPFN(num_seeds=2, num_runs=4, num_epochs=10000, save_path=None, encode_PFN_data=True)
