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
import time
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.utils.relative_distance_encoder import RelativeDistanceEncoder
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from experiments.data.load_experimental_data import load_m2ax_data
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils.relative_distance_encoder import RelativeDistanceEncoder
# import warnings
# warnings.filterwarnings("ignore")
def M2AX_GPvsPFN(
        num_seeds=20,
        test_size=0.2, 
        num_runs=16, 
        num_epochs=10000, 
        lr=0.1, 
        convergence_patience=10,
        optimizer_class=torch.optim.Adam,
        initializer_class=None,
        gp_device='cpu',
        amp_device='cuda',
        encode_PFN_data=True, # In this problem, 
        save_path='./results/M2AX',
        title=None,
        # standardize_X_gp=False, # Not supported for this problem, all categorical
        standardize_y_gp=True, # True gives best results for PFN
    ):

    if title is None:
        title = f"M2AX_{test_size}tsize_{num_epochs}epochs_{num_runs}runs_{lr}"
    else: 
        title = f"M2AX_{title}"
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
    print(f"{title}: TabPFN vs GP Comparison")
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
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()
        
        X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=seed)
    
        X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        # =============================================================================
        # GP M2AX Section 
        # =============================================================================
        print(f"\n--- {title} GP Evaluation ---")

        # Reuse PFN split, convert to torch
        X_gp_train = X_train_enc.detach().clone().to(dtype=dtype)
        X_gp_test = X_test_enc.detach().clone().to(dtype=dtype)
        y_gp_train = y_train.detach().clone().to(dtype=dtype)
        y_gp_test = y_test.detach().clone().to(dtype=dtype)

        # Normalize the GP data
        # y_gp_train_mean = y_gp_train.mean()
        # y_gp_train_std = y_gp_train.std()
        # y_gp_train_normal = (y_gp_train - y_gp_train_mean) / y_gp_train_std
        Y_scaler = gpplus.utils.StandardScaler()
        Y_scaler.fit(y_gp_train)
        # Squeeze to remove extra dimensions from keepdim=True in StandardScaler
        y_gp_train_mean = Y_scaler.mean.squeeze()
        y_gp_train_std = Y_scaler.std.squeeze()
        y_gp_train_normal = Y_scaler.transform(y_gp_train)
        y_gp_test_normal = Y_scaler.transform(y_gp_test)

        cat_encoders = 'matrix'
        cont_cols = [22]

        kernel = gpplus.kernels.LogScaleKernel(
            gpplus.kernels.MVMFKernel(
                cat_cols=cat_cols,
                cont_cols=cont_cols,
                cat_encoder=cat_encoders,
                z_dim=2,
            )
        )

        # Create GP model
        model = gpplus.models.GPR(
            X_gp_train,
            y_gp_train_normal if standardize_y_gp else y_gp_train,
            kernel_module=kernel,
            # likelihood=gpplus.likelihoods.LogGaussianLikelihood(
            #     # noise_constraint=gpytorch.constraints.GreaterThan(1e-6), 
            #     noise_prior=gpytorch.priors.LogNormalPrior(loc=np.log(y_gp_train_std**2), scale=1.0)),

        )
        if (i == 0) or (i == len(seeds) - 1):
            print(model)

        # Create trainer
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp(
            model,
            X_gp_test,
            y_gp_test,
            num_epochs=num_epochs,
            seed=seed,
            num_runs=num_runs,
            lr=lr,
            convergence_patience=convergence_patience,
            optimizer_class=optimizer_class,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_gp_train_mean if standardize_y_gp else None,
            y_train_std=y_gp_train_std if standardize_y_gp else None,
            # source_cols=source_cols,
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
        from gpplus.utils.metrics_functions import format_metric_value
        for k, v in processed_gp_metric.items():
            print(f"  {k}: {format_metric_value(k, v)}")

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
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None

                # Collect TabPFN model information
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
    print(f"\nTotal experiment time for {len(seeds)} seeds: {time.time() - t0:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\ty_standardize: {standardize_y_gp}")
    print(f"Experiment details: \n\ttest size: {test_size} ({len(X_test)} test samples, {len(X_train)} train samples)\n\tseeds: {seeds}")

    return GPPlus_M2AX_metrics, TabPFN_M2AX_metrics,

if __name__ == "__main__":
    # M2AX_GPvsPFN()
    M2AX_GPvsPFN(num_seeds=1, num_runs=4, test_size=0.2, num_epochs=100, optimizer_class=LBFGSScipy, save_path='./results/M2AX/TestStandardizeY')
    # M2AX_GPvsPFN(num_seeds=2, num_runs=4, num_epochs=10000, save_path=None, encode_PFN_data=True)