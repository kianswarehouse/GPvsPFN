import torch
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import gpytorch
import time
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import wing_mixed_variables, generate_mf_wing_data


# import warnings
# warnings.filterwarnings("ignore")
def wing_SF_GPvsPFN(num_seeds=20,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions
        num_runs=16, 
        num_epochs=10000, 
        lr=0.1, 
        convergence_patience=10,
        optimizer_class=torch.optim.Adam,
        initializer_class=None,
        gp_device='cpu',
        amp_device='cuda',
        save_path='./results/wing',
        title=None,
        standardize_X=True,
        standardize_y=True,
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
    ):
    if title is None:
        title = f"wing_SF_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"wing_SF_{train_size}D_{title}"
    
    
    amp_dtype = torch.float32
    dtype = torch.float64
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Generate data
    set_seed(0)
    
    # Calculate total samples needed
    train_per_seed = train_size * 10
    total_train = num_seeds * train_per_seed
    total_samples = num_test + total_train
    
    # Generate all unique Sobol samples at once
    print(f"Generating {total_samples} unique Sobol samples...")
    X_train_all, y_train_all, X_test_all, y_test_all = generate_mf_wing_data(
        train_samples_per_source=[total_train, 0, 0, 0], 
        test_samples_per_source=[num_test, 0, 0, 0], 
        train_noise=noise_train, 
        test_noise=noise_test, 
        noise_type=noise_type
    )
    # Drop the 11th (source) column since SF uses only s0
    if X_train_all.shape[1] == 11:
        X_train_all = X_train_all[:, :10]
    if X_test_all.shape[1] == 11:
        X_test_all = X_test_all[:, :10]
    
    
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_train_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    # X_enc_test, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Randomize across the single source (s0), then split across seeds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_seeds, train_per_seed)
        
    total_start_time = time.time()
    for i in range(num_seeds):
        seed = i  # deterministic seed per split
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()

        # Get training indices for this seed
        seed_train_indices = train_indices_2d[i]

        X_train = X_train_all[seed_train_indices]
        # X_enc_train = X_enc_train_all[seed_train_indices]
        y_train = y_train_all[seed_train_indices]

        # Single-fidelity: no source column

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Reuse PFN split, convert to torch
        X_train = X_train.detach().clone().to(dtype=dtype)
        X_test = X_test_all.detach().clone().to(dtype=dtype)
        y_train = y_train.detach().clone().to(dtype=dtype)
        y_test = y_test_all.detach().clone().to(dtype=dtype)
        if standardize_X:
            Xscaler = gpplus.utils.StandardScaler()
            Xscaler.fit(X_train)
            X_train = Xscaler.transform(X_train)
            X_test = Xscaler.transform(X_test)

        # Normalize the GP data
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train_normal = (y_train - y_train_mean) / y_train_std

        # Create GP model
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            # kernel_module=kernel,
        )
        if (i == 0) or (i == num_seeds - 1):
            print(model)

        # Create trainer
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            seed=seed,
            num_runs=num_runs,
            lr=lr,
            convergence_patience=convergence_patience,
            optimizer_class=optimizer_class,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,  # Source column is at index 10 (single int = not encoded)
        )
        
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in gp_metric.items():
            print(f"  {k}: {v:.4f}")

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
            amp_dtype=amp_dtype,
            regressor=regressor,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,
        )
        
        TabPFN_metrics.append(tabpfn_metric)

        # Print results for this seed
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        # Collect model info from first seed
        if i == 0:
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": int(train_per_seed),
                "test_samples": num_test,
                "y_train_mean": float(y_train_mean.item()),
                "y_train_std": float(y_train_std.item()),
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "dtype": str(dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None
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
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title)
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
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
    print(f"\nTotal experiment time for {num_seeds} seeds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tseeds: {num_seeds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    wing_SF_GPvsPFN(num_seeds=5, train_size=80, num_test=5000, noise_train=0.05, noise_test=0.05, num_runs=4, num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_seeds=4, train_size=20, num_test=5000, noise_train=0.05, noise_test=0.05, num_runs=16, num_epochs=10000, save_path="./results/wingSF/temp")
    # wing_GPvsPFN(num_seeds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True, encode_PFN_data=True)
    # wing_GPvsPFN(num_seeds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True, encode_PFN_data=False)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=False)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=True)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=False)




