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
def wing_GPvsPFN(num_seeds=20,
        num_test=[5000, 1000, 1000, 1000],
        train_size=[10, 10, 10, 10], # total training size is train_size * number of X input dimensions
        num_runs=16, 
        num_epochs=10000, 
        lr=0.1, 
        convergence_patience=10,
        optimizer_class=gpplus.training.optimizers.LBFGSScipy,
        initializer_class=None,
        gp_device='cpu',
        amp_device='cuda',
        save_path='./results/wing',
        title=None,
        standardize_X=True,
        standardize_y=True,
        noise_train=[0.0, 0.0, 0.0, 0.0],
        noise_test=[0.0, 0.0, 0.0, 0.0],
        noise_type='gaussian',
        encode_PFN_data=True, # False gives best results for PFN
        standardization_method=2, # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently
    ):
    if title is None:
        title = f"wingMF_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"wingMF{title}_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    
    
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
    train_per_seed = np.array(train_size) * 10
    total_train = num_seeds * train_per_seed
    total_samples = sum(num_test) + sum(total_train) 
    
    # Generate all unique Sobol samples at once
    print(f"Generating {total_samples} unique Sobol samples...")
    X_train_all, y_train_all, X_test_all, y_test_all = generate_mf_wing_data(
        train_samples_per_source=total_train, 
        test_samples_per_source=num_test, 
        train_noise=noise_train, 
        test_noise=noise_test, 
        noise_type=noise_type
    )
    
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_train_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=-1)
    X_enc_test_all, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=-1)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Randomize WITHIN each source group, then split across seeds
    train_indices_2d_per_source = []
    start_idx = 0
    
    for i, n_samples in enumerate(train_per_seed):
        # Get indices for this source (total samples for this source = n_samples * num_seeds)
        total_samples_for_source = n_samples * num_seeds
        source_indices = torch.arange(start_idx, start_idx + total_samples_for_source)
        # Randomize within this source group
        source_indices = source_indices[torch.randperm(len(source_indices))]
        # Split this source's indices across seeds
        source_2d = source_indices.reshape(num_seeds, n_samples)
        train_indices_2d_per_source.append(source_2d)
        start_idx += total_samples_for_source
        
    total_start_time = time.time()
    for i in range(num_seeds):
        seed = i  # deterministic seed per split
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()

        # Get training indices for this seed from each source
        seed_train_indices = []
        for source_2d in train_indices_2d_per_source:
            seed_train_indices.extend(source_2d[i].tolist())
        seed_train_indices = torch.tensor(seed_train_indices)

        # Unified train/test tensors used by BOTH GP and PFN
        X_train_orig = X_train_all[seed_train_indices]
        X_train = X_enc_train_all[seed_train_indices]
        y_train = y_train_all[seed_train_indices]
        # Tests are fixed across seeds; use encoded version
        # X_test = X_enc_test

        # Verify source distribution for this seed
        source_counts = [torch.sum(X_train_orig[:, -1] == i).item() for i in range(4)]
        print(f"Source distribution for seed {i}: {source_counts}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Use unified X_train/X_test for GP as well
        X_train = X_train.detach().clone().to(dtype=dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=dtype)
        y_train = y_train.detach().clone().to(dtype=dtype)
        y_test = y_test_all.detach().clone().to(dtype=dtype)
        # Get high-fidelity mask for standardization
        
        X_train, X_test, y_train_normal, y_train_mean, y_train_std = gpplus.utils.standardize_mf_data(
            X_train,
            X_test,
            y_train,
            cont_cols,
            source_cols,
            standardize_X=standardize_X,
            standardize_y=standardize_y,
            standardization_method=standardization_method,
        )

        # cat_cols was returned by the encoder; CombinedKernel expects only cat indices
        kernel = gpplus.kernels.LogScaleKernel(gpplus.kernels.MVMFKernel(
        # kernel = gpplus.kernels.CombinedKernel(
                cont_cols=cont_cols, 
                cat_cols=cat_cols, 
                source_cols=source_cols)
        )
        
        # Create GP model
        from gpplus.means import MultiMean
        from gpplus.likelihoods import MultiLikelihood
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=kernel,
            mean_module=gpplus.means.MultiMean(encoded_cols=source_cols),
            likelihood=gpplus.likelihoods.MultiLikelihood(encoded_cols=source_cols, training_data=X_train),
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
            if v is None:
                print(f"  {k}: None")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")
        
        # Use the same unified inputs for PFN
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_train,
            X_test,
            y_train_normal if standardize_y else y_train,
            y_test,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            regressor=regressor,
            source_cols=source_cols,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
        )
        
        TabPFN_metrics.append(tabpfn_metric)

        # Print results for this seed
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        # Collect model info from first seed
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
            # Always report overall stats
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }
            # For original data, source is always in the last column (-1)
            source_indices_test = X_test_all[:, -1].long()
            unique_sources = torch.unique(source_indices_test)
            if len(unique_sources) > 1:
                # MF: Also calculate per-source stats
                for source_idx in unique_sources:
                    source_mask = source_indices_test == source_idx
                    y_test_source = y_test_all[source_mask]
                    y_test_stats[f"y_test_mean_source_{source_idx.item()}"] = float(y_test_source.mean().item())
                    y_test_stats[f"y_test_std_source_{source_idx.item()}"] = float(y_test_source.std().item())
            
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": train_per_seed.tolist(),
                "test_samples": num_test,
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "dtype": str(dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                **y_test_stats
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
    wing_GPvsPFN(num_seeds=1, title = "2", train_size=[10, 10, 10, 10], num_runs=4, standardization_method=2, noise_train=[0.0025, 0.005, 0.01, 0.025], noise_test=[0.0025, 0.005, 0.01, 0.025], num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_seeds=1, train_size=[10, 10, 10, 10], num_runs=4, noise_train=[0.05, 0.1, 0.2, 0.5], noise_test=[0.05, 0.1, 0.2, 0.5], num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_seeds=3, train_size=[20, 20, 20, 20], num_runs=4, noise_train=[0.0025, 0.005, 0.01, 0.025], noise_test=[0.0025, 0.005, 0.01, 0.025], num_epochs=10000, save_path="./results/wing/test")
    # wing_GPvsPFN(num_seeds=1, train_size=[1, 1, 1, 1], noise_train=[0.05, 0.1, 0.1, 0.2], noise_test=[0.05, 0.0, 0.0, 0.0], num_runs=1, num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_seeds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True, encode_PFN_data=True)
    # wing_GPvsPFN(num_seeds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True, encode_PFN_data=False)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=False, standardize_y=False)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=False, standardize_y=True)
    # wing_GPvsPFN(num_seeds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=False)




