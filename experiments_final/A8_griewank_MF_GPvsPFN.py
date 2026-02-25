import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from tabpfn import TabPFNRegressor
from load_experimental_data import generate_griewank_data, griewank_function
import defaults
from gpplus.kernels.mvmf_kernel import MVMFKernel
from gpplus.likelihoods.multi_likelihood import MultiLikelihood
from gpplus.means.multi_mean import MultiMean
from gpplus.kernels import LogScaleKernel
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.priors import NormalPrior

# import warnings
# warnings.filterwarnings("ignore")
def griewank_MF_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=5000,
        train_size_hf=10,  # HF training size is train_size_hf * number of X input dimensions
        train_size_lf=20,  # LF training size is train_size_lf * number of X input dimensions
        dimensions=2,
        x_bounds=[-600, 600],
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/griewank',
        title=None,
        standardize_X=True,
        standardize_y=True,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
        trainer_info=True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
    ):

    if run_models == 'pfn':
        num_runs = 0

    if title is None:
        title = f"Griewank_MF_{dimensions}Dx_HF{train_size_hf}Dn_LF{train_size_lf}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"Griewank_MF_{title}_{dimensions}Dx_HF{train_size_hf}Dn_LF{train_size_lf}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Generate data
    set_seed(seed)
    
    # Calculate total samples needed
    train_per_fold_hf = train_size_hf * dimensions  # HF: train_size_hf * dimensions
    train_per_fold_lf = train_size_lf * dimensions  # LF: train_size_lf * dimensions
    train_per_fold_total = train_per_fold_hf + train_per_fold_lf  # Total per fold
    total_train = num_folds * train_per_fold_total
    total_samples = num_test + total_train
    
    print(f"Generating {total_samples} unique Sobol samples for {dimensions}D Griewank function")
    print(f"\tTest samples: {num_test}")
    print(f"\tTrain samples: {total_train} (HF: {train_per_fold_hf} per fold, LF: {train_per_fold_lf} per fold)")
    
    # Generate train and test data in one call (all HF targets initially)
    X_train_all, y_train_all, X_test_all, y_test_all = generate_griewank_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed
    )
    
    # Add one-hot encoded source column to test data (source = HF = [1, 0])
    source_test = torch.zeros(X_test_all.shape[0], 2, dtype=X_test_all.dtype)
    source_test[:, 0] = 1.0  # HF source
    X_test_with_source = torch.cat([X_test_all, source_test], dim=1)
    
    # For encoding, use X without source column initially
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: Multi-Fidelity GP Training")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, _ = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    
    # For multi-fidelity, source_cols will be the last two columns (one-hot encoded: [dimensions, dimensions+1])
    source_cols = [dimensions, dimensions + 1]
    
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across folds

    # Randomize across the single source, then split across folds
    # Use seed to ensure consistent fold splits
    torch.manual_seed(seed)
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold_total)
        
    total_start_time = time.time()
    for i in range(num_folds):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]
        
        # Split into HF and LF indices
        hf_indices = fold_train_indices[:train_per_fold_hf]
        lf_indices = fold_train_indices[train_per_fold_hf:]
        
        # Get HF data (actual Griewank function values)
        X_hf_raw = X_train_all[hf_indices]
        y_hf = y_train_all[hf_indices]
        
        # Get LF X values (will use TabPFN predictions as targets)
        X_lf_raw = X_train_all[lf_indices]
        
        # =============================================================================
        # Scale X data BEFORE TabPFN training (fit separate scalers for HF and LF)
        # Note: For TabPFN to work, we need HF and LF X in the same space, so we use
        # HF scaler for both. But we fit separate scalers for proper multi-fidelity setup.
        # =============================================================================
        has_lf_data = len(lf_indices) > 0 and train_size_lf > 0
        
        # Fit separate X scalers for HF and LF (each fitted on their own data)
        if standardize_X:
            if x_standardize_method == 0:
                Xscaler_hf = gpplus.utils.StandardScaler()
                X_scaling_type = "StandardScaler (Gaussian)"
            elif x_standardize_method == 1:
                Xscaler_hf = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                X_scaling_type = "UniformScaler [0, 1]"
            elif x_standardize_method == 2:
                Xscaler_hf = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                X_scaling_type = "UniformScaler [-1, 1]"
            else:
                raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
            
            # Fit HF X scaler on HF data
            Xscaler_hf.fit(X_hf_raw[:, cont_cols])
            X_hf_scaled = X_hf_raw.clone()
            X_hf_scaled[:, cont_cols] = Xscaler_hf.transform(X_hf_raw[:, cont_cols])
            
            if has_lf_data:
                # Fit LF X scaler on LF data
                if x_standardize_method == 0:
                    Xscaler_lf = gpplus.utils.StandardScaler()
                elif x_standardize_method == 1:
                    Xscaler_lf = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                elif x_standardize_method == 2:
                    Xscaler_lf = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                Xscaler_lf.fit(X_lf_raw[:, cont_cols])
                X_lf_scaled = X_lf_raw.clone()
                X_lf_scaled[:, cont_cols] = Xscaler_lf.transform(X_lf_raw[:, cont_cols])
                
                # For TabPFN: use HF scaler for LF X so they're in the same space
                # (TabPFN needs both HF and LF X in the same scale)
                X_lf_scaled_for_tabpfn = X_lf_raw.clone()
                X_lf_scaled_for_tabpfn[:, cont_cols] = Xscaler_hf.transform(X_lf_raw[:, cont_cols])
            else:
                X_lf_scaled = None
                X_lf_scaled_for_tabpfn = None
                Xscaler_lf = None
            
            # For test data scaling, use HF scaler (since test is HF)
            Xscaler = Xscaler_hf
        else:
            X_scaling_type = "None"
            X_hf_scaled = X_hf_raw
            X_lf_scaled = X_lf_raw if has_lf_data else None
            X_lf_scaled_for_tabpfn = X_lf_raw if has_lf_data else None
            Xscaler = None
            Xscaler_hf = None
            Xscaler_lf = None
        
        # =============================================================================
        # Generate LF targets via TabPFN (only if LF data exists)
        # TabPFN is trained on SCALED HF data and predicts on SCALED LF X
        # =============================================================================
        if has_lf_data:
            print(f"\n--- Generating LF targets via TabPFN (Fold {i+1}/{num_folds}) ---")
            print(f"X scaling: {X_scaling_type} (fitted on HF data only)")
            
            # Get TabPFN predictions directly without computing metrics
            import numpy as np
            import warnings
            X_hf_np = X_hf_scaled.detach().cpu().numpy()
            y_hf_np = y_hf.detach().cpu().numpy().ravel()
            # Use HF-scaled LF X for TabPFN (so both are in the same space)
            X_lf_np = X_lf_scaled_for_tabpfn.detach().cpu().numpy()
            
            # Train TabPFN on SCALED HF data
            regressor.fit(X_hf_np, y_hf_np)
            
            # Get predictions on SCALED LF X (point estimates only, no metrics)
            # TabPFN predictions are in the original y scale (same as training y_hf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Suppress divide-by-zero warnings
                y_lf_pred = regressor.predict(X_lf_np)
                # Handle dict output from TabPFN
                if isinstance(y_lf_pred, dict):
                    y_lf_pred = y_lf_pred.get("mean", y_lf_pred.get("median"))
            
            y_lf = torch.tensor(y_lf_pred, dtype=gp_dtype).reshape(-1)
            
            # Calculate true Griewank values for LF X locations (without noise for comparison)
            # Use raw X_lf for true values (not scaled)
            y_lf_true = griewank_function(X_lf_raw, dimensions=dimensions)
            
            print(f"HF samples: {len(X_hf_scaled)}, LF samples: {len(X_lf_scaled)}")
            print(f"HF y range: [{y_hf.min().item():.4f}, {y_hf.max().item():.4f}], mean: {y_hf.mean().item():.4f}, std: {y_hf.std().item():.4f}")
            print(f"LF predictions range: [{y_lf.min().item():.4f}, {y_lf.max().item():.4f}], mean: {y_lf.mean().item():.4f}, std: {y_lf.std().item():.4f}")
            print(f"LF true values range: [{y_lf_true.min().item():.4f}, {y_lf_true.max().item():.4f}], mean: {y_lf_true.mean().item():.4f}, std: {y_lf_true.std().item():.4f}")
            
            # Check if LF predictions are in reasonable range compared to HF
            # TabPFN predictions should be in the same space as HF targets
            hf_mean = y_hf.mean().item()
            hf_std = y_hf.std().item()
            lf_mean = y_lf.mean().item()
            lf_std = y_lf.std().item()
            mean_diff = abs(lf_mean - hf_mean) / (hf_std + 1e-8)
            std_ratio = lf_std / (hf_std + 1e-8)
            print(f"LF vs HF: mean difference (normalized): {mean_diff:.4f}, std ratio: {std_ratio:.4f}")
            
            # Calculate TabPFN accuracy metrics
            mse_tabpfn = torch.mean((y_lf - y_lf_true) ** 2).item()
            mae_tabpfn = torch.mean(torch.abs(y_lf - y_lf_true)).item()
            rmse_tabpfn = torch.sqrt(torch.mean((y_lf - y_lf_true) ** 2)).item()
            r2_tabpfn = 1.0 - torch.sum((y_lf - y_lf_true) ** 2) / torch.sum((y_lf_true - y_lf_true.mean()) ** 2)
            print(f"TabPFN Accuracy: RMSE={rmse_tabpfn:.4f}, MAE={mae_tabpfn:.4f}, R²={r2_tabpfn.item():.4f}")
            
            # Create true vs predicted plot for TabPFN
            import matplotlib.pyplot as plt
            y_lf_np = y_lf.detach().cpu().numpy()
            y_lf_true_np = y_lf_true.detach().cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_lf_true_np, y_lf_np, alpha=0.5, s=20)
            
            # Add perfect prediction line
            min_val = min(y_lf_true_np.min(), y_lf_np.min())
            max_val = max(y_lf_true_np.max(), y_lf_np.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            plt.xlabel('True Griewank Value', fontsize=12)
            plt.ylabel('TabPFN Prediction', fontsize=12)
            plt.title(f'TabPFN Accuracy: True vs Predicted (Fold {i+1})\nRMSE={rmse_tabpfn:.4f}, R²={r2_tabpfn.item():.4f}', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(save_path) / f'tabpfn_true_vs_pred_fold{i+1}.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved TabPFN true vs predicted plot to: {plot_path}")
            plt.close()
        else:
            print(f"\n--- No LF data requested (train_size_lf=0), using HF only (Fold {i+1}/{num_folds}) ---")
            print(f"HF samples: {len(X_hf_scaled)}")
            print(f"HF y range: [{y_hf.min().item():.4f}, {y_hf.max().item():.4f}], mean: {y_hf.mean().item():.4f}, std: {y_hf.std().item():.4f}")
        
        # =============================================================================
        # Scale y data BEFORE combining (fit separate scalers for HF and LF)
        # =============================================================================
        # Fit separate Y scalers for HF and LF (each fitted on their own data)
        if standardize_y:
            # HF y scaler (fitted on HF y)
            Yscaler_hf = gpplus.utils.StandardScaler()
            Yscaler_hf.fit(y_hf.reshape(-1, 1))
            y_hf_mean = Yscaler_hf.mean
            y_hf_std = Yscaler_hf.std
            y_hf_normal = Yscaler_hf.transform(y_hf.reshape(-1, 1)).reshape(-1)
            
            # LF y scaler (fitted on LF y, if LF data exists)
            if has_lf_data:
                Yscaler_lf = gpplus.utils.StandardScaler()
                Yscaler_lf.fit(y_lf.reshape(-1, 1))
                y_lf_mean = Yscaler_lf.mean
                y_lf_std = Yscaler_lf.std
                y_lf_normal = Yscaler_lf.transform(y_lf.reshape(-1, 1)).reshape(-1)
            else:
                y_lf_mean = None
                y_lf_std = None
                y_lf_normal = None
            
            # For test data scaling, use HF scaler (since test is HF)
            y_train_mean = y_hf_mean
            y_train_std = y_hf_std
        else:
            y_train_mean = None
            y_train_std = None
            y_hf_mean = None
            y_hf_std = None
            y_lf_mean = None
            y_lf_std = None
            y_hf_normal = y_hf
            y_lf_normal = y_lf if has_lf_data else None
        
        # =============================================================================
        # Combine HF and LF data with source columns (one-hot encoded)
        # Use SCALED X data and SCALED y data (both already scaled above)
        # =============================================================================
        # One-hot encode source: [1, 0] for HF, [0, 1] for LF
        source_hf = torch.zeros(X_hf_scaled.shape[0], 2, dtype=X_hf_scaled.dtype)
        source_hf[:, 0] = 1.0  # HF source
        
        X_hf_with_source = torch.cat([X_hf_scaled, source_hf], dim=1)
        
        if has_lf_data:
            source_lf = torch.zeros(X_lf_scaled.shape[0], 2, dtype=X_lf_scaled.dtype)
            source_lf[:, 1] = 1.0  # LF source
            X_lf_with_source = torch.cat([X_lf_scaled, source_lf], dim=1)
            # Combine HF and LF (using scaled y)
            X_train = torch.cat([X_hf_with_source, X_lf_with_source], dim=0)
            y_train = torch.cat([y_hf_normal, y_lf_normal], dim=0)
        else:
            # Only HF data (using scaled y)
            X_train = X_hf_with_source
            y_train = y_hf_normal
        
        # Prepare data (X and y are already scaled, now prepare test data)
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        
        # Scale test X using the same scaler fitted on HF data
        if standardize_X:
            X_test_scaled = X_test_with_source.clone()
            X_test_scaled[:, cont_cols] = Xscaler.transform(X_test_all[:, cont_cols])
            X_test = X_test_scaled.detach().clone().to(dtype=gp_dtype)
        else:
            X_test = X_test_with_source.detach().clone().to(dtype=gp_dtype)
        
        # y_train is already scaled (y_train_normal from above)
        # For consistency with the rest of the code, use y_train_normal variable name
        y_train_normal = y_train

        # =============================================================================
        # GP Section with MVMF Kernel
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} Multi-Fidelity GP Training ---")
            
            # Create MVMF kernel
            kernel = MVMFKernel(
                cont_cols=cont_cols,
                cat_cols=cat_cols,
                source_cols=source_cols,
                source_encoder="matrix",  # Use matrix encoder for source
                z_dim=2
            )
            kernel = LogScaleKernel(kernel)
            
            # Create MultiLikelihood for per-fidelity noise
            # MultiLikelihood can handle one-hot encoded columns
            likelihood = MultiLikelihood(
                encoded_cols=source_cols,  # List of one-hot encoded column indices
                training_data=X_train
            )
            
            # Create MultiMean for per-fidelity mean functions
            # MultiMean expects encoded_cols as a list for one-hot encoding
            means = [ZeroMean()]  # For HF (source 0)
            means.append(ConstantMean())  # For LF (source 1)
            mean = MultiMean(
                means=means,
                encoded_cols=source_cols,  # List of one-hot encoded column indices [dimensions, dimensions+1]
            )
            
            # Create GP model with MVMF kernel and multi-fidelity components
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kernel,
                mean_module=mean,  # Use MultiMean for per-fidelity mean functions
                likelihood=likelihood,  # Use MultiLikelihood for per-fidelity noise
                # mean_module=defaults.SF_mean,
                # likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_folds - 1):
                lf_count = len(X_lf_scaled) if has_lf_data else 0
                print(f"X_train: {X_train.shape} (HF: {len(X_hf_scaled)}, LF: {lf_count})")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                print(f"y_train_normal (combined) mean: {y_train_normal.mean().item():.4f}, std: {y_train_normal.std().item():.4f}")
                print(f"y_train_mean (from HF only): {y_train_mean.item():.4f}, y_train_std (from HF only): {y_train_std.item():.4f}")
                print(f"Source column index: {source_cols}")
                print(model)

            # Create trainer
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
                source_cols=source_cols,
                trainer_info=trainer_info,
            )
            GPPlus_metrics.append(gp_metric)
            
            # Accumulate trainer info if available
            if gp_trainer_info:
                # Add fold information to trainer log
                gp_trainer_info["fold"] = i + 1
                gp_trainer_info["metrics"] = gp_metric  # Include metrics for this fold
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nMulti-Fidelity GP Results (Fold {i+1}/{num_folds})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}")
        
        # Collect model info from first fold
        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            if run_models in [None, 'gp']:
                gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples_hf": len(X_hf_scaled),
                "train_samples_lf": len(X_lf_scaled) if has_lf_data else 0,
                "train_samples_total": X_train.shape[0],
                "test_samples": num_test,
                "y_train_mean": float(y_train_mean.item()),
                "y_train_std": float(y_train_std.item()),
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "X_scaling_type": X_scaling_type,
                "x_standardize_method": x_standardize_method,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                "x_bounds": x_bounds,
                "train_size_hf": train_size_hf,
                "train_size_lf": train_size_lf,
                **y_test_stats,
                "num_folds": num_folds,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="Multi-Fidelity GP", title=title) if run_models in [None, 'gp'] else None
    
    if save_path is not None:
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Determine file prefix based on run_models
            if run_models is not None:
                file_prefix = run_models
            else:
                file_prefix = "gpMF"
            
            # Combined single file: GP data + GP model_info
            combined_data = {}
            if run_models in [None, 'gp']:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info
                }
            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
        
        # Save trainer info if trainer_info is enabled
        if trainer_info and GPTrainer_info and run_models in [None, 'gp']:
            try:
                # Create trainer_analysis directory (same level as plots)
                trainer_analysis_dir = Path(save_path) / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
                
                # Save raw trainer info (just the parameter info)
                trainer_info_data = {
                    "title": title,
                    "num_folds": num_folds,
                    "num_runs_per_fold": num_runs,
                    "trainer_info": GPTrainer_info,
                }
                
                # Save trainer info JSON (always use "gp" prefix for GP trainer info)
                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                
            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\tX_scaling_type: {X_scaling_type}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {train_per_fold_hf} HF + {train_per_fold_lf} LF train samples per fold\n\tfolds: {num_folds}")

    return GPPlus_metrics


if __name__ == "__main__":
    griewank_MF_GPvsPFN(num_folds=1, train_size_hf=10, train_size_lf=40, dimensions=10, num_runs=16, noise_train=0.005, noise_test=0.005, save_path='./results/griewank/temp', run_models='gp')
    # griewank_MF_GPvsPFN(num_folds=1, train_size_hf=10, train_size_lf=20, dimensions=20, num_runs=4, save_path='./results/griewank/temp')
    # griewank_MF_GPvsPFN(num_folds=1, train_size_hf=10, train_size_lf=20, dimensions=40, num_runs=4, save_path='./results/griewank/temp')
