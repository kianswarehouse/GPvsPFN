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
from sklearn.utils import shuffle
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
def create_training_folds(X_train, y_train, num_folds=10, train_size=None, random_state=None):
    """
    Create training folds ensuring all samples are used at least once.
    
    Args:
        X_train: Training features
        y_train: Training targets
        num_folds: Number of folds to create
        train_size: Number of samples per fold. If None, uses all available samples.
        random_state: Random seed for reproducibility
    
    Returns:
        List of tuples (X_fold, y_fold) for each fold
    """
    n_samples = len(X_train)
    
    # If train_size not specified, use all samples in each fold
    if train_size is None:
        train_size = n_samples
    
    # Convert to numpy for indexing if needed
    if isinstance(X_train, torch.Tensor):
        X_train_np = X_train.numpy() if X_train.device.type == 'cpu' else X_train.cpu().numpy()
        y_train_np = y_train.numpy() if y_train.device.type == 'cpu' else y_train.cpu().numpy()
    else:
        X_train_np = X_train
        y_train_np = y_train
    
    # Create random indices
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    # Step 1: Distribute all samples round-robin to ensure all are used at least once
    fold_indices = [[] for _ in range(num_folds)]
    for i, idx in enumerate(indices):
        fold_idx = i % num_folds
        fold_indices[fold_idx].append(idx)
    
    # Step 2: For each fold, adjust to target train_size
    folds = []
    for fold_idx in range(num_folds):
        fold_sample_indices = fold_indices[fold_idx]
        
        if len(fold_sample_indices) > train_size:
            # Randomly sample train_size from this fold
            fold_sample_indices = rng.choice(fold_sample_indices, size=train_size, replace=False)
        elif len(fold_sample_indices) < train_size:
            # Need to add more samples to reach train_size
            needed = train_size - len(fold_sample_indices)
            # Collect all indices already assigned to folds
            all_assigned = set()
            for f_idx in range(num_folds):
                all_assigned.update(fold_indices[f_idx])
            
            # Try to get unused samples first
            unused = [idx for idx in indices if idx not in all_assigned]
            if len(unused) >= needed:
                additional = rng.choice(unused, size=needed, replace=False)
                fold_sample_indices = list(fold_sample_indices) + list(additional)
            else:
                # Not enough unused samples, use replacement from all indices
                additional = rng.choice(indices, size=needed, replace=True)
                fold_sample_indices = list(fold_sample_indices) + list(additional)
        
        # Convert back to tensor if original was tensor
        if isinstance(X_train, torch.Tensor):
            X_fold = torch.tensor(X_train_np[fold_sample_indices], dtype=X_train.dtype)
            y_fold = torch.tensor(y_train_np[fold_sample_indices], dtype=y_train.dtype)
        else:
            X_fold = X_train_np[fold_sample_indices]
            y_fold = y_train_np[fold_sample_indices]
        
        folds.append((X_fold, y_fold))
    
    # Verify and report
    total_samples = 0
    for X_fold, y_fold in folds:
        total_samples += len(X_fold)
        # Note: All samples are guaranteed to be used at least once due to round-robin distribution
    
    print(f"  Created {num_folds} folds")
    print(f"  Samples per fold: {[len(X_fold) for X_fold, _ in folds]}")
    print(f"  Total unique samples in training set: {n_samples}")
    print(f"  Total samples across all folds: {total_samples}")
    print(f"  All samples used at least once: {total_samples >= n_samples}")
    
    return folds


def M2AX_GPvsPFN(
        num_seeds=20,
        test_size=0.103, 
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
        num_folds=10,  # Number of training folds
        train_size=None,  # Number of samples per fold. If None, uses all available training samples.
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
    qual_dict = learn_encodings(X)

    X_enc, cont_cols, cat_cols, source_cols = encode_qual_data(X, qual_dict=qual_dict)
    # print(qual_dict)
    # print(cat_cols)
    TabPFN_M2AX_metrics = []
    GPPlus_M2AX_metrics = []
    # Store detailed fold-level results: [seed_idx][fold_idx] = metrics
    all_gp_fold_metrics = []  # Per-seed, per-fold GP metrics
    all_tabpfn_fold_metrics = []  # Per-seed, per-fold TabPFN metrics
    set_seed(0)  # Set global seed for reproducible seeds
    seeds = np.random.RandomState(0).choice(10**6, size=num_seeds, replace=False).tolist()
    total_start_time = time.time()  # Track total experiment time
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        seed_start_time = time.time()  # Track time for this seed
        
        # First split: separate test set
        X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=seed)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        # Create training folds from remaining training data
        print(f"\nCreating {num_folds} training folds from {len(X_train_enc)} training samples...")
        if train_size is not None:
            print(f"  Target samples per fold: {train_size}")
        else:
            print(f"  Using all {len(X_train_enc)} samples in each fold")
        
        # Create folds for both encoded and non-encoded data using the same random_state
        # This ensures both versions have the same sample distribution
        train_folds_enc = create_training_folds(X_train_enc, y_train, num_folds=num_folds, train_size=train_size, random_state=seed)
        train_folds = create_training_folds(X_train, y_train, num_folds=num_folds, train_size=train_size, random_state=seed)
        
        # Verify fold consistency
        assert len(train_folds_enc) == num_folds, f"Expected {num_folds} encoded folds, got {len(train_folds_enc)}"
        assert len(train_folds) == num_folds, f"Expected {num_folds} non-encoded folds, got {len(train_folds)}"
        for fold_idx in range(num_folds):
            assert len(train_folds_enc[fold_idx][0]) == len(train_folds[fold_idx][0]), \
                f"Fold {fold_idx}: Encoded and non-encoded fold sizes don't match"
        
        # Store fold-level metrics
        seed_gp_metrics = []
        seed_tabpfn_metrics = []
        
        # Process each fold: train separate models on each fold's training data,
        # evaluate on the same test set, then average metrics across folds
        for fold_idx, ((X_train_fold_enc, y_train_fold), (X_train_fold, _)) in enumerate(zip(train_folds_enc, train_folds)):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx+1}/{num_folds} (Seed {seed})")
            print(f"{'='*60}")
            print(f"Fold training samples: {len(X_train_fold_enc)}")
        
            # =============================================================================
            # GP M2AX Section 
            # =============================================================================
            print(f"\n--- {title} GP Evaluation (Fold {fold_idx+1}) ---")

            # Use fold data for training, test set remains the same
            X_gp_train = X_train_fold_enc.detach().clone().to(dtype=dtype) if isinstance(X_train_fold_enc, torch.Tensor) else torch.tensor(X_train_fold_enc, dtype=dtype)
            X_gp_test = X_test_enc.detach().clone().to(dtype=dtype)
            y_gp_train = y_train_fold.detach().clone().to(dtype=dtype) if isinstance(y_train_fold, torch.Tensor) else torch.tensor(y_train_fold, dtype=dtype)
            y_gp_test = y_test.detach().clone().to(dtype=dtype)

            # Normalize the GP data
            y_gp_train_mean = y_gp_train.mean()
            y_gp_train_std = y_gp_train.std()
            y_gp_train_normal = (y_gp_train - y_gp_train_mean) / y_gp_train_std

            # cat_cols was returned by the encoder; CombinedKernel expects only cat indices
            # print(cat_cols)
            # Create encoders for each categorical group
            # Each encoder needs input_dim equal to the number of columns in that specific group
            # cat_cols = [np.arange(0, 24).tolist()]
            # print(cat_cols)
            # cat_encoders = [RelativeDistanceEncoder(input_dim=len(cat_col), z_dim=2, initialization='uniform', init_radius=0.5, seed=seed) for cat_col in cat_cols]
            cat_encoders = 'matrix'
            kernel = gpplus.kernels.LogScaleKernel(
                gpplus.kernels.CombinedKernel(
                    cat_cols=cat_cols,
                    cat_encoder=cat_encoders,
                )
            )
            # kernel = gpplus.kernels.LogScaleKernel(
            #     gpplus.kernels.CombinedKernel_OneCatK(
            #         cat_cols=cat_cols,
            #         cat_encoder=cat_encoders,
            #         z_dim=2,
            #     )
            # )
            # kernel = gpplus.kernels.LogScaleKernel(gpplus.kernels.GaussianKernel(
                # ard_num_dims=len(cat_cols[0]),
            # ))

            # Create GP model
            model = gpplus.models.GPR(
                X_gp_train,
                y_gp_train_normal if standardize_y_gp else y_gp_train,
                kernel_module=kernel,
                # likelihood=gpplus.likelihoods.LogGaussianLikelihood(
                    # noise_constraint=gpytorch.constraints.GreaterThan(1e-6), 
                    # noise_prior=gpytorch.priors.LogNormalPrior(loc=np.log(y_gp_train_std**2), scale=1.0)),
                    # noise_prior=gpytorch.priors.NormalPrior(loc=0, scale=0.0001),
                # )
            )
            if (i == 0 and fold_idx == 0) or (i == len(seeds) - 1 and fold_idx == num_folds - 1):
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
            seed_gp_metrics.append(gp_metric)

            print(f"\nGP Results (Seed {seed}, Fold {fold_idx+1})")
            from gpplus.utils.metrics_functions import format_metric_value
            for k, v in gp_metric.items():
                print(f"  {k}: {format_metric_value(k, v)}")

            # =============================================================================
            # TabPFN Section
            # =============================================================================
            print(f"\n--- {title} TabPFN Evaluation (Fold {fold_idx+1}) ---")
            
            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
                X_train_fold_enc if encode_PFN_data else X_train_fold,
                X_test_enc if encode_PFN_data else X_test,
                y_train_fold,
                y_test,
                amp_device=amp_device,
                amp_dtype=amp_dtype,
                regressor=regressor,
                # source_cols=source_cols,
            )
            seed_tabpfn_metrics.append(tabpfn_metric)

            # Print results for this fold
            print(f"\nTabPFN Results (Seed {seed}, Fold {fold_idx+1})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}")
        
        # Aggregate metrics across folds for this seed (average)
        print(f"\n{'='*60}")
        print(f"Seed {seed} Summary: Averaging metrics across {num_folds} folds")
        print(f"{'='*60}")
        
        # Store fold-level metrics for this seed
        # Convert metrics to JSON-serializable format (convert numpy types to native Python types)
        # Include seed and fold number in each fold's metrics for easy identification
        seed_gp_folds_serializable = []
        for fold_idx, fold_metrics in enumerate(seed_gp_metrics):
            fold_dict = {
                "seed": seed,
                "seed_index": i,
                "fold": fold_idx + 1,  # 1-indexed for readability
                "fold_index": fold_idx,  # 0-indexed for programmatic access
            }
            for k, v in fold_metrics.items():
                if isinstance(v, (np.integer, np.floating)):
                    fold_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, np.ndarray):
                    fold_dict[k] = v.tolist()
                else:
                    fold_dict[k] = v
            seed_gp_folds_serializable.append(fold_dict)
        
        seed_tabpfn_folds_serializable = []
        for fold_idx, fold_metrics in enumerate(seed_tabpfn_metrics):
            fold_dict = {
                "seed": seed,
                "seed_index": i,
                "fold": fold_idx + 1,  # 1-indexed for readability
                "fold_index": fold_idx,  # 0-indexed for programmatic access
            }
            for k, v in fold_metrics.items():
                if isinstance(v, (np.integer, np.floating)):
                    fold_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, np.ndarray):
                    fold_dict[k] = v.tolist()
                else:
                    fold_dict[k] = v
            seed_tabpfn_folds_serializable.append(fold_dict)
        
        all_gp_fold_metrics.append(seed_gp_folds_serializable)
        all_tabpfn_fold_metrics.append(seed_tabpfn_folds_serializable)
        
        # Average GP metrics across folds
        avg_gp_metric = {}
        if seed_gp_metrics:
            for key in seed_gp_metrics[0].keys():
                avg_gp_metric[key] = np.mean([m[key] for m in seed_gp_metrics])
        
        # Average TabPFN metrics across folds
        avg_tabpfn_metric = {}
        if seed_tabpfn_metrics:
            for key in seed_tabpfn_metrics[0].keys():
                avg_tabpfn_metric[key] = np.mean([m[key] for m in seed_tabpfn_metrics])
        
        GPPlus_M2AX_metrics.append(avg_gp_metric)
        TabPFN_M2AX_metrics.append(avg_tabpfn_metric)
        
        print(f"\nAveraged GP Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        for k, v in avg_gp_metric.items():
            print(f"  {k}: {format_metric_value(k, v)}")
        
        print(f"\nAveraged TabPFN Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        for k, v in avg_tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        seed_elapsed_time = time.time() - seed_start_time
        print(f"\nSeed {seed} completed in {seed_elapsed_time:.2f}s")
        
        if i == 0:
            # Use first fold's model for info
            gp_model_info = {
                "model_str": str(model),
                "cont_cols": cont_cols,
                "cat_cols": cat_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_gp_train.shape[1],
                "train_samples_per_fold": X_gp_train.shape[0],
                "num_folds": num_folds,
                "total_train_samples": len(X_train_enc),
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
            # Convert averaged metrics to JSON-serializable format
            # Include seed information in averaged metrics
            gp_metrics_serializable = []
            for seed_idx, metric_dict in enumerate(GPPlus_M2AX_metrics):
                serializable_dict = {
                    "seed": seeds[seed_idx],
                    "seed_index": seed_idx,
                }
                for k, v in metric_dict.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                    elif isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                    else:
                        serializable_dict[k] = v
                gp_metrics_serializable.append(serializable_dict)
            
            tabpfn_metrics_serializable = []
            for seed_idx, metric_dict in enumerate(TabPFN_M2AX_metrics):
                serializable_dict = {
                    "seed": seeds[seed_idx],
                    "seed_index": seed_idx,
                }
                for k, v in metric_dict.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                    elif isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                    else:
                        serializable_dict[k] = v
                tabpfn_metrics_serializable.append(serializable_dict)
            
            # Combined single file: TabPFN data + GP data + fold-level data + model_info
            combined_data = {
                "seeds": seeds,
                "num_folds": num_folds,
                "train_size_per_fold": train_size,
                "test_size": test_size,
                "gp_data": {
                    "summary": GPPlus_M2AX_summary,
                    "averaged_metrics_per_seed": gp_metrics_serializable,  # Averaged across folds per seed
                    "fold_metrics": all_gp_fold_metrics,  # [seed_idx][fold_idx] = metrics
                    "gp_model_info": gp_model_info
                },
                "tabpfn_data": {
                    "summary": TabPFN_M2AX_summary,
                    "averaged_metrics_per_seed": tabpfn_metrics_serializable,  # Averaged across folds per seed
                    "fold_metrics": all_tabpfn_fold_metrics,  # [seed_idx][fold_idx] = metrics
                    "pfn_model_info": tabpfn_model_info
                },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
            print(f"\nSaved results to: {out_dir / f'gpVpfn_{title}.json'}")
            print(f"  - Averaged metrics for {len(seeds)} seeds")
            print(f"  - Fold-level metrics: {len(seeds)} seeds × {num_folds} folds = {len(seeds) * num_folds} total folds")
        except Exception as e:
            print(f"\nWarning: Failed to save results: {e}")
            import traceback
            traceback.print_exc()
    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal experiment time for {len(seeds)} seeds: {total_elapsed_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\ty_standardize: {standardize_y_gp}")
    print(f"Experiment details: \n\ttest size: {test_size} ({len(X_test)} test samples, {len(X_train)} train samples)\n\tnum_folds: {num_folds}\n\ttrain_size per fold: {train_size if train_size is not None else 'all available'}\n\tseeds: {seeds}")

    return GPPlus_M2AX_metrics, TabPFN_M2AX_metrics,

if __name__ == "__main__":
    # M2AX_GPvsPFN()
    M2AX_GPvsPFN(num_seeds=10, num_runs=16, num_epochs=10000, optimizer_class=LBFGSScipy, save_path='./results/M2AX/FoldsTest', num_folds=1, train_size=100)
    # M2AX_GPvsPFN(num_seeds=2, num_runs=4, num_epochs=10000, save_path=None, encode_PFN_data=True, num_folds=10, train_size=20)