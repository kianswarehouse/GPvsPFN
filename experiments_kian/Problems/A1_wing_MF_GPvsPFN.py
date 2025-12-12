import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import generate_mf_wing_data
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def wing_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=[5000, 1000, 1000, 1000],
        train_size=[10, 10, 10, 10], # total training size is train_size * number of X input dimensions
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/wing',
        title=None,
        standardize_X=True,
        standardize_y=True,
        noise_train=[0.0, 0.0, 0.0, 0.0],
        noise_test=[0.0, 0.0, 0.0, 0.0],
        noise_type='gaussian',
        encode_PFN_data=True, # False gives best results for PFN
        standardization_method=defaults.MF_STANDARDIZATION_METHOD, # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
    ):
    if title is None:
        title = f"wing_MF_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"wing_MF_{title}_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"

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
    train_per_fold = torch.tensor(train_size) * 10
    total_train = num_folds * train_per_fold
    total_samples = sum(num_test) + sum(total_train) 
    
    # Generate all unique Sobol samples at once
    print(f"Generating {total_samples} unique Sobol samples\n\tTest samples: {sum(num_test)} / Train samples: {sum(total_train)}")
    X_train_all, y_train_all, X_test_all, y_test_all = generate_mf_wing_data(
        train_samples_per_source=total_train, 
        test_samples_per_source=num_test, 
        train_noise=noise_train, 
        test_noise=noise_test, 
        noise_type=noise_type,
        seed=seed
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

    # Randomize WITHIN each source group, then split across folds
    train_indices_2d_per_source = []
    start_idx = 0
    
    for i, n_samples in enumerate(train_per_fold):
        # Get indices for this source (total samples for this source = n_samples * num_folds)
        total_samples_for_source = n_samples * num_folds
        source_indices = torch.arange(start_idx, start_idx + total_samples_for_source)
        # Randomize within this source group
        source_indices = source_indices[torch.randperm(len(source_indices))]
        # Split this source's indices across folds
        source_2d = source_indices.reshape(num_folds, n_samples)
        train_indices_2d_per_source.append(source_2d)
        start_idx += total_samples_for_source
        
    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds} {'='*20}")

        # Get training indices for this fold from each source
        fold_train_indices = []
        for source_2d in train_indices_2d_per_source:
            fold_train_indices.extend(source_2d[i].tolist())
        fold_train_indices = torch.tensor(fold_train_indices)

        # Unified train/test tensors used by BOTH GP and PFN
        X_train_orig = X_train_all[fold_train_indices]
        X_train = X_enc_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]
        # Tests are fixed across folds; use encoded version
        # X_test = X_enc_test

        # Verify source distribution for this fold
        source_counts = [torch.sum(X_train_orig[:, -1] == i).item() for i in range(len(source_cols))]
        print(f"Source distribution for fold {i}: {source_counts}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Use unified X_train/X_test for GP as well
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        # Get high-fidelity mask for standardization
        
        X_train, X_test, y_train_normal, y_train_mean, y_train_std, y_train_min = gpplus.utils.standardize_mf_data(
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
        kernel = defaults.MF_kernel(
            cont_cols=cont_cols, 
            cat_cols=cat_cols, 
            source_cols=source_cols,
        )
        
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=kernel,
            mean_module=defaults.MF_mean(encoded_cols=source_cols),
            likelihood=defaults.MF_likelihood(encoded_cols=source_cols, training_data=X_train),
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
            amp_dtype=pfn_dtype,
            regressor=regressor,
            source_cols=source_cols,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
        )
        
        TabPFN_metrics.append(tabpfn_metric)

        # Print results for this fold
        print(f"\nTabPFN Results (Fold {i+1}/{num_folds})")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        # Collect model info from first fold
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
                "train_samples": train_per_fold.tolist(),
                "test_samples": num_test,
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "standardization_method": standardization_method,
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
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    wing_GPvsPFN(num_folds=1, train_size=[10, 10, 10, 10], num_runs=4, standardization_method=2, noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_folds=1, train_size=[10, 10, 10, 10], num_runs=4, noise_train=[0.05, 0.1, 0.2, 0.5], noise_test=[0.05, 0.1, 0.2, 0.5], num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_folds=3, train_size=[20, 20, 20, 20], num_runs=4, noise_train=[0.0025, 0.005, 0.01, 0.025], noise_test=[0.0025, 0.005, 0.01, 0.025], num_epochs=10000, save_path="./results/wing/test")
    # wing_GPvsPFN(num_folds=1, train_size=[1, 1, 1, 1], noise_train=[0.05, 0.1, 0.1, 0.2], noise_test=[0.05, 0.0, 0.0, 0.0], num_runs=1, num_epochs=10000, save_path="./results/wing/temp")
    # wing_GPvsPFN(num_folds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True, encode_PFN_data=True)
    # wing_GPvsPFN(num_folds=4, num_runs=4, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True, encode_PFN_data=False)
    # wing_GPvsPFN(num_folds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=True, standardize_y=True)
    # wing_GPvsPFN(num_folds=1, num_runs=1, num_epochs=10000, save_path=None, standardize_X=False, standardize_y=False)




