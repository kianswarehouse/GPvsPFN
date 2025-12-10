import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import generate_mf_buckling_data_with_folds
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def buckling_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=[5000, 1000],
        train_size=[10, 10], # total training size is train_size * number of X input dimensions
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/buckling',
        title=None,
        standardize_X=True,
        standardize_y=True,
        noise_train=[0.0, 0.0],
        noise_test=[0.0, 0.0],
        noise_type='gaussian',
        standardization_method=defaults.MF_STANDARDIZATION_METHOD, # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN
    ):
    if title is None:
        title = f"bucklingMF_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"bucklingMF{title}_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"

    # Generate data
    set_seed(seed)
        
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Calculate total samples needed (4D problem)
    train_per_fold = torch.tensor(train_size) * 4  # 4 input dimensions for buckling
    total_train = num_folds * train_per_fold
    total_samples = sum(num_test) + sum(total_train) 
    
    # Generate all unique Sobol samples at once with pre-stratified folds (matching SF pattern)
    print(f"Generating {total_samples} unique Sobol samples...")
    X_train_folds, y_train_folds, X_test_all, y_test_all = generate_mf_buckling_data_with_folds(
        train_samples_per_source=total_train,
        test_samples_per_source=num_test,
        num_folds=num_folds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed
    )
    
    # Combine all train folds for TabPFN
    X_train_all = torch.cat(X_train_folds, dim=0)
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_test_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=-1)
    
    # Encode each fold individually for GP training
    X_train_folds_enc = []
    for fold_data in X_train_folds:
        fold_enc, _, _, _ = encode_qual_data(fold_data, qual_dict=qual_dict, source_col=-1)
        X_train_folds_enc.append(fold_enc)
    
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Debug: Print categorical distributions for each fold
    print(f"\n{'='*20} PRE-STRATIFIED FOLDS VERIFICATION {'='*20}")
    for fold in range(num_folds):  # Show all folds
        fold_data = X_train_folds[fold]
        
        print(f"\nFold {fold + 1} categorical distributions:")
        # Check I distribution (column 3)
        for i in range(3):
            count = (fold_data[:, 3] == i).sum().item()
            print(f"  I={i}: {count} samples")
        
        # Check E distribution (column 1) 
        for i in range(2):
            count = (fold_data[:, 1] == i).sum().item()
            print(f"  E={i}: {count} samples")
            
        # Check K distribution (column 2)
        for i in range(4):
            count = (fold_data[:, 2] == i).sum().item()
            print(f"  K={i}: {count} samples")
        
        # Check source distribution
        source_counts = [torch.sum(fold_data[:, -1] == i).item() for i in range(2)]
        print(f"  Source distribution: {source_counts}")
    
    print(f"{'='*60}")
        
    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {i} {'='*20}")

        # Use pre-generated fold
        X_train_orig = X_train_folds[i]
        X_train = X_train_folds_enc[i]
        y_train = y_train_folds[i]

        # Verify source distribution for this fold
        source_counts = [torch.sum(X_train_orig[:, -1] == i).item() for i in range(2)]
        print(f"Source distribution for fold {i}: {source_counts}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Convert to torch dtype and optionally standardize X
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)

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

        kernel = defaults.MF_kernel(
            cat_cols=cat_cols,
            source_cols=source_cols,
            cont_cols=cont_cols,
        )

        # Create GP model
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=kernel,
            mean_module=defaults.MF_mean(encoded_cols=source_cols),
            likelihood=defaults.MF_likelihood(encoded_cols=source_cols, training_data=X_train),
        )
        if (i == 0) or (i == num_folds - 1):
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
            source_cols=source_cols,
        )
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Fold {i}) [{i+1}/{num_folds}]")
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
        print(f"\nTabPFN Results (Fold {i}) [{i+1}/{num_folds}]")
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
    buckling_GPvsPFN(num_folds=1, train_size=[5, 5], num_runs=4, num_epochs=10000, save_path='./results/buckling/temp')
    # buckling_GPvsPFN(num_folds=3, train_size=[10, 10], num_runs=4, num_epochs=10000, save_path='./results/buckling/temp')
    # buckling_GPvsPFN(num_folds=3, train_size=[20, 10], num_runs=4, num_epochs=10000, save_path='./results/buckling/temp')
    # buckling_GPvsPFN(num_folds=3, train_size=[15, 5], num_runs=4, num_epochs=10000, save_path='./results/buckling/temp')
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=False)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=False)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', encode_PFN_data=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', encode_PFN_data=False)



