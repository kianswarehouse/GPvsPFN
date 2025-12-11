import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import generate_rastrigin_data
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def rastrigin_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions
        dimensions=5,
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/rastrigin',
        title=None,
        standardize_X=True,
        standardize_y=True,
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        shift=None,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
    ):

    shift_str = f"_shifted{shift}" if shift is not None else ""
    if title is None:
        title = f"Rastrigin{shift_str}_{dimensions}xdim_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"Rastrigin{shift_str}_{dimensions}xdim_{train_size}D_{title}"
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Generate data
    set_seed(seed)
    
    # Calculate total samples needed
    train_per_fold = train_size * dimensions  # train_size * dimensions for Rastrigin
    total_train = num_folds * train_per_fold
    total_samples = num_test + total_train
    
    print(f"Generating {total_samples} unique Sobol samples for {dimensions}D Rastrigin function\n\tTest samples: {num_test} / Train samples: {total_train}")
    if shift is not None:
        print(f"  Using shifted Rastrigin with shift: {shift}")
    
    # Generate train and test data in one call
    X_train_all, y_train_all, X_test_all, y_test_all = generate_rastrigin_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
        shift=shift
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    # _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Randomize across the single source, then split across folds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)
        
    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]
        X_train = X_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Reuse PFN split, convert to torch (unified)
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        if standardize_X:
            Xscaler = gpplus.utils.StandardScaler()
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])

        # Normalize the GP data
        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        # Create GP model (default kernel like SF wing)
        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=defaults.SF_kernel,
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
            optimizer_class=optimizer_class,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,
        )
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Fold {i+1}/{num_folds})")
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
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": X_train.shape[0],
                "test_samples": num_test,
                "y_train_mean": float(y_train_mean.item()),
                "y_train_std": float(y_train_std.item()),
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
                "shift": str(shift) if shift is not None else None,
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
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}\n\tshift: {shift}")
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Standard Rastrigin
    rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=20, num_runs=4, save_path='./results/rastrigin/temp')
    rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=40, num_runs=4, save_path='./results/rastrigin/temp')
    rastrigin_GPvsPFN(num_folds=1, train_size=20, dimensions=20, num_runs=4, save_path='./results/rastrigin/temp')
    rastrigin_GPvsPFN(num_folds=1, train_size=20, dimensions=40, num_runs=4, save_path='./results/rastrigin/temp')
    
    # Shifted Rastrigin examples (commented out)
    rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=20, num_runs=4, save_path='./results/rastrigin/temp', shift=2.0)
    # shift_vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # for 5D
    # rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=5, num_runs=4, save_path='./results/rastrigin/temp', shift=shift_vector)

