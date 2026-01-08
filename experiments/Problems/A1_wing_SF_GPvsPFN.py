import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from data.load_experimental_data import generate_mf_wing_data
from gpplus.training import PrescreeningRecorder
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def wing_SF_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,        
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/wing',
        title=None,
        standardize_X=True,
        standardize_y=True,
        x_standardize_method=2,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
        trainer_info=False,  # Set to True if you want trainer info
        use_recorder=False,  # Set to False to disable PrescreeningRecorder
    ):
    
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

    # Randomize across the single source (s0), then split across folds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)
        
    total_start_time = time.time()
    for i in range(num_folds):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]

        X_train = X_train_all[fold_train_indices]
        # X_enc_train = X_enc_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Reuse PFN split, convert to torch
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        # Determine X scaling type
        X_scaling_type = "None"
        if standardize_X:
            if x_standardize_method == 0:
                Xscaler = gpplus.utils.StandardScaler()
                X_scaling_type = "StandardScaler (Gaussian)"
            elif x_standardize_method == 1:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                X_scaling_type = "UniformScaler [0, 1]"
            elif x_standardize_method == 2:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                X_scaling_type = "UniformScaler [-1, 1]"
            else:
                raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
            Xscaler.fit(X_train)
            X_train = Xscaler.transform(X_train)
            X_test = Xscaler.transform(X_test)


        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        # Create GP model
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
        # Enable pre-screening: generate N_test = N_initialization * D_x Sobol samples,
        # evaluate loss for all, and select best N_initialization for optimization
        # Prescreening is enabled/disabled based on use_recorder parameter
        enable_prescreening = use_recorder  # Enable prescreening when use_recorder is True
        input_dim = X_train.shape[1]  # 10 for wing problem
        
        # Create recorder to track pre-screening data (if enabled)
        recorder = None
        if use_recorder:
            recorder = PrescreeningRecorder(
                save_path=os.path.join(save_path, f"prescreening_data_fold_{i+1}_seed_{fold_seed}.json") if save_path else None,
                verbose=True
            )
        
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
            source_cols=source_cols,  # Source column is at index 10 (single int = not encoded)
            trainer_info=trainer_info,  # Set to True if you want trainer info
            # enable_prescreening=enable_prescreening,
            # input_dim=input_dim,  # Pass input dimension for pre-screening
            # prescreening_warmup_epochs=0,  # Warmup epochs for better loss evaluation
            # prescreening_warmup_lr=0.01,  # Learning rate for warmup
            # recorder=recorder,
        )
        
        # Save and print recorder summary
        if enable_prescreening and recorder is not None:
            recorder.save()
            if len(recorder.prescreening_data) > 0:
                recorder.print_summary()
                # Save as table (CSV) - with error handling
                try:
                    recorder.save_table(format="csv")
                except PermissionError:
                    print("Warning: Could not save CSV table - file may be open in another program.")
                    print("  You can still access the data from the JSON file or call save_table() later.")
                except Exception as e:
                    print(f"Warning: Could not save CSV table: {e}")
                # Print table (first 20 rows)
                recorder.print_table(max_rows=20, include_params=False)
            else:
                print("Warning: PrescreeningRecorder: No pre-screening data was recorded. Pre-screening may not have run.")
        
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Fold {i+1}/{num_folds})")
        from gpplus.utils.metrics_functions import format_metric_value
        for k, v in gp_metric.items():
            print(f"  {k}: {format_metric_value(k, v)}")

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
            print(f"  {k}: {v:.4f}")
        
        # Collect model info from first fold
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
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
                "train_samples": int(train_per_fold),
                "test_samples": num_test,
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "x_standardize_method": x_standardize_method,
                "X_scaling_type": X_scaling_type,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                "enable_prescreening": enable_prescreening,
                "prescreening_warmup_epochs": 0,
                "prescreening_warmup_lr": 0.01,
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
    wing_SF_GPvsPFN(num_folds=2, train_size=10, num_test=5000, noise_train=0.00, noise_test=0.0, num_runs=4, num_epochs=100, save_path="./results/wing/temp", use_recorder=True)

