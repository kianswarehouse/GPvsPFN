import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from tabpfn import TabPFNRegressor
from load_experimental_data import generate_mf_buckling_data_with_folds
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def buckling_SF_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions (4)
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/buckling',
        title=None,
        standardize_X=True,
        standardize_y=True,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        standardize_y_log_scale=False,
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
        trainer_info=True,
        MF_kernel=True,
        run_pfn: bool = True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
    ):
    if run_models == 'pfn':
        num_runs = 0
        run_pfn = True
    elif run_models == 'gp':
        run_pfn = False
    # else run_models is None, keep run_pfn as passed
    
    if title is None:
        title = f"buckling_SF_{train_size}Dn_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"buckling_SF_{title}_{train_size}Dn_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    
    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device) if run_pfn else None
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None
    
    # Calculate total samples needed (4D problem)
    train_per_fold = train_size * 4
    total_train = num_folds * train_per_fold
    total_samples = num_test + total_train
    
    # Generate all unique Sobol samples at once for single-fidelity (use only s0)
    print(f"Generating {total_samples} unique Sobol samples\n\tTest samples: {num_test} / Train samples: {total_train}")
    X_train_folds, y_train_folds, X_test_all, y_test_all = generate_mf_buckling_data_with_folds(
        train_samples_per_source=[total_train, 0],
        test_samples_per_source=[num_test, 0],
        num_folds=num_folds,
        train_noise=[noise_train, 0.0],
        test_noise=[noise_test, 0.0],
        noise_type=noise_type,
        seed=seed,
    )

    # Drop the 5th (source) column since SF uses only s0
    for i in range(len(X_train_folds)):
        if X_train_folds[i].shape[1] == 5:
            X_train_folds[i] = X_train_folds[i][:, :4]
    if X_test_all.shape[1] == 5:
        X_test_all = X_test_all[:, :4]
    
    # Combine all train folds for TabPFN
    X_train_all = torch.cat(X_train_folds, dim=0)
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_test_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # X_enc_train_all, _, _, _ = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    print('cont_cols:', cont_cols)
    print('cat_cols:', cat_cols)
    print('source_cols:', source_cols)
    # Encode each fold individually for GP training
    X_train_folds_enc = []
    for fold_data in X_train_folds:
        fold_enc, _, _, _ = encode_qual_data(fold_data, qual_dict=qual_dict, source_col=None)
        X_train_folds_enc.append(fold_enc)
    
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across folds

    # Debug: Print categorical distributions for each fold
    print(f"\n{'='*20} PRE-STRATIFIED FOLDS VERIFICATION {'='*20}")
    for fold in range(min(3, num_folds)):  # Show first 3 folds
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
    
    print(f"{'='*60}")
        
    total_start_time = time.time()
    for i in range(num_folds):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Use pre-generated fold
        X_train = X_train_folds_enc[i]
        y_train = y_train_folds[i]
        
        # Prepare data (standardization) - ALWAYS DO THIS for both GP and PFN
        # Convert to torch dtype and optionally standardize X
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=gp_dtype)
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
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])

        # Normalize the GP data
        # Print unscaled y_train statistics
        print(f"\n--- Fold {i+1}/{num_folds} y_train statistics (unscaled) ---")
        print(f"  Mean: {y_train.mean().item():.6f}")
        print(f"  Std: {y_train.std().item():.6f}")
        print(f"  Median: {y_train.median().item():.6f}")
        print(f"  Min: {y_train.min().item():.6f}")
        print(f"  Max: {y_train.max().item():.6f}")
        
        if standardize_y_log_scale:
            Yscaler = gpplus.utils.LogScaler()#C=y_train.median().item())
        else:
            Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)
        # Get C from LogScaler if using log scale to ensure it matches
        log_scale_C = Yscaler.C if standardize_y_log_scale else None
        
        # Print scaled y_train statistics
        if standardize_y_log_scale:
            # Compute log-space values before standardization for display
            y_train_log_space = torch.log(y_train + log_scale_C)
            print(f"\n--- Fold {i+1}/{num_folds} y_train statistics (log-scaled) ---")
            print(f"  LogScaler C: {log_scale_C}")
            print(f"  Log-scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Log-scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Log-scaled Median: {y_train_normal.median().item():.6f}")
            print(f"  Log-scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Log-scaled Max: {y_train_normal.max().item():.6f}")
            print(f"  Log-space mean (before standardization): {y_train_mean.item() if y_train_mean.numel() == 1 else y_train_mean.squeeze().item():.6f}")
            print(f"  Log-space std (before standardization): {y_train_std.item() if y_train_std.numel() == 1 else y_train_std.squeeze().item():.6f}")
            print(f"  Log-space median (before standardization): {y_train.median().item() if y_train.median().numel() == 1 else y_train.median().squeeze().item():.6f}")
            print(f"  Log-space min (before standardization): {y_train_log_space.min().item():.6f}")
            print(f"  Log-space max (before standardization): {y_train_log_space.max().item():.6f}")
        else:
            print(f"\n--- Fold {i+1}/{num_folds} y_train statistics (standard-scaled) ---")
            print(f"  Scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Scaled Median: {y_train_normal.median().item():.6f}")
            print(f"  Scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Scaled Max: {y_train_normal.max().item():.6f}")
        
        # =============================================================================
        # GP Section 
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} GP Training ---")
            
            if MF_kernel:
                kenrel = defaults.MF_kernel(
                    cont_cols=cont_cols,
                    cat_cols=cat_cols,
                    source_cols=source_cols,
                )
            else:
                kenrel = defaults.SF_kernel

            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kenrel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_folds - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                if standardize_y_log_scale:
                    print(f"LogScaler C: {log_scale_C}")
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
                standardize_y_log_scale=standardize_y_log_scale,
                log_scale_C=log_scale_C,
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

            print(f"\nGP Results (Fold {i+1}/{num_folds})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_pfn and run_models in [None, 'pfn']:
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
                standardize_y_log_scale=standardize_y_log_scale,
                log_scale_C=log_scale_C,
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
            
            if run_models in [None, 'gp']:
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
                    "standardize_y_log_scale": standardize_y_log_scale,
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
            tabpfn_model_info = None
            if run_pfn and run_models in [None, 'pfn']:
                tabpfn_model_info = {
                    "model_path": regressor.model_path,
                    "fit_mode": regressor.fit_mode,
                    "device": str(regressor.device),
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
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, 'gp'] else None
    TabPFN_summary = None
    if run_pfn and run_models in [None, 'pfn']:
        TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        if run_pfn and run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if run_models is not None:
                file_prefix = run_models
            else:
                file_prefix = "gpVpfn"
            
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {}
            if run_models in [None, 'gp']:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info
                }
            if run_pfn and run_models in [None, 'pfn']:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info,
                }
            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
        
        # Save trainer info if trainer_info is enabled
        if trainer_info and GPTrainer_info:
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
                
                # Save trainer info JSON
                trainer_info_file = trainer_analysis_dir / f"gpVpfn_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                
            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # buckling_SF_GPvsPFN(num_folds=5, train_size=20, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp')
    buckling_SF_GPvsPFN(num_folds=1, train_size=20, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', run_pfn=False)
    # buckling_SF_GPvsPFN(num_folds=1, train_size=20, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', title = "SF_kernel", MF_kernel=False)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=False)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=False)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', encode_PFN_data=True)
    # buckling_GPvsPFN(num_folds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', encode_PFN_data=False)



