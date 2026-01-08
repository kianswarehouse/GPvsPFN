import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import load_dns_rom_data_all
from sklearn.model_selection import train_test_split
import numpy as np
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def DNS_ROM_MF_GPvsPFN(num_folds=defaults.NUM_FOLDS,
        test_size=0.2,
        num_runs=defaults.TRAINER_NUM_RUNS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/DNS_ROM',
        title=None,
        encode_data=True,
        standardize_X=True,
        standardize_y=True,
        standardization_method=defaults.MF_STANDARDIZATION_METHOD,
        x_standardize_method=0,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype=defaults.DTYPE_GP,
        pfn_dtype=defaults.DTYPE_PFN,
        trainer_info=True,
    ):
    if title is None:
        title = f"DNS_ROM_MF_{test_size}test_{num_runs}runs"
    else: 
        title = f"DNS_ROM_MF_{title}_{test_size}test_{num_runs}runs"

    # Load data
    set_seed(seed)   

    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None
    
    # Load all DNS ROM data
    print("Loading DNS ROM data...")
    X, y = load_dns_rom_data_all(print_info=True)
    
    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)
    print(f"Total samples: {len(X)}")
    print(f"X dimensions: {X.shape[1]}")

    # Prepare encoded data once
    if encode_data:
        qual_dict = learn_encodings(X, cont_cols=[1])
        print(qual_dict)
        X_enc, cont_cols, cat_cols, source_cols = encode_qual_data(X, qual_dict=qual_dict, source_col=-1)
        print(f"cont_cols: {cont_cols}")
        print(f"cat_cols: {cat_cols}")
        print(f"source_cols: {source_cols}")
    else:
        cont_cols = list(range(6))  # First 6 columns are continuous features
        cat_cols = []
        source_cols = [6]  # Last column is source
        qual_dict = {}
    
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across folds
    gp_model_info = None
    tabpfn_model_info = None
    
    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Train/test split on ALL data (all sources mixed) - like am_data
        # Use different seed for each fold to get different splits
        fold_seed = seed + i
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        
        # Convert to numpy for sklearn train_test_split
        X_np = X_enc.numpy() if encode_data else X.numpy()
        y_np = y.numpy()
        
        # Train/test split on all data together
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=fold_seed
        )
        
        # Convert back to torch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Test y mean: {y_test.mean().item():.4f}, Test y std: {y_test.std().item():.4f}")
        
        # Get source info for verification
        # After encoding, source is one-hot encoded, so we need to find which column is 1
        if source_cols and len(source_cols) > 0:
            # source_cols is a list of indices for the one-hot encoded source columns
            source_onehot_train = X_train[:, source_cols]  # Shape: (N, 4)
            source_onehot_test = X_test[:, source_cols]    # Shape: (N, 4)
            # Get the source index by finding which column is 1
            source_train = torch.argmax(source_onehot_train, dim=1)  # Shape: (N,)
            source_test = torch.argmax(source_onehot_test, dim=1)    # Shape: (N,)
        else:
            # If not encoded, source is in the last column
            source_train = X_train[:, -1].long()
            source_test = X_test[:, -1].long()
        
        # Verify source distribution
        source_counts_train = [torch.sum(source_train == j).item() for j in range(4)]
        source_counts_test = [torch.sum(source_test == j).item() for j in range(4)]
        print(f"Train source distribution: {source_counts_train}")
        print(f"Test source distribution: {source_counts_test}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Use unified X_train/X_test for GP as well
        X_train_gp = X_train.detach().clone().to(dtype=gp_dtype)
        X_test_gp = X_test.detach().clone().to(dtype=gp_dtype)
        y_train_gp = y_train.detach().clone().to(dtype=gp_dtype)
        y_test_gp = y_test.detach().clone().to(dtype=gp_dtype)
        # Get high-fidelity mask for standardization
        
        X_train_gp, X_test_gp, y_train_normal, y_train_mean, y_train_std = gpplus.utils.standardize_mf_data(
            X_train_gp,
            X_test_gp,
            y_train_gp,
            cont_cols,
            source_cols,
            standardize_X=standardize_X,
            standardize_y=standardize_y,
            standardization_method=standardization_method,
            x_standardize_method=x_standardize_method,
        )

        # cat_cols was returned by the encoder; CombinedKernel expects only cat indices
        kernel = defaults.MF_kernel(
            cont_cols=cont_cols, 
            cat_cols=cat_cols, 
            source_cols=source_cols,
        )
        
        model = gpplus.models.GPR(
            X_train_gp,
            y_train_normal if standardize_y else y_train_gp,
            kernel_module=kernel,
            mean_module=defaults.MF_mean(encoded_cols=source_cols),
            likelihood=defaults.MF_likelihood(encoded_cols=source_cols, training_data=X_train_gp),
        )
        if (i == 0) or (i == num_folds - 1):
            print(f"X_train: {X_train_gp.shape}")
            print(f"X_test: {X_test_gp.shape}")
            print(f"y_test mean: {y_test_gp.mean().item()} / y_test std: {y_test_gp.std().item()}")
            print(model)

        # Create trainer
        gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
            model,
            X_test_gp,
            y_test_gp,
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
            X_train_gp,
            X_test_gp,
            y_train_normal if standardize_y else y_train_gp,
            y_test_gp,
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
            # Calculate y_test mean and std for this fold
            y_test_stats = {
                "y_test_mean": float(y_test_gp.mean().item()),
                "y_test_std": float(y_test_gp.std().item())
            }
            # For test data, source is in source_cols
            source_col_idx = source_cols[0] if source_cols else -1
            source_indices_test = X_test_gp[:, source_col_idx].long()
            unique_sources = torch.unique(source_indices_test)
            if len(unique_sources) > 1:
                # MF: Also calculate per-source stats
                for source_idx in unique_sources:
                    source_mask = source_indices_test == source_idx
                    y_test_source = y_test_gp[source_mask]
                    y_test_stats[f"y_test_mean_source_{source_idx.item()}"] = float(y_test_source.mean().item())
                    y_test_stats[f"y_test_std_source_{source_idx.item()}"] = float(y_test_source.std().item())
            
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train_gp.shape[1],
                "test_size": test_size,
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
                "encoded_data": encode_data,
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
    print(f"Experiment details: \n\ttest_size: {test_size}\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    DNS_ROM_MF_GPvsPFN(num_folds=1, test_size=0.2, num_runs=4, standardization_method=2, num_epochs=10000, save_path="./results/DNS_ROM/temp")