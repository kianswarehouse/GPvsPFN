import json
from pathlib import Path

import numpy as np
import torch
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
import matplotlib.pyplot as plt
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.training.eval2 import evaluate_gp_model
from tabpfn import TabPFNRegressor
from load_experimental_data import generate_zakharov_data
import defaults

# import warnings
# warnings.filterwarnings("ignore")
def zakharov_GPvsPFN(num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10,  # total training size is train_size * number of X input dimensions
        dimensions=5,
        x_bounds=[-5, 10],
        num_inits=defaults.TRAINER_NUM_INITS,
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        min_epochs=defaults.TRAINER_MIN_EPOCHS,
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/zakharov',
        title=None,
        standardize_X=defaults.STANDARDIZE_X,
        standardize_y=defaults.STANDARDIZE_Y,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        standardize_y_log_scale=True,
        log_y_epsilon=1e-8,
        log_y_C=None,
        log_y_point_inverse: str = "mean",
        compare_log_point_inverses: bool = False,
        noise_train=0.0,
        noise_test=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
        trainer_info=True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
        log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
        single_dataset=True,
        plot_train_y_hist=True,
        plot_train_y_scaled_hist=True,
        plot_pred_scatter=True,
        hist_bins=30,
        # If True: one Sobol train set (and test set) for every run; run_seed still varies.
        # If False: draw a larger train pool, shuffle, and give each run a disjoint train slice (legacy).
    ):

    if run_models == 'pfn':
        num_inits = 0

    if title is None:
        title = f"Zakharov_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else:
        title = f"Zakharov_{title}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device, random_state=seed)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None

    # Generate data
    set_seed(seed)
    
    train_per_run = train_size * dimensions  # train_size * dimensions for Zakharov
    if single_dataset:
        n_train_generate = train_per_run
        total_samples = num_test + n_train_generate
        print(
            f"Generating {total_samples} unique Sobol samples for {dimensions}D Zakharov function\n\t"
            f"Test samples: {num_test} / Train samples: {train_per_run} "
            f"(single_dataset=True: same train data for all {num_runs} runs)"
        )
    else:
        num_runs_gen = max(num_runs, 20)
        n_train_generate = num_runs_gen * train_per_run
        total_samples = num_test + n_train_generate
        print(
            f"Generating {total_samples} unique Sobol samples for {dimensions}D Zakharov function\n\t"
            f"Test samples: {num_test} / Train pool: {n_train_generate} "
            f"(single_dataset=False: disjoint train slices, {train_per_run} points per run)"
        )

    # Generate train and test data in one call
    X_train_all, y_train_all, X_test_all, y_test_all = generate_zakharov_data(
        n_train=n_train_generate,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed
    )
    if plot_train_y_hist and save_path is not None:
        try:
            plot_dir = Path(plot_save_path)
            plot_dir.mkdir(parents=True, exist_ok=True)
            hist_path = plot_dir / f"y_train_hist_{title}.png"
            y_vals = y_train_all.detach().cpu().reshape(-1).numpy()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(y_vals, bins=hist_bins, alpha=0.85, color="steelblue", edgecolor="black")
            ax.set_title(f"{title}\nZakharov training y histogram")
            ax.set_xlabel("y_train")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved training y histogram: {hist_path}")
        except Exception as e:
            print(f"Saving training y histogram failed: {e}")
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []
    TabPFN_metrics_alt = []
    GPPlus_metrics_alt = []
    GPTrainer_info = []  # Accumulate trainer logs across runs

    if not single_dataset:
        torch.manual_seed(seed)
        all_indices = torch.randperm(n_train_generate)
        train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)

    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        if single_dataset:
            X_train = X_train_all
            y_train = y_train_all
        else:
            run_train_indices = train_indices_2d[i]
            X_train = X_train_all[run_train_indices]
            y_train = y_train_all[run_train_indices]

        # Prepare data (standardization) - ALWAYS DO THIS for both GP and PFN
        # Reuse PFN split, convert to torch (unified)
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        X_train_raw_for_pfn = X_train.detach().clone()
        X_test_raw_for_pfn = X_test.detach().clone()
        # Determine X scaling type
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
        else:
            X_scaling_type = "None"

        # Normalize the data
        y_train_min_pre_scale = float(y_train.min().item())
        print(f"  y_train min (before target scaling): {y_train_min_pre_scale:.6f}")
        if standardize_y_log_scale:
            if log_y_C is not None:
                computed_log_y_C = float(log_y_C)
            else:
                if y_train.min().item() < 0:
                    computed_log_y_C = float((-y_train.min()).item() + 2 * (y_train.std()).item() + log_y_epsilon)
                else:
                    computed_log_y_C = float(log_y_epsilon)
            Yscaler = gpplus.utils.LogScaler(epsilon=log_y_epsilon, C=computed_log_y_C)
        else:
            Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)
        log_scale_C = Yscaler.C if standardize_y_log_scale else None
        y_scaled = standardize_y or standardize_y_log_scale

        if standardize_y_log_scale:
            y_train_log_space = torch.log(y_train + log_scale_C)
            print(f"\n--- Run {i+1}/{num_runs} y_train statistics (log-scaled) ---")
            print(f"  LogScaler C: {log_scale_C}")
            print(f"  Log-scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Log-scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Log-scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Log-scaled Max: {y_train_normal.max().item():.6f}")
            print(f"  Log-space min (before standardization): {y_train_log_space.min().item():.6f}")
            print(f"  Log-space max (before standardization): {y_train_log_space.max().item():.6f}")

        if (
            plot_train_y_scaled_hist
            and save_path is not None
            and i == 0
            and y_scaled
        ):
            try:
                plot_dir = Path(plot_save_path)
                plot_dir.mkdir(parents=True, exist_ok=True)
                if standardize_y_log_scale:
                    transform_note = "log(y+C) then z-scored (model targets)"
                    hist_fname = f"y_train_scaled_log_{title}.png"
                else:
                    transform_note = "z-scored y (model targets)"
                    hist_fname = f"y_train_scaled_z_{title}.png"
                scaled_path = plot_dir / hist_fname
                scaled_vals = y_train_normal.detach().cpu().reshape(-1).numpy()
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(scaled_vals, bins=hist_bins, alpha=0.85, color="darkorange", edgecolor="black")
                ax.set_title(f"{title}\nTraining targets after scaling\n({transform_note})")
                ax.set_xlabel("y_train_normal (fed to GP / TabPFN)")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(scaled_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved scaled training y histogram: {scaled_path}")
            except Exception as e:
                print(f"Saving scaled training y histogram failed: {e}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} GP Training ---")

            model = gpplus.models.GPR(
                X_train,
                y_train_normal if y_scaled else y_train,
                kernel_module=defaults.SF_kernel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_runs - 1):
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
                seed=run_seed,
                num_inits=num_inits,
                lr=lr,
                convergence_patience=convergence_patience,
                min_epochs=min_epochs,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                optimizer_kwargs=optimizer_kwargs,
                initializer_class=initializer_class,
                device=gp_device,
                y_train_mean=y_train_mean if y_scaled else None,
                y_train_std=y_train_std if y_scaled else None,
                standardize_y_log_scale=standardize_y_log_scale,
                log_scale_C=log_scale_C if standardize_y_log_scale else None,
                log_y_point_inverse=log_y_point_inverse,
                source_cols=source_cols,
                trainer_info=trainer_info,
                callbacks=defaults.get_default_gp_callbacks(
                    optimizer_class,
                    callback_save_path=callback_save_path,
                    log_lbfgs_inner=log_lbfgs_inner,
                ),
                callback_save_path=callback_save_path,
                log_lbfgs_inner=log_lbfgs_inner,
            )
            GPPlus_metrics.append(gp_metric)
            gp_metric_alt = None
            if False and compare_log_point_inverses and standardize_y_log_scale:
                alt_inverse = "median" if log_y_point_inverse == "mean" else "mean"
                gp_metric_alt, _, _, _ = train_eval_gp(
                    model,
                    X_test,
                    y_test,
                    num_epochs=num_epochs,
                    seed=run_seed,
                    num_inits=num_inits,
                    lr=lr,
                    convergence_patience=convergence_patience,
                    min_epochs=min_epochs,
                    min_loss_change=min_loss_change,
                    optimizer_class=optimizer_class,
                    optimizer_kwargs=optimizer_kwargs,
                    initializer_class=initializer_class,
                    device=gp_device,
                    y_train_mean=y_train_mean if y_scaled else None,
                    y_train_std=y_train_std if y_scaled else None,
                    standardize_y_log_scale=True,
                    log_scale_C=log_scale_C,
                    log_y_point_inverse=alt_inverse,
                    source_cols=source_cols,
                    trainer_info=False,
                    callbacks=[],
                    log_lbfgs_inner=log_lbfgs_inner,
                )
                GPPlus_metrics_alt.append(gp_metric_alt)
                print(
                    f"[GP compare] primary={log_y_point_inverse} RMSE={gp_metric.get('RMSE')} | "
                    f"alt={alt_inverse} RMSE={gp_metric_alt.get('RMSE')}"
                )
                for k in ("RRMSE", "RMSE", "MAE", "CRPS", "NIS", "NCRPS"):
                    if k in gp_metric_alt:
                        gp_metric[f"{k}_{alt_inverse}"] = gp_metric_alt[k]

            if (i == 0) and standardize_y_log_scale and run_models in [None, "gp"]:
                _yt = y_test.detach().cpu().numpy().reshape(-1)
                _yp = np.asarray(y_pred_gp, dtype=np.float64).reshape(-1)
                assert _yt.shape[0] == _yp.shape[0] == X_test.shape[0]
                print(
                    f"[Zakharov eval sanity] n_test={_yt.shape[0]} (matches X_test rows); "
                    f"y_true min/median/max={_yt.min():.4g} / {np.median(_yt):.4g} / {_yt.max():.4g}; "
                    f"y_pred min/median/max={np.nanmin(_yp):.4g} / {np.nanmedian(_yp):.4g} / {np.nanmax(_yp):.4g}"
                )
            
            # Accumulate trainer info if available
            if gp_trainer_info:
                # Add fold information to trainer log
                gp_trainer_info["fold"] = i + 1
                gp_trainer_info["metrics"] = gp_metric  # Include metrics for this fold
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nGP Results (Fold {i+1}/{num_runs})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
            if plot_pred_scatter and save_path is not None:
                try:
                    plot_dir = Path(plot_save_path)
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    y_true_np = y_test.detach().cpu().reshape(-1).numpy()
                    y_pred_np = np.asarray(y_pred_gp).reshape(-1)
                    lim_min = float(min(np.min(y_true_np), np.min(y_pred_np)))
                    lim_max = float(max(np.max(y_true_np), np.max(y_pred_np)))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_true_np, y_pred_np, s=10, alpha=0.5, color="royalblue")
                    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1)
                    ax.set_title(f"{title}\nGP y_pred vs y_true (original scale)")
                    ax.set_xlabel("y_true")
                    ax.set_ylabel("y_pred")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    orig_path = plot_dir / f"scatter_gp_original_run{i+1}_{title}.png"
                    fig.savefig(orig_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved GP scatter (original): {orig_path}")
                    if standardize_y_log_scale and (log_scale_C is not None):
                        y_true_shift = y_true_np + float(log_scale_C)
                        y_pred_shift = y_pred_np + float(log_scale_C)
                        valid = (
                            np.isfinite(y_true_shift)
                            & np.isfinite(y_pred_shift)
                            & (y_true_shift > 0)
                            & (y_pred_shift > 0)
                        )
                        if np.any(valid):
                            y_true_log = np.log(y_true_shift[valid])
                            y_pred_log = np.log(y_pred_shift[valid])
                            lmin = float(min(np.min(y_true_log), np.min(y_pred_log)))
                            lmax = float(max(np.max(y_true_log), np.max(y_pred_log)))
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.scatter(y_true_log, y_pred_log, s=10, alpha=0.5, color="darkorange")
                            ax.plot([lmin, lmax], [lmin, lmax], "k--", lw=1)
                            ax.set_title(f"{title}\nGP y_pred vs y_true (log scale)")
                            ax.set_xlabel("log(y_true + C)")
                            ax.set_ylabel("log(y_pred + C)")
                            ax.grid(True, alpha=0.25)
                            fig.tight_layout()
                            log_path = plot_dir / f"scatter_gp_log_run{i+1}_{title}.png"
                            fig.savefig(log_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"Saved GP scatter (log): {log_path}")
                    # Training scatter (GP): predict on X_train with trained model.
                    y_train_pred_std, _, _, y_train_pred_stddev = evaluate_gp_model(model, X_train)
                    y_train_true_np = y_train.detach().cpu().reshape(-1).numpy()
                    y_train_pred_std_np = y_train_pred_std.detach().cpu().reshape(-1).numpy()
                    y_train_pred_stddev_np = y_train_pred_stddev.detach().cpu().reshape(-1).numpy()
                    if standardize_y_log_scale and (log_scale_C is not None):
                        mean_val = float(y_train_mean.squeeze().item()) if hasattr(y_train_mean, "squeeze") else float(y_train_mean)
                        std_val = float(y_train_std.squeeze().item()) if hasattr(y_train_std, "squeeze") else float(y_train_std)
                        log_mu = (y_train_pred_std_np * std_val) + mean_val
                        if log_y_point_inverse == "mean":
                            log_sig = y_train_pred_stddev_np * std_val
                            log_sig_eff = np.minimum(log_sig, 3.0)
                            y_train_pred_np = np.exp(log_mu + 0.5 * (log_sig_eff**2)) - float(log_scale_C)
                        else:
                            y_train_pred_np = np.exp(log_mu) - float(log_scale_C)
                    elif y_scaled:
                        mean_val = float(y_train_mean.squeeze().item()) if hasattr(y_train_mean, "squeeze") else float(y_train_mean)
                        std_val = float(y_train_std.squeeze().item()) if hasattr(y_train_std, "squeeze") else float(y_train_std)
                        y_train_pred_np = (y_train_pred_std_np * std_val) + mean_val
                    else:
                        y_train_pred_np = y_train_pred_std_np
                    tmin = float(min(np.min(y_train_true_np), np.min(y_train_pred_np)))
                    tmax = float(max(np.max(y_train_true_np), np.max(y_train_pred_np)))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_train_true_np, y_train_pred_np, s=12, alpha=0.55, color="navy")
                    ax.plot([tmin, tmax], [tmin, tmax], "k--", lw=1)
                    ax.set_title(f"{title}\nGP y_pred vs y_true (train, original scale)")
                    ax.set_xlabel("y_train_true")
                    ax.set_ylabel("y_train_pred")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    train_orig_path = plot_dir / f"scatter_gp_train_original_run{i+1}_{title}.png"
                    fig.savefig(train_orig_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved GP scatter (train/original): {train_orig_path}")
                    if standardize_y_log_scale and (log_scale_C is not None):
                        y_train_true_shift = y_train_true_np + float(log_scale_C)
                        y_train_pred_shift = y_train_pred_np + float(log_scale_C)
                        valid_t = (
                            np.isfinite(y_train_true_shift)
                            & np.isfinite(y_train_pred_shift)
                            & (y_train_true_shift > 0)
                            & (y_train_pred_shift > 0)
                        )
                        if np.any(valid_t):
                            y_train_true_log = np.log(y_train_true_shift[valid_t])
                            y_train_pred_log = np.log(y_train_pred_shift[valid_t])
                            ltmin = float(min(np.min(y_train_true_log), np.min(y_train_pred_log)))
                            ltmax = float(max(np.max(y_train_true_log), np.max(y_train_pred_log)))
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.scatter(y_train_true_log, y_train_pred_log, s=12, alpha=0.55, color="darkorange")
                            ax.plot([ltmin, ltmax], [ltmin, ltmax], "k--", lw=1)
                            ax.set_title(f"{title}\nGP y_pred vs y_true (train, log scale)")
                            ax.set_xlabel("log(y_train_true + C)")
                            ax.set_ylabel("log(y_train_pred + C)")
                            ax.grid(True, alpha=0.25)
                            fig.tight_layout()
                            train_log_path = plot_dir / f"scatter_gp_train_log_run{i+1}_{title}.png"
                            fig.savefig(train_log_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"Saved GP scatter (train/log): {train_log_path}")
                except Exception as e:
                    print(f"Saving GP scatter plots failed: {e}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_models in [None, 'pfn']:
            print(f"\n--- {title} TabPFN Training ---")
            
            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
                X_train_raw_for_pfn,
                X_test_raw_for_pfn,
                y_train,
                y_test,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                source_cols=source_cols,
            )
            TabPFN_metrics.append(tabpfn_metric)
            # PFN always runs on raw targets now; disable alternate inverse compare path
            # that is only meaningful for transformed log-target evaluation.

            # Print results for this run
            print(f"\nTabPFN Results (Run {i+1}/{num_runs})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
            if plot_pred_scatter and save_path is not None:
                try:
                    plot_dir = Path(plot_save_path)
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    y_true_np = y_test.detach().cpu().reshape(-1).numpy()
                    y_pred_np = np.asarray(y_pred_tabpfn).reshape(-1)
                    lim_min = float(min(np.min(y_true_np), np.min(y_pred_np)))
                    lim_max = float(max(np.max(y_true_np), np.max(y_pred_np)))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_true_np, y_pred_np, s=10, alpha=0.5, color="seagreen")
                    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1)
                    ax.set_title(f"{title}\nPFN y_pred vs y_true (original scale)")
                    ax.set_xlabel("y_true")
                    ax.set_ylabel("y_pred")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    orig_path = plot_dir / f"scatter_pfn_original_run{i+1}_{title}.png"
                    fig.savefig(orig_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved PFN scatter (original): {orig_path}")
                    if standardize_y_log_scale and (log_scale_C is not None):
                        y_true_shift = y_true_np + float(log_scale_C)
                        y_pred_shift = y_pred_np + float(log_scale_C)
                        valid = (
                            np.isfinite(y_true_shift)
                            & np.isfinite(y_pred_shift)
                            & (y_true_shift > 0)
                            & (y_pred_shift > 0)
                        )
                        if np.any(valid):
                            y_true_log = np.log(y_true_shift[valid])
                            y_pred_log = np.log(y_pred_shift[valid])
                            lmin = float(min(np.min(y_true_log), np.min(y_pred_log)))
                            lmax = float(max(np.max(y_true_log), np.max(y_pred_log)))
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.scatter(y_true_log, y_pred_log, s=10, alpha=0.5, color="darkorange")
                            ax.plot([lmin, lmax], [lmin, lmax], "k--", lw=1)
                            ax.set_title(f"{title}\nPFN y_pred vs y_true (log scale)")
                            ax.set_xlabel("log(y_true + C)")
                            ax.set_ylabel("log(y_pred + C)")
                            ax.grid(True, alpha=0.25)
                            fig.tight_layout()
                            log_path = plot_dir / f"scatter_pfn_log_run{i+1}_{title}.png"
                            fig.savefig(log_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"Saved PFN scatter (log): {log_path}")
                    # Training scatter (PFN): predict on X_train if API available.
                    try:
                        X_train_np = X_train.detach().cpu().numpy()
                        y_train_true_np = y_train.detach().cpu().reshape(-1).numpy()
                        y_train_pred_std_np = np.asarray(regressor.predict(X_train_np)).reshape(-1)
                        if standardize_y_log_scale and (log_scale_C is not None):
                            mean_val = float(y_train_mean.squeeze().item()) if hasattr(y_train_mean, "squeeze") else float(y_train_mean)
                            std_val = float(y_train_std.squeeze().item()) if hasattr(y_train_std, "squeeze") else float(y_train_std)
                            log_mu = (y_train_pred_std_np * std_val) + mean_val
                            y_train_pred_np = np.exp(log_mu) - float(log_scale_C)
                        elif y_scaled:
                            mean_val = float(y_train_mean.squeeze().item()) if hasattr(y_train_mean, "squeeze") else float(y_train_mean)
                            std_val = float(y_train_std.squeeze().item()) if hasattr(y_train_std, "squeeze") else float(y_train_std)
                            y_train_pred_np = (y_train_pred_std_np * std_val) + mean_val
                        else:
                            y_train_pred_np = y_train_pred_std_np
                        tmin = float(min(np.min(y_train_true_np), np.min(y_train_pred_np)))
                        tmax = float(max(np.max(y_train_true_np), np.max(y_train_pred_np)))
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.scatter(y_train_true_np, y_train_pred_np, s=12, alpha=0.55, color="darkgreen")
                        ax.plot([tmin, tmax], [tmin, tmax], "k--", lw=1)
                        ax.set_title(f"{title}\nPFN y_pred vs y_true (train, original scale)")
                        ax.set_xlabel("y_train_true")
                        ax.set_ylabel("y_train_pred")
                        ax.grid(True, alpha=0.25)
                        fig.tight_layout()
                        train_orig_path = plot_dir / f"scatter_pfn_train_original_run{i+1}_{title}.png"
                        fig.savefig(train_orig_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print(f"Saved PFN scatter (train/original): {train_orig_path}")
                        if standardize_y_log_scale and (log_scale_C is not None):
                            y_train_true_shift = y_train_true_np + float(log_scale_C)
                            y_train_pred_shift = y_train_pred_np + float(log_scale_C)
                            valid_t = (
                                np.isfinite(y_train_true_shift)
                                & np.isfinite(y_train_pred_shift)
                                & (y_train_true_shift > 0)
                                & (y_train_pred_shift > 0)
                            )
                            if np.any(valid_t):
                                y_train_true_log = np.log(y_train_true_shift[valid_t])
                                y_train_pred_log = np.log(y_train_pred_shift[valid_t])
                                ltmin = float(min(np.min(y_train_true_log), np.min(y_train_pred_log)))
                                ltmax = float(max(np.max(y_train_true_log), np.max(y_train_pred_log)))
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.scatter(y_train_true_log, y_train_pred_log, s=12, alpha=0.55, color="darkorange")
                                ax.plot([ltmin, ltmax], [ltmin, ltmax], "k--", lw=1)
                                ax.set_title(f"{title}\nPFN y_pred vs y_true (train, log scale)")
                                ax.set_xlabel("log(y_train_true + C)")
                                ax.set_ylabel("log(y_train_pred + C)")
                                ax.grid(True, alpha=0.25)
                                fig.tight_layout()
                                train_log_path = plot_dir / f"scatter_pfn_train_log_run{i+1}_{title}.png"
                                fig.savefig(train_log_path, dpi=150, bbox_inches="tight")
                                plt.close(fig)
                                print(f"Saved PFN scatter (train/log): {train_log_path}")
                    except Exception as e_train:
                        print(f"Saving PFN train scatter plots skipped: {e_train}")
                except Exception as e:
                    print(f"Saving PFN scatter plots failed: {e}")
        
        # Collect model info from first fold
        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            if run_models in [None, 'gp']:
                _log_c_rule = (
                    ("user log_y_C" if log_y_C is not None else "C = -min(y_train) + eps")
                    if standardize_y_log_scale
                    else None
                )
                _y_pt_interp = ""
                if standardize_y_log_scale:
                    if log_y_point_inverse == "mean":
                        _y_pt_interp = (
                            "For log-shift runs, point metrics use E[y]=exp(mu_log+sigma_log^2/2)-C on the test set; "
                            "95% intervals use log-space quantiles then exp(.)-C. "
                        )
                    else:
                        _y_pt_interp = (
                            "For log-shift runs, point metrics use exp(mu_log)-C (median-style); "
                            "95% intervals use log-space quantiles then exp(.)-C. "
                        )
                else:
                    _y_pt_interp = "Linear y denormalization when y is scaled."
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
                "standardize_y_log_scale": standardize_y_log_scale,
                "log_y_epsilon": log_y_epsilon,
                "log_y_C": log_y_C,
                "log_y_point_inverse": log_y_point_inverse,
                "compare_log_point_inverses": compare_log_point_inverses,
                "log_y_C_rule": _log_c_rule,
                "log_scale_C": float(log_scale_C) if log_scale_C is not None else None,
                "y_point_prediction_interpretation": _y_pt_interp,
                "X_scaling_type": X_scaling_type,
                "x_standardize_method": x_standardize_method,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_inits": num_inits,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                "x_bounds": x_bounds,
                **y_test_stats,
                "num_runs": num_runs,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                }
            
            if run_models in [None, 'pfn']:
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
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if run_models in [None, 'pfn'] else None
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, 'gp'] else None
    TabPFN_summary_alt = analyze_metrics(TabPFN_metrics_alt, print_summary=False, label=f"TabPFN_alt", title=title) if TabPFN_metrics_alt else None
    GPPlus_summary_alt = analyze_metrics(GPPlus_metrics_alt, print_summary=False, label=f"GP_alt", title=title) if GPPlus_metrics_alt else None
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        if run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
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
                file_prefix = "gpVpfn"
            
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {}
            if run_models in [None, 'gp']:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "summary_alt": GPPlus_summary_alt,
                    "metrics_alt": GPPlus_metrics_alt,
                    "gp_model_info": gp_model_info
                }
            if run_models in [None, 'pfn']:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "summary_alt": TabPFN_summary_alt,
                    "metrics_alt": TabPFN_metrics_alt,
                    "pfn_model_info": tabpfn_model_info
                }
            # Append defaults.py source at end of JSON for reproducibility
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
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
                trainer_info_by_run = {
                    f"run_{entry.get('run', i + 1)}": entry
                    for i, entry in enumerate(GPTrainer_info)
                }
                trainer_info_data = {
                    "title": title,
                    "num_runs": num_runs,
                    "num_inits_per_run": num_inits,
                    "trainer_info": trainer_info_by_run,
                }
                
                # Save trainer info JSON (always use "gp" prefix for GP trainer info)
                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                try:
                    from plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data
                    plot_trainer_analysis_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except Exception as plot_e:
                    print(f"Trainer analysis plotting skipped: {plot_e}")
                try:
                    from plot_epoch_metrics import plot_iter_metrics_from_data
                    plot_iter_metrics_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except ValueError:
                    pass  # no epoch_metrics in data
                except Exception as e:
                    print(f"Epoch metrics plotting skipped: {e}")
                
            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of inits: {num_inits}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\tX_scaling_type: {X_scaling_type}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\truns: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    zakharov_GPvsPFN(
        num_runs=10,
        train_size=20,
        dimensions=20,
        num_inits=16,
        noise_train=0.002,
        noise_test=0.002,
        save_path="./results/zakharov/temp",
        run_models=None,
    )
    # zakharov_GPvsPFN(num_runs=1, train_size=20, dimensions=20, num_runs=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=40, dimensions=20, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=80, dimensions=20, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=10, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=20, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=40, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=80, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')












