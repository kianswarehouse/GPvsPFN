"""
CV-GP vs PFN for Griewank (with num_folds):

- num_folds: number of train partitions (replicates); total_train = num_folds * (train_size * dimensions).
- For each fold: that fold's training data is split into CV_folds parts. Train one GP per part
  with num_runs each; collect θ1..θ_CV_folds. Then train one GP on the full fold data with
  exactly CV_folds initializations (from those θ's). TabPFN on same full fold data.
- Results: one GP and one PFN metric per fold; summaries and JSON over num_folds.

"""
import torch
import json
import time
from pathlib import Path

from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics, compute_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.training import FixedParameterInitializer
from gpplus.training.hyperparameter_utils import set_hyperparameters_from_final_dict
from gpplus.training.eval2 import evaluate_gp_model
from tabpfn import TabPFNRegressor

from load_experimental_data import generate_griewank_data
import defaults


def _get_best_final_dict(gp_trainer_info):
    """Extract the 'final' hyperparameter dict from the best run in trainer info."""
    if not gp_trainer_info:
        return None
    best = gp_trainer_info.get("best_parameters")
    if best is not None and isinstance(best.get("final_parameters"), dict):
        return best["final_parameters"]
    runs = gp_trainer_info.get("runs", [])
    if not runs:
        return None
    best_run = min(
        (r for r in runs if r.get("loss") is not None),
        key=lambda r: r.get("loss", float("inf")),
        default=None,
    )
    if best_run is not None and isinstance(best_run.get("final_parameters"), dict):
        return best_run["final_parameters"]
    for r in runs:
        fin = r.get("final_parameters") or r.get("final")
        if isinstance(fin, dict):
            return fin
    return None


def _print_cv_hyperparameters(label: str, final_dict: dict) -> None:
    """Print key hyperparameters from a CV part 'final' dict."""
    if not final_dict:
        print(f"  [{label}] (empty)")
        return
    raw_noise = final_dict.get("raw_noise")
    raw_outputscale = final_dict.get("raw_outputscale")
    raw_lengthscales = final_dict.get("raw_lengthscales")
    print(f"  --- CV part hyperparameters: {label} ---")
    print(f"  raw_noise:        {raw_noise}")
    print(f"  raw_outputscale: {raw_outputscale}")
    if raw_lengthscales is not None:
        ls = raw_lengthscales if isinstance(raw_lengthscales, (list, tuple)) else [raw_lengthscales]
        if len(ls) <= 10:
            print(f"  raw_lengthscales:  {ls}")
        else:
            print(f"  raw_lengthscales:  [{ls[0]:.4f}, {ls[1]:.4f}, ... ({len(ls)} total)]")
    print(f"  ----------------------------------------")


def _best_phase1_part_by_loss(phase1_cv_parts: list) -> tuple:
    """Return (best_part_index_0based, best_theta) from Phase 1 by lowest MLL loss."""
    best_idx = 0
    best_loss = float("inf")
    for i, part in enumerate(phase1_cv_parts):
        info = part.get("trainer_info") or {}
        loss = None
        if info.get("best_parameters") and "loss" in (info["best_parameters"] or {}):
            loss = info["best_parameters"]["loss"]
        else:
            for r in info.get("runs") or []:
                if r.get("loss") is not None:
                    if loss is None or r["loss"] < loss:
                        loss = r["loss"]
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_idx = i
    return best_idx, phase1_cv_parts[best_idx]["final_parameters"]


def _phase2_final_from_metrics(gp_metric: dict) -> dict:
    """Extract Phase 2 model state from gp_metric for trainer analysis (raw_noise, raw_outputscale, raw_lengthscales)."""
    out = {
        "raw_noise": gp_metric.get("raw_noise"),
        "raw_outputscale": gp_metric.get("outputscale"),  # gp_metric often has outputscale not raw_outputscale
        "noise": gp_metric.get("noise"),
        "outputscale": gp_metric.get("outputscale"),
        "best_epoch": gp_metric.get("best_epoch"),
        "RRMSE": gp_metric.get("RRMSE"),
    }
    # Rebuild raw_lengthscales from cont_lengthscale_0, cont_lengthscale_1, ...
    ls_list = []
    i = 0
    while f"cont_lengthscale_{i}" in gp_metric:
        ls_list.append(gp_metric[f"cont_lengthscale_{i}"])
        i += 1
    if ls_list:
        out["raw_lengthscales"] = ls_list
    return out


def _print_metrics_block(label: str, gp_metric: dict) -> None:
    """Print GP metrics: main metrics first, then final chosen hyperparameters, then rest (excluding per-part cv_part_*)."""
    print(f"\n{label}")
    # Main metrics first
    key_first = ("RRMSE", "RMSE", "Total_Time", "Training_Time", "Prediction_Time")
    seen = set()
    for k in key_first:
        if k in gp_metric:
            seen.add(k)
            v = gp_metric[k]
            if v is None:
                print(f"  {k}: None")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    # Final chosen hyperparameters: each on its own line (all lengthscales printed individually)
    chosen_scalar_keys = ("phase2_best_part", "chosen_raw_noise", "chosen_raw_outputscale", "chosen_noise", "chosen_outputscale")
    chosen_ls_key = "chosen_raw_lengthscales"
    if any(gp_metric.get(k) is not None for k in chosen_scalar_keys) or gp_metric.get(chosen_ls_key) is not None:
        print("  --- Final chosen hyperparameters (model in use) ---")
        for k in chosen_scalar_keys:
            if k not in gp_metric or gp_metric[k] is None:
                continue
            seen.add(k)
            v = gp_metric[k]
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        ls = gp_metric.get(chosen_ls_key)
        if ls is not None:
            seen.add(chosen_ls_key)
            for i, val in enumerate(ls):
                try:
                    print(f"  chosen_lengthscale_{i}: {float(val):.6f}")
                except (TypeError, ValueError):
                    print(f"  chosen_lengthscale_{i}: {val}")
        print("  ------------------------------------------------")
    # Rest sorted, but skip per-part cv_part_* so we don't list all parts
    for k, v in sorted(gp_metric.items()):
        if k in seen or k.startswith("cv_part_"):
            continue
        if v is None:
            print(f"  {k}: None")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, (list, tuple)):
            if len(v) <= 10:
                try:
                    print(f"  {k}: {[float(x) for x in v]}")
                except (TypeError, ValueError):
                    print(f"  {k}: {v}")
            else:
                try:
                    fmt = [f"{float(x):.4f}" for x in v[:2]]
                    print(f"  {k}: [{', '.join(fmt)}, ... ({len(v)} total)]")
                except (TypeError, ValueError):
                    print(f"  {k}: (list len {len(v)})")
        else:
            print(f"  {k}: {v}")


def griewankCV_GPvsPFN(
    num_test=5000,
    train_size=10,
    dimensions=20,
    num_folds=defaults.NUM_FOLDS,
    CV_folds=2,
    x_bounds=(-600, 600),
    num_runs=defaults.TRAINER_NUM_RUNS,
    num_epochs=defaults.TRAINER_NUM_EPOCHS,
    lr=defaults.TRAINER_LR,
    convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs=defaults.TRAINER_MIN_EPOCHS,
    min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device=defaults.TRAINER_GP_DEVICE,
    amp_device=defaults.TRAINER_AMP_DEVICE,
    save_path="./results/griewank",
    title=None,
    standardize_X=True,
    standardize_y=True,
    x_standardize_method=defaults.X_STANDARDIZE_METHOD,
    noise_train=0.0,
    noise_test=0.0,
    noise_type=defaults.NOISE_TYPE,
    seed=defaults.SEED,
    seed_trainer=defaults.SEED_TRAINER,
    gp_dtype=defaults.DTYPE_GP,
    pfn_dtype=defaults.DTYPE_PFN,
    trainer_info=True,
    run_models=None,  # None=both, 'gp'=GP only, 'pfn'=PFN only
    phase2_retrain=False,  # If False: use best Phase 1 θ on full model, no training (compare covar complexity)
):
    if run_models == "pfn":
        num_runs = 0  # PFN doesn't use num_runs

    # Same as standard: total_train = num_folds * (train_size * dimensions); each fold has train_per_fold samples
    train_per_fold = train_size * dimensions
    total_train = num_folds * train_per_fold
    train_per_part = train_per_fold // CV_folds
    total_samples = num_test + total_train

    phase2_suffix = "" if phase2_retrain else "_phase2noTrain"
    if title is None:
        title = (
            f"GriewankCV_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]"
            f"_{num_folds}folds_CV{CV_folds}_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}{phase2_suffix}"
        )
    else:
        title = (
            f"GriewankCV_{title}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]"
            f"_{num_folds}folds_CV{CV_folds}_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}{phase2_suffix}"
        )

    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device)
    plot_save_path = f"{save_path}/plots" if save_path else None
    print(f" CV-GP: {num_folds} fold(s), each fold: {CV_folds} parts (num_runs={num_runs} per part) then full fold with {CV_folds} inits")
    print(f" Generating {total_samples} samples (train={total_train}, test={num_test}, {train_per_fold} per fold, {train_per_part} per CV part)")
    set_seed(seed)

    X_train_all, y_train_all, X_test_all, y_test_all = generate_griewank_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=list(x_bounds),
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    X_train_all = X_train_all.to(dtype=gp_dtype)
    X_test_all = X_test_all.to(dtype=gp_dtype)
    y_train_all = y_train_all.to(dtype=gp_dtype)
    y_test_all = y_test_all.to(dtype=gp_dtype)

    X = torch.cat([X_test_all, X_train_all], dim=0)
    qual_dict = learn_encodings(X)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)

    # Fold indices: same as standard (num_folds partitions of train data)
    torch.manual_seed(seed)
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)

    def _scale_fit_transform(X_tr, X_te, y_tr):
        X_tr = X_tr.detach().clone()
        X_te = X_te.detach().clone()
        y_tr = y_tr.detach().clone()
        if standardize_X:
            if x_standardize_method == 0:
                scaler_x = gpplus.utils.StandardScaler()
            elif x_standardize_method == 1:
                scaler_x = gpplus.utils.UniformScaler(scale_to_neg_one=False)
            else:
                scaler_x = gpplus.utils.UniformScaler(scale_to_neg_one=True)
            scaler_x.fit(X_tr[:, cont_cols])
            X_tr[:, cont_cols] = scaler_x.transform(X_tr[:, cont_cols])
            X_te[:, cont_cols] = scaler_x.transform(X_te[:, cont_cols])
        if standardize_y:
            scaler_y = gpplus.utils.StandardScaler()
            scaler_y.fit(y_tr)
            y_mean, y_std = scaler_y.mean, scaler_y.std
            y_tr_n = scaler_y.transform(y_tr)
        else:
            y_mean, y_std = None, None
            y_tr_n = y_tr
        return X_tr, X_te, y_tr_n, y_mean, y_std

    cholesky_jitter = getattr(defaults, "TRAINER_CHOLESKY_JITTER", 1e-6)
    GPPlus_metrics = []
    TabPFN_metrics = []
    GPTrainer_info = []
    gp_model_info = None
    tabpfn_model_info = None
    total_start_time = time.time()

    for fold_i in range(num_folds):
        fold_seed = (seed_trainer if seed_trainer is not None else seed) + fold_i
        print(f"\n{'='*60} FOLD {fold_i + 1}/{num_folds} (seed {fold_seed}) {'='*60}")
        fold_indices = train_indices_2d[fold_i]
        X_train_fold = X_train_all[fold_indices]
        y_train_fold = y_train_all[fold_indices]

        # Within this fold: split into CV_folds parts (deterministic per fold)
        torch.manual_seed(fold_seed)
        perm_fold = torch.randperm(train_per_fold)
        part_size = train_per_fold // CV_folds
        cv_init_params = []
        X_full_s = None
        y_mean_full = None
        y_std_full = None

        if run_models in [None, "gp"]:
            cv_part_times = []
            phase1_cv_parts = []  # Per-CV-part results for trainer analysis
            # ---------- Phase 1: Train one GP per CV part in series, each with num_runs ----------
            for k in range(CV_folds):
                start = k * part_size
                end = (k + 1) * part_size if k < CV_folds - 1 else train_per_fold
                part_idx = perm_fold[start:end]
                X_part = X_train_fold[part_idx]
                y_part = y_train_fold[part_idx]
                print(f"  CV part {k+1}/{CV_folds} (n={X_part.shape[0]}, num_runs={num_runs})")
                t_part_start = time.time()
                X_part_s, X_test_s, y_part_n, y_mean_k, y_std_k = _scale_fit_transform(X_part, X_test_all, y_part)
                model_part = gpplus.models.GPR(
                    X_part_s,
                    y_part_n,
                    kernel_module=defaults.SF_kernel,
                    mean_module=defaults.SF_mean,
                    likelihood=defaults.SF_likelihood,
                )
                gp_metric_part, _, _, info_k = train_eval_gp(
                    model_part,
                    X_test_s,
                    y_test_all,
                    num_epochs=num_epochs,
                    seed=fold_seed + 100 + k,
                    num_runs=num_runs,
                    lr=lr,
                    convergence_patience=convergence_patience,
                    min_loss_change=min_loss_change,
                    optimizer_class=optimizer_class,
                    initializer_class=initializer_class,
                    device=gp_device,
                    y_train_mean=y_mean_k,
                    y_train_std=y_std_k,
                    source_cols=source_cols,
                    trainer_info=True,
                    cholesky_jitter=cholesky_jitter,
                )
                elapsed_part = time.time() - t_part_start
                cv_part_times.append(elapsed_part)
                print(f"  CV part {k+1}/{CV_folds} time: {elapsed_part:.2f}s")
                theta_k = _get_best_final_dict(info_k)
                if theta_k is None:
                    raise RuntimeError(f"Fold {fold_i+1} CV part {k+1}: could not get final parameters.")
                cv_init_params.append(theta_k)
                _print_cv_hyperparameters(f"θ{k+1} (fold {fold_i+1} part {k+1}/{CV_folds})", theta_k)
                # Print Phase 1 metrics for this CV part (how well it did before Phase 2)
                _print_metrics_block(
                    f"CV part {k+1}/{CV_folds} results (before Phase 2)",
                    gp_metric_part,
                )
                # Store Phase 1 result for trainer analysis JSON
                phase1_cv_parts.append({
                    "part": k + 1,
                    "n_train": int(X_part.shape[0]),
                    "time_s": elapsed_part,
                    "metrics": gp_metric_part,
                    "final_parameters": theta_k,
                    "trainer_info": info_k,  # runs, best_parameters, etc.
                })

            # ---------- Phase 2: full fold ----------
            X_full_s, X_test_full_s, y_full_n, y_mean_full, y_std_full = _scale_fit_transform(
                X_train_fold, X_test_all, y_train_fold
            )
            best_part_idx, best_theta = None, None
            if not phase2_retrain:
                best_part_idx, best_theta = _best_phase1_part_by_loss(phase1_cv_parts)
            # Full model always uses full training set → covariance is n_full × n_full
            print(f"\n--- {title} Fold {fold_i + 1}/{num_folds} GP {'Training' if phase2_retrain else '(no train: best Phase 1 θ)'} ---")
            print(f"X_train: {X_full_s.shape}  (n_full={X_full_s.shape[0]} → covar {X_full_s.shape[0]}×{X_full_s.shape[0]})")
            print(f"X_test: {X_test_full_s.shape}")
            print(f"y_test mean: {y_test_all.mean().item()} / y_test std: {y_test_all.std().item()}")
            model_full = gpplus.models.GPR(
                X_full_s,
                y_full_n,
                kernel_module=defaults.SF_kernel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )

            if phase2_retrain:
                # Phase 2: CV_folds separate full-data models, each with 1 init from one CV part; train each, then evaluate all
                print(f"  Full fold: {CV_folds} separate models (1 init each from θ1..θ{CV_folds}), train then evaluate all")
                phase2_metrics_list = []
                for k in range(CV_folds):
                    model_full_k = gpplus.models.GPR(
                        X_full_s,
                        y_full_n,
                        kernel_module=defaults.SF_kernel,
                        mean_module=defaults.SF_mean,
                        likelihood=defaults.SF_likelihood,
                    )
                    print(f"  Phase 2 model {k+1}/{CV_folds} (init from CV part {k+1})")
                    t_k = time.time()
                    gp_metric_k, _, _, gp_trainer_info_k = train_eval_gp(
                        model_full_k,
                        X_test_full_s,
                        y_test_all,
                        num_epochs=num_epochs,
                        seed=fold_seed + 200 + k,
                        num_runs=1,
                        lr=lr,
                        convergence_patience=convergence_patience,
                        min_epochs=min_epochs,
                        min_loss_change=min_loss_change,
                        optimizer_class=optimizer_class,
                        initializer_class=FixedParameterInitializer,
                        initializer_kwargs={"parameter_dicts": [cv_init_params[k]]},
                        device=gp_device,
                        y_train_mean=y_mean_full,
                        y_train_std=y_std_full,
                        source_cols=source_cols,
                        trainer_info=trainer_info,
                        cholesky_jitter=cholesky_jitter,
                    )
                    elapsed_k = time.time() - t_k
                    gp_metric_k["fold"] = fold_i + 1
                    gp_metric_k["phase2_init_part"] = k + 1
                    gp_metric_k["phase2_retrain"] = True
                    gp_metric_k["cv_full_fold_time_s"] = elapsed_k
                    for j, theta in enumerate(cv_init_params):
                        prefix = f"cv_part_{j+1}_"
                        gp_metric_k[prefix + "raw_noise"] = theta.get("raw_noise")
                        gp_metric_k[prefix + "raw_outputscale"] = theta.get("raw_outputscale")
                        gp_metric_k[prefix + "raw_lengthscales"] = theta.get("raw_lengthscales")
                        if j < len(cv_part_times):
                            gp_metric_k[prefix + "time_s"] = cv_part_times[j]
                    phase2_metrics_list.append(gp_metric_k)
                    _print_metrics_block(f"  Phase 2 model {k+1}/{CV_folds} (init part {k+1})", gp_metric_k)
                    GPPlus_metrics.append(gp_metric_k)
                    if gp_trainer_info_k:
                        gp_trainer_info_k["fold"] = fold_i + 1
                        gp_trainer_info_k["phase2_init_part"] = k + 1
                        gp_trainer_info_k["metrics"] = gp_metric_k
                        gp_trainer_info_k["phase1_cv_parts"] = phase1_cv_parts
                        gp_trainer_info_k["phase2_intended_inits"] = cv_init_params
                        phase2_final_k = _phase2_final_from_metrics(gp_metric_k)
                        phase2_final_k["chosen_from_part"] = k + 1
                        gp_trainer_info_k["phase2_final"] = phase2_final_k
                        gp_trainer_info_k["final_chosen_hyperparameters"] = phase2_final_k
                        GPTrainer_info.append(gp_trainer_info_k)
                    rrmse_k = gp_metric_k.get("RRMSE")
                    nis_k = gp_metric_k.get("NIS")
                    print(f"  Phase 2 model {k+1}/{CV_folds}: RRMSE={rrmse_k:.4f}, NIS={nis_k:.4f}, time={elapsed_k:.2f}s")
                # Per-fold summary of all 4 models
                rrmses = [m.get("RRMSE") for m in phase2_metrics_list if m.get("RRMSE") is not None]
                nises = [m.get("NIS") for m in phase2_metrics_list if m.get("NIS") is not None]
                if rrmses:
                    print(f"  Fold {fold_i+1} Phase 2 all models: RRMSE min={min(rrmses):.4f} max={max(rrmses):.4f} mean={sum(rrmses)/len(rrmses):.4f}")
                if nises:
                    print(f"  Fold {fold_i+1} Phase 2 all models: NIS   min={min(nises):.4f} max={max(nises):.4f} mean={sum(nises)/len(nises):.4f}")
                gp_metric = phase2_metrics_list[0]
                gp_trainer_info = None
                model_full = model_full_k
                elapsed_full_fold = sum(m.get("cv_full_fold_time_s", 0) for m in phase2_metrics_list)
            else:
                # No train: use best Phase 1 θ on full model, evaluate only (compare covar complexity)
                print(f"  Full fold: phase2_retrain=False → using best Phase 1 part {best_part_idx + 1} (by -MLL), no training")
                set_hyperparameters_from_final_dict(model_full, best_theta)
                print(model_full)
                t_pred = time.time()
                y_pred, _, _, output_std = evaluate_gp_model(model_full, X_test_full_s)
                prediction_time = time.time() - t_pred
                # Denormalize
                if y_mean_full is not None and y_std_full is not None:
                    y_pred = (y_pred * y_std_full) + y_mean_full
                    output_std = output_std * y_std_full
                y_test_np = y_test_all.detach().cpu().numpy().reshape(-1)
                y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
                output_std_np = output_std.detach().cpu().numpy().reshape(-1)
                gp_metric = compute_metrics(
                    y_test_np, y_pred_np, output_std=output_std_np,
                    training_time=0.0, prediction_time=prediction_time,
                )
                gp_metric["phase2_retrain"] = False
                gp_metric["phase2_best_part"] = best_part_idx + 1
                # Final chosen = best Phase 1 part's θ (only set shown in print)
                gp_metric["chosen_raw_noise"] = best_theta.get("raw_noise")
                gp_metric["chosen_raw_outputscale"] = best_theta.get("raw_outputscale")
                gp_metric["chosen_noise"] = best_theta.get("noise")
                gp_metric["chosen_outputscale"] = best_theta.get("outputscale")
                gp_metric["chosen_raw_lengthscales"] = best_theta.get("raw_lengthscales")
                gp_metric["y_train_mean"] = float(y_mean_full.item()) if hasattr(y_mean_full, "item") else float(y_mean_full)
                gp_metric["y_train_std"] = float(y_std_full.item()) if hasattr(y_std_full, "item") else float(y_std_full)
                for k, theta in enumerate(cv_init_params):
                    prefix = f"cv_part_{k+1}_"
                    gp_metric[prefix + "raw_noise"] = theta.get("raw_noise")
                    gp_metric[prefix + "raw_outputscale"] = theta.get("raw_outputscale")
                    gp_metric[prefix + "raw_lengthscales"] = theta.get("raw_lengthscales")
                    if k < len(cv_part_times):
                        gp_metric[prefix + "time_s"] = cv_part_times[k]
                gp_metric["cv_full_fold_time_s"] = prediction_time
                # Minimal trainer_info so we still save phase1_cv_parts and phase2_final to trainer analysis JSON
                gp_trainer_info = {
                    "phase2_retrain": False,
                    "best_parameters": {"final_parameters": best_theta, "loss": None},
                    "runs": [],
                }
                elapsed_full_fold = prediction_time
            if not phase2_retrain:
                _print_metrics_block(f"GP Results (Fold {fold_i+1}/{num_folds})", gp_metric)
                GPPlus_metrics.append(gp_metric)
                if gp_trainer_info:
                    gp_trainer_info["fold"] = fold_i + 1
                    gp_trainer_info["metrics"] = gp_metric
                    gp_trainer_info["phase1_cv_parts"] = phase1_cv_parts
                    gp_trainer_info["phase2_intended_inits"] = cv_init_params
                    chosen = {
                        "chosen_from_part": int(gp_metric["phase2_best_part"]),
                        "raw_noise": gp_metric.get("chosen_raw_noise"),
                        "raw_outputscale": gp_metric.get("chosen_raw_outputscale"),
                        "raw_lengthscales": gp_metric.get("chosen_raw_lengthscales"),
                        "noise": gp_metric.get("chosen_noise"),
                        "outputscale": gp_metric.get("chosen_outputscale"),
                        "RRMSE": gp_metric.get("RRMSE"),
                    }
                    gp_trainer_info["final_chosen_hyperparameters"] = chosen
                    gp_trainer_info["phase2_final"] = chosen
                    GPTrainer_info.append(gp_trainer_info)
            rrmse = gp_metric.get("RRMSE")
            if phase2_retrain:
                rrmses_fold = [m.get("RRMSE") for m in phase2_metrics_list if m.get("RRMSE") is not None]
                print(f"  CV-GP Fold {fold_i+1}: 4 Phase 2 models, RRMSE range [{min(rrmses_fold):.4f}, {max(rrmses_fold):.4f}]" if rrmses_fold else f"  CV-GP Fold {fold_i+1}: done")
            else:
                print(f"  CV-GP Fold {fold_i+1}: RRMSE={rrmse:.4f}" if rrmse is not None else f"  CV-GP Fold {fold_i+1}: done")

            if fold_i == 0:
                y_test_stats = {
                    "y_test_mean": float(y_test_all.mean().item()),
                    "y_test_std": float(y_test_all.std().item()),
                }
                X_scaling_type = (
                    "UniformScaler [-1, 1]"
                    if x_standardize_method == 2
                    else ("UniformScaler [0, 1]" if x_standardize_method == 1 else "StandardScaler (Gaussian)")
                )
                gp_model_info = {
                    "model_str": str(model_full),
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "source_cols": source_cols,
                    "qual_dict": qual_dict,
                    "input_dim": X_full_s.shape[1],
                    "train_samples": int(X_full_s.shape[0]),
                    "test_samples": num_test,
                    "y_train_mean": float(y_mean_full.item()),
                    "y_train_std": float(y_std_full.item()),
                    "standardize_X": standardize_X,
                    "standardize_y": standardize_y,
                    "X_scaling_type": X_scaling_type,
                    "x_standardize_method": x_standardize_method,
                    "dtype": str(gp_dtype),
                    "device": str(gp_device),
                    "num_epochs": num_epochs,
                    "num_runs": CV_folds,
                    "num_runs_per_cv_part": num_runs,
                    "num_folds": num_folds,
                    "CV_folds": CV_folds,
                    "train_per_fold": train_per_fold,
                    "train_per_part": train_per_part,
                    "lr": lr,
                    "optimizer": optimizer_class.__name__,
                    "convergence_patience": convergence_patience,
                    "initializer": "FixedParameterInitializer (CV)",
                    "phase2_retrain": phase2_retrain,
                    "x_bounds": list(x_bounds),
                    **y_test_stats,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                }

        # ---------- TabPFN on this fold's full data ----------
        if run_models in [None, "pfn"]:
            if run_models == "pfn" or X_full_s is None:
                X_full_s, X_test_full_s, y_full_n, y_mean_full, y_std_full = _scale_fit_transform(
                    X_train_fold, X_test_all, y_train_fold
                )
            print(f"  TabPFN (fold {fold_i+1}, n_train={X_train_fold.shape[0]})")
            tabpfn_metric, _, _ = train_eval_PFN(
                X_full_s,
                X_test_full_s,
                y_full_n,
                y_test_all,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                source_cols=source_cols,
                y_train_mean=y_mean_full,
                y_train_std=y_std_full,
            )
            TabPFN_metrics.append(tabpfn_metric)
            rrmse_pfn = tabpfn_metric.get("RRMSE")
            print(f"  TabPFN Fold {fold_i+1}: RRMSE={rrmse_pfn:.4f}" if rrmse_pfn is not None else f"  TabPFN Fold {fold_i+1}: done")
            if fold_i == 0:
                tabpfn_model_info = {
                    "model_path": regressor.model_path,
                    "fit_mode": regressor.fit_mode,
                    "device": str(regressor.device),
                    "inference_precision": regressor.inference_precision,
                    "random_state": regressor.random_state,
                    "use_autocast": getattr(regressor, "use_autocast_", None),
                    "forced_inference_dtype": str(regressor.forced_inference_dtype_) if getattr(regressor, "forced_inference_dtype_", None) is not None else None,
                }

    # ---------- Summary and save ----------
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {num_folds} fold(s), {len(GPPlus_metrics)} CV-GP metric(s), {len(TabPFN_metrics)} TabPFN metric(s)")
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="CV-GP", title=title) if GPPlus_metrics else None
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if TabPFN_metrics else None

    if save_path:
        save_path_p = Path(save_path)
        save_path_p.mkdir(parents=True, exist_ok=True)
        if run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "CV-GP"], title=title, save_path=plot_save_path)
        try:
            file_prefix = "gpVpfn" if run_models is None else (run_models if run_models else "gpVpfn")
            combined_data = {}
            if run_models in [None, "gp"] and gp_model_info is not None:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                }
            if run_models in [None, "pfn"] and tabpfn_model_info is not None:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info,
                }
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
            (save_path_p / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
            print(f"\nSaved to {save_path_p / f'{file_prefix}_{title}.json'}")
        except Exception:
            pass
        if trainer_info and GPTrainer_info and run_models in [None, "gp"]:
            try:
                trainer_analysis_dir = save_path_p / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
                trainer_info_data = {
                    "title": title,
                    "CV_folds": CV_folds,
                    "num_runs_full_data": CV_folds,
                    "num_runs_per_cv_part": num_runs,
                    "trainer_info": GPTrainer_info,
                }
                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                try:
                    from plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data
                    plot_trainer_analysis_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except Exception as plot_e:
                    print(f"Trainer analysis plotting skipped: {plot_e}")
            except Exception as e:
                print(f"Error saving trainer info: {e}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    griewankCV_GPvsPFN(
        num_folds=1,
        train_size=80,
        dimensions=20,
        CV_folds=2,
        num_runs=defaults.TRAINER_NUM_RUNS,
        save_path="./results/griewank",
        noise_train=0.005,
        noise_test=0.005,
        run_models=None,
        phase2_retrain=True,  # False: use best Phase 1 θ on full model, no training (compare covar complexity)
    )
