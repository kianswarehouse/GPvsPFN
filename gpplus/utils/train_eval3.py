import time

import numpy as np
import torch
from gpplus.training import GPTrainerV3 as GPTrainer
from gpplus.training.callbacks import (
    FinalParameterStorageCallback,
    IterationParameterCallback,
    EpochParameterCallback,
    JitterTrackingCallback,
    LBFGSInnerMetricsCallbackV3,
    PrintTrainingMetricsCallback,
)
from gpplus.training.stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition
from gpplus.training.eval2 import evaluate_gp_model
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils.metrics_functions import compute_metrics, compute_per_source_metrics


def _process_trainer_info(stored_params, train_results, num_epochs):
    """
    Process stored parameters from callback into a structured trainer log.
    Delegates to the shared trainer_analysis module.
    """
    from gpplus.training.trainer_analysis import build_trainer_analysis_payload
    return build_trainer_analysis_payload(stored_params, train_results, num_epochs)


def train_eval_gp(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    seed: int,
    num_inits: int,
    lr: float,
    convergence_patience: int = 10,
    min_epochs: int = 0,
    min_loss_change: float = 1e-7,
    optimizer_class=None,
    optimizer_kwargs: dict | None = None,
    initializer_class=None,
    initializer_kwargs: dict | None = None,
    device: str = "cpu",
    # dtype: torch.dtype = torch.float64,
    y_train_mean: torch.Tensor | dict | None = None,
    y_train_std: torch.Tensor | dict | None = None,
    standardize_y_log_scale: bool = False,
    log_scale_C: float | None = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    source_cols: int | list[int] | None = None,
    trainer_info: bool = False,
    cholesky_jitter: float = 1e-6,  # Jitter for Cholesky; use larger (e.g. 1e-5, 1e-4) for large n
    fold_index: int | None = None,  # Fold index for multi-fold experiments (sets fold_index on callbacks)
    callbacks: list | None = None,  # Optional list of callbacks (if None, creates default callbacks)
    callback_save_path: str | None = None,  # Base path for saving callback data (if None, uses default paths)
    print_epoch_metrics: bool = False,  # If True, print NLL/NIS/LOO_NLL/KF at each epoch
    log_lbfgs_inner: bool = True,  # Enabled by default: log/store per-iteration loss inside LBFGSScipy step(); metrics via callbacks
    lbfgs_inner_extra_metrics: list | None = None,  # Optional [(name, fn(context)->float), ...] for LBFGSInnerMetricsCallbackV3
):
    """
    Train a GP model and evaluate metrics on the provided test set.

    This v3 variant:
    - Uses GPTrainerV3 (v2 trainer core + original aggregation of callbacks).
    - Supports advanced optimization losses and per-iteration L-BFGS metrics.
    - Preserves v1 callback/aggregation behaviour (parameter/jitter tracking).
    """
    # Set optimizer kwargs based on optimizer type (Adam convenience)
    if optimizer_class is torch.optim.Adam or (
        isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.Adam)
    ):
        optimizer_kwargs = {"lr": lr if lr is not None else 0.01}

    # Use provided callbacks or create default ones
    if callbacks is None:
        callbacks = [FinalParameterStorageCallback(save_file=None, verbose=False)]

        # Add optimizer-specific parameter callbacks
        if optimizer_class is not None:
            # save_file: only set when callback_save_path is set → callbacks write JSON; None → no callback file I/O
            if callback_save_path is not None:
                epoch_save_file = f"{callback_save_path}/epoch_parameters.json"
                jitter_save_file = f"{callback_save_path}/jitter_tracking.json"
            else:
                epoch_save_file = None
                jitter_save_file = None

            # LBFGSScipy: log per-iteration parameters and (optionally) inner metrics via callbacks
            if optimizer_class is LBFGSScipy or (
                isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy)
            ):
                callbacks.append(
                    IterationParameterCallback(
                        save_file=None,
                        verbose=False,
                        save_every_n_iterations=20,
                    )
                )
                if log_lbfgs_inner:
                    callbacks.append(
                        LBFGSInnerMetricsCallbackV3(
                            log_record_every_n_iters=1,
                            log_metrics_every_n_iters=1,
                            log_nll=True,
                            log_nis=True,
                            log_loo=True,
                            log_kf=True,
                            log_residual_mse=True,
                            extra_metrics=lbfgs_inner_extra_metrics or [],
                        )
                    )
            # Adam: log per-epoch parameters
            elif optimizer_class is torch.optim.Adam or (
                isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.Adam)
            ):
                callbacks.append(
                    EpochParameterCallback(
                        save_file=epoch_save_file,
                        verbose=False,
                        save_every_n_epochs=20,
                    )
                )

            # # Jitter tracking across runs/epochs
            # callbacks.append(
            #     JitterTrackingCallback(
            #         save_file=jitter_save_file,
            #         verbose=True,
            #     )
            # )

    # Optional epoch-metric printing callback
    if print_epoch_metrics:
        callbacks.append(PrintTrainingMetricsCallback())

    # Set fold_index on callbacks that support it
    if fold_index is not None:
        for cb in callbacks:
            if hasattr(cb, "set_fold_index"):
                cb.set_fold_index(fold_index)

    trainer = GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_inits=num_inits,
        stop_conditions=[
            ConvergencePatienceStopCondition(patience=convergence_patience),
            MinLossChangeStopCondition(min_loss_change=min_loss_change),
        ],
        min_epochs=min_epochs,
        callbacks=callbacks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        initializer_class=initializer_class,
        initializer_kwargs=initializer_kwargs,
        cholesky_jitter=cholesky_jitter,
    )

    # Training (use trainer's full_train_time for training_time when available)
    train_results = trainer.train()
    training_time = float(getattr(trainer, "full_train_time", 0.0) or 0.0)

    # Measure prediction time
    t_pred_start = time.time()
    y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
    prediction_time = time.time() - t_pred_start

    # Check for NaN/Inf in model outputs before denormalization
    if torch.any(~torch.isfinite(y_pred)):
        nan_count = torch.sum(~torch.isfinite(y_pred)).item()
        import warnings

        warnings.warn(
            f"train_eval_gp: Model predictions contain {nan_count} NaN/Inf values before denormalization. "
            "This may indicate model training issues. Replacing with mean prediction (0 in standardized space)."
        )
        y_pred = torch.where(torch.isfinite(y_pred), y_pred, torch.zeros_like(y_pred))

    if torch.any(~torch.isfinite(output_std)):
        nan_count = torch.sum(~torch.isfinite(output_std)).item()
        import warnings

        warnings.warn(
            f"train_eval_gp: Model output_std contains {nan_count} NaN/Inf values before denormalization. "
            "This may indicate model training issues."
        )
        output_std = torch.where(torch.isfinite(output_std), output_std, torch.ones_like(output_std) * 1e-6)

    # Denormalization logic (unchanged from v1/v2)
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            if source_cols is not None:
                is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
                if is_onehot:
                    onehot_cols = X_test[:, source_cols]
                    source_indices_test = torch.argmax(onehot_cols, dim=1)
                else:
                    source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
                    source_indices_test = X_test[:, source_col].long()

                y_pred_denorm = torch.zeros_like(y_pred)
                output_std_denorm = torch.zeros_like(output_std)
                unique_sources = torch.unique(source_indices_test)

                for source_idx in unique_sources:
                    source_mask = source_indices_test == source_idx
                    source_key = source_idx.item()

                    if source_key in y_train_mean:
                        mean = y_train_mean[source_key]
                        std = y_train_std[source_key]
                    elif 0 in y_train_mean:
                        mean = y_train_mean[0]
                        std = y_train_std[0]
                    else:
                        first_key = list(y_train_mean.keys())[0]
                        mean = y_train_mean[first_key]
                        std = y_train_std[first_key]

                    if isinstance(mean, torch.Tensor):
                        mean = mean.squeeze()
                    if isinstance(std, torch.Tensor):
                        std = std.squeeze()

                    if standardize_y_log_scale:
                        log_y_pred = (y_pred[source_mask] * std) + mean
                        log_y_std = output_std[source_mask] * std
                        max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                        log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                        exp_log_y = torch.exp(log_y_pred)
                        y_pred_denorm[source_mask] = exp_log_y - log_scale_C
                        output_std_denorm[source_mask] = exp_log_y * log_y_std
                    else:
                        y_pred_denorm[source_mask] = (y_pred[source_mask] * std) + mean
                        output_std_denorm[source_mask] = output_std[source_mask] * std

                y_pred = y_pred_denorm
                output_std = output_std_denorm
            else:
                first_key = list(y_train_mean.keys())[0]
                mean = y_train_mean[first_key]
                std = y_train_std[first_key]

                if isinstance(mean, torch.Tensor):
                    mean = mean.squeeze()
                if isinstance(std, torch.Tensor):
                    std = std.squeeze()
                if standardize_y_log_scale:
                    log_y_pred = (y_pred * std) + mean
                    log_y_std = output_std * std
                    max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                    log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                    exp_log_y = torch.exp(log_y_pred)
                    y_pred = exp_log_y - log_scale_C
                    output_std = exp_log_y * log_y_std
                else:
                    y_pred = (y_pred * std) + mean
                    output_std = output_std * std
        else:
            mean_val = y_train_mean.squeeze() if isinstance(y_train_mean, torch.Tensor) else y_train_mean
            std_val = y_train_std.squeeze() if isinstance(y_train_std, torch.Tensor) else y_train_std

            if standardize_y_log_scale:
                log_y_pred = (y_pred * std_val) + mean_val
                log_y_std = output_std * std_val
                max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                exp_log_y = torch.exp(log_y_pred)
                y_pred = exp_log_y - log_scale_C
                output_std = exp_log_y * log_y_std
            else:
                y_pred = (y_pred * std_val) + mean_val
                output_std = output_std * std_val

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)

    # Check for NaN/Inf values and replace with reasonable defaults
    if np.any(~np.isfinite(y_pred_np)):
        nan_mask = ~np.isfinite(y_pred_np)
        if standardize_y_log_scale:
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            if len(valid_preds) > 0:
                replacement = np.median(valid_preds)
            else:
                if y_train_mean is not None:
                    if isinstance(y_train_mean, dict):
                        log_mean_val = list(y_train_mean.values())[0]
                    else:
                        log_mean_val = y_train_mean
                    if isinstance(log_mean_val, torch.Tensor):
                        log_mean_val = (
                            log_mean_val.item()
                            if log_mean_val.numel() == 1
                            else log_mean_val.cpu().numpy()
                        )
                    replacement = np.exp(float(log_mean_val)) - log_scale_C
                else:
                    replacement = 0.0
            y_pred_np[nan_mask] = replacement
        else:
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            replacement = np.median(valid_preds) if len(valid_preds) > 0 else 0.0
            y_pred_np[nan_mask] = replacement

    if np.any(~np.isfinite(output_std_np)):
        nan_mask = ~np.isfinite(output_std_np)
        valid_std = output_std_np[np.isfinite(output_std_np)]
        replacement = np.median(valid_std) if len(valid_std) > 0 else 1.0
        output_std_np[nan_mask] = replacement

    gp_metric = compute_metrics(
        y_test,
        y_pred_np,
        output_std_np,
        training_time=training_time,
        prediction_time=prediction_time,
    )
    # When trainer provides timing breakdown: report exactly Total_Time, Full_Train_Time, Train_Time, Log_Time, Prediction_Time at top (no duplicates)
    if (
        hasattr(trainer, "full_train_time")
        and trainer.full_train_time is not None
        and hasattr(trainer, "train_time")
        and trainer.train_time is not None
        and hasattr(trainer, "log_time")
        and trainer.log_time is not None
    ):
        _time_keys = ("Total_Time", "Full_Train_Time", "Train_Time", "Log_Time", "Prediction_Time")
        _skip = ("Total_Time", "Training_Time", "Prediction_Time")
        new_metric = {
            "Total_Time": gp_metric["Total_Time"],
            "Full_Train_Time": float(trainer.full_train_time),
            "Train_Time": float(trainer.train_time),
            "Log_Time": float(trainer.log_time),
            "Prediction_Time": gp_metric["Prediction_Time"],
        }
        for k, v in gp_metric.items():
            if k not in _skip:
                new_metric[k] = v
        gp_metric = new_metric

    # Extract noise and noise_std (will be added in correct order later)
    noise_std = None
    noise_std_original_scale = None
    try:
        noise_variance = model.likelihood.noise.detach().cpu()
        noise_std = np.sqrt(noise_variance)

        if y_train_std is not None:
            if isinstance(y_train_std, dict):
                if 0 in y_train_std:
                    std_to_use = y_train_std[0]
                    mean_to_use = (
                        y_train_mean[0]
                        if isinstance(y_train_mean, dict) and 0 in y_train_mean
                        else None
                    )
                else:
                    std_to_use = list(y_train_std.values())[0]
                    mean_to_use = (
                        list(y_train_mean.values())[0]
                        if isinstance(y_train_mean, dict)
                        else None
                    )
            else:
                std_to_use = y_train_std
                mean_to_use = y_train_mean

            if standardize_y_log_scale:
                if mean_to_use is not None:
                    if isinstance(mean_to_use, torch.Tensor):
                        log_mean = (
                            mean_to_use.item()
                            if mean_to_use.numel() == 1
                            else mean_to_use.cpu().numpy()
                        )
                    else:
                        log_mean = float(mean_to_use) if np.isscalar(mean_to_use) else mean_to_use

                    if isinstance(std_to_use, torch.Tensor):
                        log_std = (
                            std_to_use.item()
                            if std_to_use.numel() == 1
                            else std_to_use.cpu().numpy()
                        )
                    else:
                        log_std = float(std_to_use) if np.isscalar(std_to_use) else std_to_use

                    log_noise_std = noise_std * log_std
                    noise_std_original_scale = np.exp(log_mean) * log_noise_std
                else:
                    if isinstance(std_to_use, torch.Tensor):
                        std_val = (
                            std_to_use.item()
                            if std_to_use.numel() == 1
                            else std_to_use.cpu().numpy()
                        )
                    else:
                        std_val = std_to_use
                    noise_std_original_scale = noise_std * std_val
            else:
                if isinstance(std_to_use, torch.Tensor):
                    std_val = (
                        std_to_use.item()
                        if std_to_use.numel() == 1
                        else std_to_use.cpu().numpy()
                    )
                else:
                    std_val = std_to_use
                noise_std_original_scale = noise_std * std_val
        else:
            noise_std_original_scale = noise_std
    except Exception as e:
        import logging

        logging.warning(f"Could not extract noise std: {e}")

    # Always extract directly from the model after training (best model already loaded)
    best_model_metrics = None

    best_run = (
        min(
            [r for r in train_results if r.get("loss") is not None],
            key=lambda x: x.get("loss", float("inf")),
            default=None,
        )
        if train_results
        else None
    )

    num_epochs_actual = trainer.num_epochs if hasattr(trainer, "num_epochs") else None
    best_epoch_value = None
    jitter_value = None
    jitter_max_value = None

    if best_run is not None:
        callback_data = best_run.get("callback_data", {}) or {}
        for cb_key, stored_params_list in callback_data.items():
            if "FinalParameterStorage" in cb_key or "ParameterStorage" in cb_key:
                if stored_params_list:
                    import math

                    def _loss_or_inf(rec):
                        val = rec.get("best_loss")
                        return float(val) if val is not None else math.inf

                    record = min(stored_params_list, key=_loss_or_inf)
                    best_epoch_value = record.get("best_epoch", best_epoch_value)
                    jitter_value = record.get("jitter", jitter_value)
                    jitter_max_value = record.get("jitter_max", jitter_max_value)
                    if num_epochs_actual is None:
                        num_epochs_actual = record.get("num_epochs", num_epochs_actual)
                break

    if hasattr(trainer, "cholesky_jitter"):
        temp_callback = FinalParameterStorageCallback(verbose=False)
        extracted_params = temp_callback._extract_final_parameters(
            model,
            epoch=0,
            best_loss=best_run.get("loss") if best_run else None,
            cholesky_jitter=trainer.cholesky_jitter,
            best_epoch=None,
            jitter_max=None,
        )
        if extracted_params:
            lengthscales_extracted = extracted_params.get("lengthscales")
            cat_lengthscales_extracted = extracted_params.get("cat_lengthscales")
            source_lengthscales_extracted = extracted_params.get("source_lengthscales")
            best_model_metrics = {
                "num_epochs": num_epochs_actual
                if num_epochs_actual is not None
                else extracted_params.get("num_epochs"),
                "best_epoch": best_epoch_value,
                "best_loss": best_run.get("loss") if best_run else extracted_params.get("best_loss"),
                "jitter": jitter_value if jitter_value is not None else extracted_params.get("jitter"),
                "jitter_max": jitter_max_value,
                "raw_noise": extracted_params.get("raw_noise"),
                "outputscale": extracted_params.get("outputscale"),
                "raw_power": extracted_params.get("raw_power"),
                "power": extracted_params.get("power"),
                "lengthscales": lengthscales_extracted,
                "cat_lengthscales": cat_lengthscales_extracted,
                "source_lengthscales": source_lengthscales_extracted,
            }

    if best_model_metrics:
        gp_metric["jitter"] = best_model_metrics.get("jitter")
        if best_model_metrics.get("jitter_max") is not None:
            gp_metric["jitter_max"] = best_model_metrics.get("jitter_max")

        def add_metric(name, value):
            if value is None:
                return
            if hasattr(value, "numpy"):
                value = value.numpy()
            if hasattr(value, "size") and value.size == 1:
                gp_metric[name] = float(value.item() if hasattr(value, "item") else value.flat[0])
            elif hasattr(value, "__len__") and len(value) > 1:
                for i, v in enumerate(value):
                    gp_metric[f"{name}_{i}"] = float(v)
            else:
                gp_metric[name] = float(value)

        try:
            raw_noise = model.likelihood.raw_noise.detach().cpu()
            add_metric(
                "raw_noise",
                raw_noise.numpy().flatten() if raw_noise.numel() > 1 else raw_noise.item(),
            )
        except Exception:
            add_metric("raw_noise", best_model_metrics.get("raw_noise"))

        add_metric("noise", noise_std)
        add_metric("noise_std", noise_std_original_scale)
        gp_metric["outputscale"] = best_model_metrics.get("outputscale")

        raw_power = best_model_metrics.get("raw_power")
        power = best_model_metrics.get("power")
        if raw_power is not None:
            gp_metric["raw_power"] = (
                float(raw_power)
                if isinstance(raw_power, (int, float))
                else float(raw_power.item() if hasattr(raw_power, "item") else raw_power)
            )
        if power is not None:
            gp_metric["power"] = (
                float(power)
                if isinstance(power, (int, float))
                else float(power.item() if hasattr(power, "item") else power)
            )

        gp_metric.update(
            {
                "num_epochs": best_model_metrics.get("num_epochs"),
                "best_epoch": best_model_metrics.get("best_epoch"),
            }
        )

        lengthscales = best_model_metrics.get("lengthscales")
        if lengthscales is None:
            pass
        elif isinstance(lengthscales, (list, tuple)):
            if len(lengthscales) > 0:
                for i, ls_val in enumerate(lengthscales):
                    gp_metric[f"cont_lengthscale_{i}"] = ls_val
        else:
            gp_metric["cont_lengthscale_0"] = lengthscales

        cat_lengthscales = best_model_metrics.get("cat_lengthscales")
        if cat_lengthscales is not None:
            if isinstance(cat_lengthscales, (list, tuple)):
                if len(cat_lengthscales) > 0:
                    for i, ls_val in enumerate(cat_lengthscales):
                        gp_metric[f"cat_lengthscale_{i}"] = ls_val
            else:
                gp_metric["cat_lengthscale_0"] = cat_lengthscales

        source_lengthscales = best_model_metrics.get("source_lengthscales")
        if source_lengthscales is not None:
            if isinstance(source_lengthscales, (list, tuple)):
                if len(source_lengthscales) > 0:
                    for i, ls_val in enumerate(source_lengthscales):
                        gp_metric[f"source_lengthscale_{i}"] = ls_val
            else:
                gp_metric["source_lengthscale_0"] = source_lengthscales
    else:
        pass

        def add_metric(name, value):
            if value is None:
                return
            if hasattr(value, "size"):
                if value.size == 1:
                    gp_metric[name] = float(value.item() if hasattr(value, "item") else value)
                else:
                    for i, v in enumerate(value):
                        gp_metric[f"{name}_{i}"] = float(v)
            elif isinstance(value, (list, tuple)) and len(value) > 1:
                for i, v in enumerate(value):
                    gp_metric[f"{name}_{i}"] = float(v)
            else:
                gp_metric[name] = float(value)

        add_metric("noise", noise_std)
        add_metric("noise_std", noise_std_original_scale)

    # Per-source metrics
    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]

        gp_per_source_metric = compute_per_source_metrics(
            y_test,
            y_pred_np,
            output_std_np,
            X_test,
            source_columns=source_cols,
            training_time=training_time,
            prediction_time=prediction_time,
        )

        for source_name, source_metrics in gp_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                gp_metric[f"{source_name}_{metric_name}"] = metric_value

    # Store y_train_mean/std used for this run
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            for source_key, mean_val in y_train_mean.items():
                gp_metric[f"y_train_mean_source_{source_key}"] = float(
                    mean_val.item() if hasattr(mean_val, "item") else mean_val
                )
            for source_key, std_val in y_train_std.items():
                gp_metric[f"y_train_std_source_{source_key}"] = float(
                    std_val.item() if hasattr(std_val, "item") else std_val
                )
        else:
            if hasattr(y_train_mean, "item"):
                mean_val = (
                    y_train_mean.item()
                    if y_train_mean.numel() == 1
                    else y_train_mean.squeeze().item()
                )
            else:
                mean_val = y_train_mean
            if hasattr(y_train_std, "item"):
                std_val = (
                    y_train_std.item()
                    if y_train_std.numel() == 1
                    else y_train_std.squeeze().item()
                )
            else:
                std_val = y_train_std
            gp_metric["y_train_mean"] = float(mean_val)
            gp_metric["y_train_std"] = float(std_val)

    # Trainer info structure (includes optional lbfgs_inner_metrics)
    gp_trainer_info = None
    if trainer_info:
        from gpplus.training.trainer_analysis import build_stored_params_from_results, build_trainer_analysis_payload
        all_stored_params = build_stored_params_from_results(train_results)

        if all_stored_params:
            gp_trainer_info = build_trainer_analysis_payload(all_stored_params, train_results, num_epochs)
        else:
            # Fallback when no callback data: extract from model and best run
            best_run = (
                min(
                    [r for r in train_results if r.get("loss") is not None],
                    key=lambda x: x.get("loss", float("inf")),
                    default=None,
                )
                if train_results
                else None
            )
            if best_run is not None and hasattr(trainer, "cholesky_jitter"):
                temp_cb = FinalParameterStorageCallback(verbose=False)
                extracted = temp_cb._extract_final_parameters(
                    model,
                    epoch=best_run.get("init_index", 0),
                    best_loss=best_run.get("loss"),
                    cholesky_jitter=trainer.cholesky_jitter,
                    best_epoch=None,
                )
                if extracted:
                    run_id = int(best_run.get("init_index", 0)) + 1
                    record = {
                        "run": run_id,
                        "num_epochs": extracted.get(
                            "num_epochs",
                            getattr(trainer, "num_epochs", None),
                        ),
                        "best_epoch": extracted.get("best_epoch"),
                        "best_loss": best_run.get("loss"),
                        "jitter": extracted.get("jitter"),
                        "initial": None,
                        "final": extracted,
                    }
                    all_stored_params = [record]
                    gp_trainer_info = build_trainer_analysis_payload(all_stored_params, train_results, num_epochs)

            if gp_trainer_info is None:
                gp_trainer_info = {
                    "inits": [],
                    "best_parameters": None,
                    "average_final_parameters": {},
                }
        # Add trainer timing to gp_trainer_info for JSON output
        if gp_trainer_info is not None and hasattr(trainer, "full_train_time"):
            if trainer.full_train_time is not None:
                gp_trainer_info["full_train_time"] = float(trainer.full_train_time)
            if trainer.train_time is not None:
                gp_trainer_info["train_time"] = float(trainer.train_time)
            if trainer.log_time is not None:
                gp_trainer_info["log_time"] = float(trainer.log_time)

    # Always return 4 values (gp_trainer_info may be None)
    return gp_metric, y_pred_np, output_std_np, gp_trainer_info


def train_eval_PFN(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    amp_device: str,
    amp_dtype,
    regressor=None,
    y_train_mean=None,
    y_train_std=None,
    standardize_y_log_scale: bool = False,
    log_scale_C: float | None = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    source_cols=None,
):
    """
    Thin wrapper around PFN evaluation identical to v1/v2.
    Left unchanged so existing PFN experiments can swap to train_eval3
    without behavioural changes on the PFN side.
    """
    import numpy as np
    import torch

    try:
        from tabpfn import TabPFNRegressor as StandardTabPFNRegressor

        is_standard_tabpfn = isinstance(regressor, StandardTabPFNRegressor)
    except ImportError:
        is_standard_tabpfn = False

    if is_standard_tabpfn:
        X_train_np = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        X_test_np = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_train_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_test_np = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

        if y_train_np.ndim > 1:
            y_train_np = y_train_np.ravel()
        if y_test_np.ndim > 1:
            y_test_np = y_test_np.ravel()

        t_fit_start = time.time()
        if not hasattr(regressor, "feature_names_in_"):
            regressor.fit(X_train_np, y_train_np)
        training_time = time.time() - t_fit_start

        t_pred_start = time.time()
        y_var_tabpfn = None
        lower_95_test = None
        upper_95_test = None

        full_predictions = regressor.predict(
            X_test_np,
            output_type="full",
            quantiles=[0.025, 0.975],
        )
        t_predict_call = time.time() - t_pred_start
        print(f"[TIMER] predict(output_type='full') took: {t_predict_call:.4f}s")

        if "logits" in full_predictions and "criterion" in full_predictions:
            logits = full_predictions["logits"]
            criterion = full_predictions["criterion"]
            if hasattr(criterion, "variance"):
                t_var_calc_start = time.time()
                if isinstance(logits, np.ndarray):
                    logits = torch.tensor(logits)
                variance = criterion.variance(logits)
                y_var_tabpfn = variance.detach().cpu().numpy()
                t_var_calc = time.time() - t_var_calc_start
                print(f"[TIMER] criterion.variance(logits) calculation took: {t_var_calc:.4f}s")
                y_pred_tabpfn = full_predictions.get("mean")

                tabpfn_logits_for_crps = logits.detach().cpu()
                tabpfn_bar_dist_for_crps = criterion
                print(f"[CRPS] Storing logits shape: {logits.shape} for CRPS")

        if "quantiles" in full_predictions:
            q_95 = full_predictions["quantiles"]
            if isinstance(q_95, list) and len(q_95) >= 2:
                lower_95_test, upper_95_test = q_95[0], q_95[1]

        if isinstance(y_pred_tabpfn, torch.Tensor):
            y_pred_tabpfn = y_pred_tabpfn.detach().cpu().numpy()
        if isinstance(y_var_tabpfn, torch.Tensor):
            y_var_tabpfn = y_var_tabpfn.detach().cpu().numpy()
        if y_pred_tabpfn.ndim > 1:
            y_pred_tabpfn = y_pred_tabpfn.ravel()
        if isinstance(y_var_tabpfn, np.ndarray) and y_var_tabpfn.ndim > 1:
            y_var_tabpfn = y_var_tabpfn.ravel()
        elif not isinstance(y_var_tabpfn, np.ndarray):
            y_var_tabpfn = np.array(y_var_tabpfn)
        prediction_time = time.time() - t_pred_start

        y_pred_test = y_pred_tabpfn
        output_std_test = np.sqrt(y_var_tabpfn)
        if "tabpfn_logits_for_crps" in locals():
            tabpfn_logits_test = tabpfn_logits_for_crps
            tabpfn_bar_dist_test = tabpfn_bar_dist_for_crps
    else:
        t_fit_start = time.time()
        X_all = np.concatenate([X_train, X_test], axis=0)
        Y_all = np.concatenate([y_train, np.zeros_like(y_test)], axis=0)

        X_all = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1)
        Y_all = torch.tensor(Y_all, dtype=torch.float32).reshape(-1, 1, 1)
        training_time = time.time() - t_fit_start

        single_eval_pos = len(X_train)
        t_pred_start = time.time()
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_all, Y_all, single_eval_pos)
            logits = out["standard"]
            y_mean = regressor.predict_mean(logits)
            y_var = regressor.predict_variance(logits)

            logits_test = logits[-len(y_test) :]
            tabpfn_logits_for_crps = logits_test.detach().cpu()
            tabpfn_bar_dist_for_crps = regressor.bardist_
            print(f"[CRPS] Storing logits shape: {logits_test.shape} for CRPS")

        y_pred = y_mean.detach().cpu().numpy().reshape(-1)
        output_std = (y_var.detach().cpu().numpy().reshape(-1)) ** 0.5
        prediction_time = time.time() - t_pred_start

        y_pred_test = y_pred[-len(y_test) :]
        output_std_test = output_std[-len(y_test) :]
        lower_95_test = None
        upper_95_test = None
        try:
            logits_test_for_q = logits[-len(y_test) :]
            q025 = regressor.bardist_.icdf(logits_test_for_q, 0.025).detach().cpu().numpy().reshape(-1)
            q975 = regressor.bardist_.icdf(logits_test_for_q, 0.975).detach().cpu().numpy().reshape(-1)
            lower_95_test, upper_95_test = q025, q975
        except Exception:
            lower_95_test, upper_95_test = None, None
        if "tabpfn_logits_for_crps" in locals():
            tabpfn_logits_test = tabpfn_logits_for_crps
            tabpfn_bar_dist_test = tabpfn_bar_dist_for_crps

    y_test_normalized = y_test.copy() if isinstance(y_test, np.ndarray) else np.array(y_test)

    if (y_train_mean is not None) and (y_train_std is not None):
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            if source_cols is not None:
                if isinstance(X_test, np.ndarray):
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                else:
                    X_test_tensor = X_test

                is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
                if is_onehot:
                    onehot_cols = X_test_tensor[:, source_cols]
                    source_indices_test = torch.argmax(onehot_cols, dim=1)
                else:
                    source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
                    source_indices_test = X_test_tensor[:, source_col].long()

                source_indices_test_np = source_indices_test.detach().cpu().numpy()

                y_pred_test_denorm = y_pred_test.copy()
                output_std_test_denorm = output_std_test.copy()
                lower_95_test_denorm = lower_95_test.copy() if lower_95_test is not None else None
                upper_95_test_denorm = upper_95_test.copy() if upper_95_test is not None else None
                unique_sources = np.unique(source_indices_test_np)

                for source_idx in unique_sources:
                    source_mask = source_indices_test_np == source_idx
                    source_key = int(source_idx)

                    if source_key in y_train_mean:
                        mean = float(y_train_mean[source_key])
                        std = float(y_train_std[source_key])
                    elif 0 in y_train_mean:
                        mean = float(y_train_mean[0])
                        std = float(y_train_std[0])
                    else:
                        first_key = list(y_train_mean.keys())[0]
                        mean = float(y_train_mean[first_key])
                        std = float(y_train_std[first_key])

                    if standardize_y_log_scale:
                        log_y_pred = (y_pred_test[source_mask] * std) + mean
                        log_y_std = output_std_test[source_mask] * std
                        max_log_val = 700.0
                        log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                        exp_log_y = np.exp(log_y_pred)
                        y_pred_test_denorm[source_mask] = exp_log_y - log_scale_C
                        output_std_test_denorm[source_mask] = exp_log_y * log_y_std

                        if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                            log_lower = (lower_95_test[source_mask] * std) + mean
                            log_upper = (upper_95_test[source_mask] * std) + mean
                            log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                            log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                            lower_95_test_denorm[source_mask] = np.exp(log_lower) - log_scale_C
                            upper_95_test_denorm[source_mask] = np.exp(log_upper) - log_scale_C
                    else:
                        y_pred_test_denorm[source_mask] = (y_pred_test[source_mask] * std) + mean
                        output_std_test_denorm[source_mask] = output_std_test[source_mask] * std
                        if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                            lower_95_test_denorm[source_mask] = (lower_95_test[source_mask] * std) + mean
                            upper_95_test_denorm[source_mask] = (upper_95_test[source_mask] * std) + mean

                y_pred_test = y_pred_test_denorm
                output_std_test = output_std_test_denorm
                if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                    lower_95_test = lower_95_test_denorm
                    upper_95_test = upper_95_test_denorm
            else:
                first_key = list(y_train_mean.keys())[0]
                mean = float(y_train_mean[first_key])
                std = float(y_train_std[first_key])
                if standardize_y_log_scale:
                    log_y_pred = (y_pred_test * std) + mean
                    log_y_std = output_std_test * std
                    max_log_val = 700.0
                    log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                    exp_log_y = np.exp(log_y_pred)
                    y_pred_test = exp_log_y - log_scale_C
                    output_std_test = exp_log_y * log_y_std
                    if lower_95_test is not None and upper_95_test is not None:
                        log_lower = (lower_95_test * std) + mean
                        log_upper = (upper_95_test * std) + mean
                        log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                        log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                        lower_95_test = np.exp(log_lower) - log_scale_C
                        upper_95_test = np.exp(log_upper) - log_scale_C
                else:
                    y_pred_test = (y_pred_test * std) + mean
                    output_std_test = output_std_test * std
                    if lower_95_test is not None and upper_95_test is not None:
                        lower_95_test = (lower_95_test * std) + mean
                        upper_95_test = (upper_95_test * std) + mean
        else:
            if standardize_y_log_scale:
                log_y_pred = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                log_y_std = output_std_test * float(y_train_std)
                max_log_val = 700.0
                log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                exp_log_y = np.exp(log_y_pred)
                y_pred_test = exp_log_y - log_scale_C
                output_std_test = exp_log_y * log_y_std
                if lower_95_test is not None and upper_95_test is not None:
                    log_lower = (lower_95_test * float(y_train_std)) + float(y_train_mean)
                    log_upper = (upper_95_test * float(y_train_std)) + float(y_train_mean)
                    log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                    log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                    lower_95_test = np.exp(log_lower) - log_scale_C
                    upper_95_test = np.exp(log_upper) - log_scale_C
            else:
                y_pred_test = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                output_std_test = output_std_test * float(y_train_std)
                if lower_95_test is not None and upper_95_test is not None:
                    lower_95_test = (lower_95_test * float(y_train_std)) + float(y_train_mean)
                    upper_95_test = (upper_95_test * float(y_train_std)) + float(y_train_mean)

    tabpfn_logits_for_metrics = tabpfn_logits_test if "tabpfn_logits_test" in locals() else None
    tabpfn_bar_dist_for_metrics = tabpfn_bar_dist_test if "tabpfn_bar_dist_test" in locals() else None
    y_test_for_crps = y_test_normalized if "y_test_normalized" in locals() else y_test
    metrics = compute_metrics(
        y_test,
        y_pred_test,
        output_std_test,
        training_time=training_time,
        prediction_time=prediction_time,
        tabpfn_logits=tabpfn_logits_for_metrics,
        tabpfn_bar_dist=tabpfn_bar_dist_for_metrics,
        y_test_normalized=y_test_for_crps,
        lower_95=lower_95_test,
        upper_95=upper_95_test,
    )

    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]

        pfn_per_source_metric = compute_per_source_metrics(
            y_test, y_pred_test, output_std_test, X_test, source_columns=source_cols
        )

        for source_name, source_metrics in pfn_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                metrics[f"{source_name}_{metric_name}"] = metric_value

    return metrics, y_pred_test, output_std_test

