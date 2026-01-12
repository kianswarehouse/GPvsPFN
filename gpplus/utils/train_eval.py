import time

import numpy as np
import torch
from gpplus.training import GPTrainer
from gpplus.training.callbacks import FinalParameterStorageCallback
# from gpplus.training.eval import evaluate_gp_model
from gpplus.training.stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition
from gpplus.training.eval2 import evaluate_gp_model
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils.metrics_functions import compute_metrics, compute_per_source_metrics


def _process_trainer_info(stored_params, train_results, num_epochs):
    """
    Process stored parameters from callback into a structured trainer log.
    
    Args:
        stored_params: List of parameter records from FinalParameterStorageCallback
        train_results: Results from trainer.train()
        num_epochs: Number of epochs used in training
        
    Returns:
        dict: Structured trainer log with runs, best_parameters, and average_final_parameters
    """
    import numpy as np
    
    runs_data = []
    best_run_idx = None
    best_loss = float('inf')
    
    # Process each run
    for i, record in enumerate(stored_params):
        run_data = {
            "run": record.get("run", i + 1),
            "num_epochs": record.get("num_epochs", num_epochs),
            "best_epoch": record.get("best_epoch"),
            "epoch": record.get("epoch"),
            "loss": record.get("best_loss"),
            "jitter": record.get("jitter"),
            "initial_parameters": record.get("initial", {}),
            "final_parameters": record.get("final", {}),
        }
        runs_data.append(run_data)
        
        # Track best run
        loss = record.get("best_loss")
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_run_idx = i
    
    # Extract best parameters
    best_parameters = None
    if best_run_idx is not None and best_run_idx < len(stored_params):
        best_record = stored_params[best_run_idx]
        best_parameters = {
            "run": best_record.get("run", best_run_idx + 1),
            "num_epochs": best_record.get("num_epochs", num_epochs),
            "best_epoch": best_record.get("best_epoch"),
            "loss": best_record.get("best_loss"),
            "jitter": best_record.get("jitter"),
            "initial_parameters": best_record.get("initial", {}),
            "final_parameters": best_record.get("final", {}),
        }
    
    # Calculate average final parameters statistics
    avg_final_params = {}
    if stored_params:
        # Collect all final parameter values
        param_collections = {}
        
        for record in stored_params:
            final = record.get("final", {})
            for key, value in final.items():
                if key not in param_collections:
                    param_collections[key] = []
                
                # Handle different types of values
                if value is None:
                    continue
                elif isinstance(value, (list, tuple)):
                    # For lists (like lengthscales), collect all values
                    if key not in param_collections:
                        param_collections[key] = []
                    param_collections[key].extend([float(v) for v in value if v is not None])
                elif isinstance(value, (int, float)):
                    param_collections[key].append(float(value))
                elif hasattr(value, 'item'):  # Tensor
                    param_collections[key].append(float(value.item()))
        
        # Calculate statistics for each parameter
        for key, values in param_collections.items():
            if len(values) > 0:
                values_array = np.array(values)
                avg_final_params[key] = {
                    "min": float(np.min(values_array)),
                    "mean": float(np.mean(values_array)),
                    "median": float(np.median(values_array)),
                    "std": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                    "max": float(np.max(values_array)),
                    "count": int(len(values_array)),
                }
    
    return {
        "runs": runs_data,
        "best_parameters": best_parameters,
        "average_final_parameters": avg_final_params,
    }


def train_eval_gp(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    seed: int,
    num_runs: int,
    lr: float,
    convergence_patience: int=10,
    min_loss_change: float=1e-7,
    optimizer_class=None,
    initializer_class=None,
    device: str = "cpu",
    # dtype: torch.dtype = torch.float64,
    y_train_mean: torch.Tensor | None = None,
    y_train_std: torch.Tensor | None = None,
    standardize_y_log_scale: bool = False,
    log_scale_C: float = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    y_train_min: torch.Tensor | dict | None = None,  # Unused, kept for backward compatibility
    source_cols: int | list[int] | None = None,
    trainer_info: bool = False,
    # Prescreening parameters
    enable_prescreening: bool = False,
    num_test: int | None = None,  # Total candidates to evaluate (default: num_runs * input_dim)
    prescreening_warmup_epochs: int = 0,
    prescreening_warmup_lr: float = 0.01,
    prescreening_optimizer_class=None,  # Optimizer for warmup (default: torch.optim.Adam)
    recorder=None,  # PrescreeningRecorder instance
):
    """
    Train a GP model and evaluate metrics on the provided test set.

    If y_train_mean/std are provided, predictions and uncertainty are denormalized
    before metrics are computed.

    If source_cols is provided, per-source metrics will be computed and added to the
    main metrics dictionary with source-specific prefixes.

    Args:
        source_cols: Column index(es) for source identification. If provided, per-source
                    metrics will be computed and added to the main metrics.
                    - If int: single source column (data not encoded)
                    - If list with 1 element: converted to int (data not encoded)
                    - If list with multiple elements: multiple source columns (one-hot encoded data)

    Returns:
        gp_metric: dict of computed metrics (includes Total_Time, Training_Time, Prediction_Time)
                  and per-source metrics if source_cols is provided
        y_pred: numpy array of predictions (denormalized if mean/std provided)
        output_std: numpy array of predictive std (denormalized if mean/std provided)
    """
    # Set optimizer kwargs based on optimizer type
    # if optimizer_class == LBFGSScipy or (
    #     hasattr(optimizer_class, "__name__") and optimizer_class.__name__ == "LBFGSScipy"
    # ):
    #     optimizer_kwargs = {
    #         "max_iter": 2000,
    #         "max_eval": 5000,
    #         "tolerance_grad": 1e-5,
    #         "tolerance_change": 1e-9,
    #         "history_size": 10,
    #     }
    # else:
    #     optimizer_kwargs = {"lr": lr}

    callbacks = [FinalParameterStorageCallback(save_file=None, verbose=False)]

    trainer = GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs,
        stop_conditions=[ConvergencePatienceStopCondition(patience=convergence_patience), MinLossChangeStopCondition(min_loss_change=min_loss_change)],
        callbacks=callbacks,
        optimizer_class=optimizer_class,
        device=device,
        initializer_class=initializer_class,
        enable_prescreening=enable_prescreening,
        num_test=num_test,
        prescreening_warmup_epochs=prescreening_warmup_epochs,
        prescreening_warmup_lr=prescreening_warmup_lr,
        prescreening_optimizer_class=prescreening_optimizer_class,
        recorder=recorder,
    )

    # Measure training time
    t_train_start = time.time()
    train_results = trainer.train()
    training_time = time.time() - t_train_start

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
        # Replace NaN/Inf with 0 (mean in standardized space) - this will be handled correctly in inverse transform
        y_pred = torch.where(torch.isfinite(y_pred), y_pred, torch.zeros_like(y_pred))
    
    if torch.any(~torch.isfinite(output_std)):
        nan_count = torch.sum(~torch.isfinite(output_std)).item()
        import warnings
        warnings.warn(
            f"train_eval_gp: Model output_std contains {nan_count} NaN/Inf values before denormalization. "
            "This may indicate model training issues."
        )
        # Replace NaN/Inf with small positive value
        output_std = torch.where(torch.isfinite(output_std), output_std, torch.ones_like(output_std) * 1e-6)

    if y_train_mean is not None and y_train_std is not None:
        # Check if y_train_mean/std are dictionaries (method 2: per-source standardization)
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            # Extract source indices from X_test for per-source denormalization
            if source_cols is not None:
                is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
                if is_onehot:
                    onehot_cols = X_test[:, source_cols]
                    source_indices_test = torch.argmax(onehot_cols, dim=1)
                else:
                    source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
                    source_indices_test = X_test[:, source_col].long()

                # Denormalize per source
                y_pred_denorm = torch.zeros_like(y_pred)
                output_std_denorm = torch.zeros_like(output_std)
                unique_sources = torch.unique(source_indices_test)

                for source_idx in unique_sources:
                    source_mask = source_indices_test == source_idx
                    source_key = source_idx.item()

                    # Use the mean/std for this source, or fallback to source 0 or first available
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
                    
                    # Ensure mean/std are properly squeezed to avoid broadcasting issues
                    if isinstance(mean, torch.Tensor):
                        mean = mean.squeeze()
                    if isinstance(std, torch.Tensor):
                        std = std.squeeze()

                    if standardize_y_log_scale:
                        # Unstandardize in log space
                        log_y_pred = (y_pred[source_mask] * std) + mean
                        log_y_std = output_std[source_mask] * std
                        # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                        max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                        log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                        # Convert from log space to original space: exp(log_data) - C
                        # This is the inverse of log(y + C)
                        exp_log_y = torch.exp(log_y_pred)
                        y_pred_denorm[source_mask] = exp_log_y - log_scale_C
                        # Use delta method: std_original ≈ exp(log_mean) * log_std
                        output_std_denorm[source_mask] = exp_log_y * log_y_std
                    else:
                        y_pred_denorm[source_mask] = (y_pred[source_mask] * std) + mean
                        output_std_denorm[source_mask] = output_std[source_mask] * std

                y_pred = y_pred_denorm
                output_std = output_std_denorm
            else:
                # If source_cols not provided but we have dicts, use first available source
                first_key = list(y_train_mean.keys())[0]
                mean = y_train_mean[first_key]
                std = y_train_std[first_key]
                
                # Ensure mean/std are properly squeezed to avoid broadcasting issues
                if isinstance(mean, torch.Tensor):
                    mean = mean.squeeze()
                if isinstance(std, torch.Tensor):
                    std = std.squeeze()
                if standardize_y_log_scale:
                    # Unstandardize in log space
                    log_y_pred = (y_pred * std) + mean
                    log_y_std = output_std * std
                    # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                    max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                    log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                    # Convert from log space to original space: exp(log_data) - C
                    # This is the inverse of log(y + C)
                    exp_log_y = torch.exp(log_y_pred)
                    y_pred = exp_log_y - log_scale_C
                    # Use delta method: std_original ≈ exp(log_mean) * log_std
                    output_std = exp_log_y * log_y_std
                else:
                    y_pred = (y_pred * std) + mean
                    output_std = output_std * std
                    print(f"y_pred after denorm (normal, first 5): {y_pred[:5].squeeze().tolist()}")
                    print(f"y_pred after denorm (normal, last 5): {y_pred[-5:].squeeze().tolist()}")
        else:
            # Single mean/std for all data (methods 0 and 1)
            # Ensure mean/std are properly squeezed to avoid broadcasting issues
            mean_val = y_train_mean.squeeze() if isinstance(y_train_mean, torch.Tensor) else y_train_mean
            std_val = y_train_std.squeeze() if isinstance(y_train_std, torch.Tensor) else y_train_std
            
            if standardize_y_log_scale:
                # Unstandardize in log space
                log_y_pred = (y_pred * std_val) + mean_val
                log_y_std = output_std * std_val
                # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                # Convert from log space to original space: exp(log_data) - C
                # This is the inverse of log(y + C)
                exp_log_y = torch.exp(log_y_pred)
                y_pred = exp_log_y - log_scale_C
                # Use delta method: std_original ≈ exp(log_mean) * log_std
                output_std = exp_log_y * log_y_std
                print(f"y_pred after denorm (single, first 5): {y_pred[:5].squeeze().tolist()}")
                print(f"y_pred after denorm (single, last 5): {y_pred[-5:].squeeze().tolist()}")
            else:
                y_pred = (y_pred * std_val) + mean_val
                output_std = output_std * std_val

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)
    
    # Check for NaN/Inf values and replace with reasonable defaults
    if np.any(~np.isfinite(y_pred_np)):
        nan_mask = ~np.isfinite(y_pred_np)
        if standardize_y_log_scale:
            # For log-scale, use median of valid predictions or fallback to exp(log_mean) - epsilon
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            if len(valid_preds) > 0:
                replacement = np.median(valid_preds)
            else:
            # Fallback: use exp(log_mean) - epsilon (inverse of log(y + epsilon))
                if y_train_mean is not None:
                    if isinstance(y_train_mean, dict):
                        log_mean_val = list(y_train_mean.values())[0]
                    else:
                        log_mean_val = y_train_mean
                    if isinstance(log_mean_val, torch.Tensor):
                        log_mean_val = log_mean_val.item() if log_mean_val.numel() == 1 else log_mean_val.cpu().numpy()
                    replacement = np.exp(float(log_mean_val)) - log_scale_C
                else:
                    replacement = 0.0
            y_pred_np[nan_mask] = replacement
        else:
            # For standard scale, use median of valid predictions
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            replacement = np.median(valid_preds) if len(valid_preds) > 0 else 0.0
            y_pred_np[nan_mask] = replacement
    
    if np.any(~np.isfinite(output_std_np)):
        nan_mask = ~np.isfinite(output_std_np)
        valid_std = output_std_np[np.isfinite(output_std_np)]
        replacement = np.median(valid_std) if len(valid_std) > 0 else 1.0
        output_std_np[nan_mask] = replacement

    gp_metric = compute_metrics(
        y_test, y_pred_np, output_std_np, training_time=training_time, prediction_time=prediction_time
    )

    # Extract noise and noise_std (will be added in correct order later)
    noise_std = None
    noise_std_original_scale = None
    try:
        # Get noise (variance) from likelihood - this is already transformed (10^raw_noise)
        noise_variance = model.likelihood.noise.detach().cpu()
        # Compute noise std = sqrt(noise)
        noise_std = np.sqrt(noise_variance)

        # Convert to original output scale if y was standardized
        if y_train_std is not None:
            if isinstance(y_train_std, dict):
                # For per-source std, use first available or source 0
                if 0 in y_train_std:
                    std_to_use = y_train_std[0]
                    mean_to_use = y_train_mean[0] if isinstance(y_train_mean, dict) and 0 in y_train_mean else None
                else:
                    std_to_use = list(y_train_std.values())[0]
                    mean_to_use = list(y_train_mean.values())[0] if isinstance(y_train_mean, dict) else None
            else:
                std_to_use = y_train_std
                mean_to_use = y_train_mean
            
            if standardize_y_log_scale:
                # For log-scale, noise_std is in log-standardized space
                # Convert it to original scale: noise_std_original ≈ exp(log_mean) * noise_std_log
                if mean_to_use is not None:
                    # Convert to tensors/numpy as needed
                    if isinstance(mean_to_use, torch.Tensor):
                        log_mean = mean_to_use.item() if mean_to_use.numel() == 1 else mean_to_use.cpu().numpy()
                    else:
                        log_mean = float(mean_to_use) if np.isscalar(mean_to_use) else mean_to_use
                    
                    if isinstance(std_to_use, torch.Tensor):
                        log_std = std_to_use.item() if std_to_use.numel() == 1 else std_to_use.cpu().numpy()
                    else:
                        log_std = float(std_to_use) if np.isscalar(std_to_use) else std_to_use
                    
                    # Unstandardize noise_std in log space, then convert to original space
                    log_noise_std = noise_std * log_std
                    # Convert from log space to original space using delta method
                    noise_std_original_scale = np.exp(log_mean) * log_noise_std
                else:
                    # Fallback: just use std conversion (less accurate)
                    if isinstance(std_to_use, torch.Tensor):
                        std_val = std_to_use.item() if std_to_use.numel() == 1 else std_to_use.cpu().numpy()
                    else:
                        std_val = std_to_use
                    noise_std_original_scale = noise_std * std_val
            else:
                if isinstance(std_to_use, torch.Tensor):
                    std_val = std_to_use.item() if std_to_use.numel() == 1 else std_to_use.cpu().numpy()
                else:
                    std_val = std_to_use
                noise_std_original_scale = noise_std * std_val
        else:
            noise_std_original_scale = noise_std
    except Exception as e:
        # If extraction fails, don't break the function
        import logging

        logging.warning(f"Could not extract noise std: {e}")

    # Always extract directly from the model after training (the best model is already loaded)
    # This is more reliable than relying on callbacks, especially with multi-run training
    # where callbacks are deep-copied and JSON files get overwritten
    best_model_metrics = None

    # Find best run from train_results to get metadata
    best_run = (
        min(
            [r for r in train_results if r.get("loss") is not None],
            key=lambda x: x.get("loss", float("inf")),
            default=None,
        )
        if train_results
        else None
    )

    # Determine num_epochs from trainer
    num_epochs_actual = trainer.num_epochs if hasattr(trainer, "num_epochs") else None

    if hasattr(trainer, "cholesky_jitter"):
        # Extract metrics directly from the loaded best model (which is already in model after training)
        # Create a temporary callback instance to use the extraction method
        temp_callback = FinalParameterStorageCallback(verbose=False)
        extracted_params = temp_callback._extract_final_parameters(
            model,
            epoch=best_run.get("run_index", 0) if best_run else 0,  # Use run_index as proxy for epoch
            best_loss=best_run.get("loss") if best_run else None,
            cholesky_jitter=trainer.cholesky_jitter,
            best_epoch=None,  # Not available from results
        )
        if extracted_params:
            lengthscales_extracted = extracted_params.get("lengthscales")
            cat_lengthscales_extracted = extracted_params.get("cat_lengthscales")
            source_lengthscales_extracted = extracted_params.get("source_lengthscales")
            # Convert to expected format
            best_model_metrics = {
                "num_epochs": num_epochs_actual
                if num_epochs_actual is not None
                else extracted_params.get("num_epochs"),
                "best_epoch": extracted_params.get("best_epoch"),
                "best_loss": best_run.get("loss") if best_run else extracted_params.get("best_loss"),
                "jitter": extracted_params.get("jitter"),
                "raw_noise": extracted_params.get("raw_noise"),
                "outputscale": extracted_params.get("outputscale"),
                "lengthscales": lengthscales_extracted,
                "cat_lengthscales": cat_lengthscales_extracted,
                "source_lengthscales": source_lengthscales_extracted,
            }

    if best_model_metrics:
        # Add metrics in desired order: jitter, raw_noise (all), noise (all), noise_std (all), outputscale
        gp_metric["jitter"] = best_model_metrics.get("jitter")

        # Helper to add array/scalar metrics
        def add_metric(name, value):
            if value is None:
                return
            # Convert to numpy if it's a tensor
            if hasattr(value, "numpy"):
                value = value.numpy()
            # Check if it's a single-element array
            if hasattr(value, "size") and value.size == 1:
                gp_metric[name] = float(value.item() if hasattr(value, "item") else value.flat[0])
            elif hasattr(value, "__len__") and len(value) > 1:
                for i, v in enumerate(value):
                    gp_metric[f"{name}_{i}"] = float(v)
            else:
                gp_metric[name] = float(value)

        # Extract and add raw_noise
        try:
            raw_noise = model.likelihood.raw_noise.detach().cpu()
            add_metric("raw_noise", raw_noise.numpy().flatten() if raw_noise.numel() > 1 else raw_noise.item())
        except Exception:
            add_metric("raw_noise", best_model_metrics.get("raw_noise"))

        add_metric("noise", noise_std)
        add_metric("noise_std", noise_std_original_scale)
        gp_metric["outputscale"] = best_model_metrics.get("outputscale")

        # Add other metrics
        gp_metric.update(
            {
                "num_epochs": best_model_metrics.get("num_epochs"),
                "best_epoch": best_model_metrics.get("best_epoch"),
            }
        )
        # Add individual lengthscales as separate metrics
        lengthscales = best_model_metrics.get("lengthscales")
        # Debug output
        if lengthscales is None:
            print(
                f"[DEBUG train_eval] lengthscales is None. best_model_metrics keys: {list(best_model_metrics.keys())}"
            )
            print(f"[DEBUG train_eval] best_model_metrics content: {best_model_metrics}")
        elif isinstance(lengthscales, (list, tuple)):
            print(
                f"[DEBUG train_eval] Found {len(lengthscales)} cont_lengthscales: {lengthscales[:5]}..."
                if len(lengthscales) > 5
                else f"[DEBUG train_eval] Found {len(lengthscales)} cont_lengthscales: {lengthscales}"
            )
            if len(lengthscales) > 0:
                for i, ls_val in enumerate(lengthscales):
                    gp_metric[f"cont_lengthscale_{i}"] = ls_val
            else:
                # Empty list - this shouldn't happen but handle it
                import logging

                logging.warning("lengthscales is an empty list")
        else:
            # Single value case (scalar)
            print(f"[DEBUG train_eval] lengthscales is a scalar: {lengthscales}")
            gp_metric["cont_lengthscale_0"] = lengthscales

        # Add individual cat_lengthscales as separate metrics
        cat_lengthscales = best_model_metrics.get("cat_lengthscales")
        if cat_lengthscales is not None:
            if isinstance(cat_lengthscales, (list, tuple)):
                print(
                    f"[DEBUG train_eval] Found {len(cat_lengthscales)} cat_lengthscales: {cat_lengthscales[:5]}..."
                    if len(cat_lengthscales) > 5
                    else f"[DEBUG train_eval] Found {len(cat_lengthscales)} cat_lengthscales: {cat_lengthscales}"
                )
                if len(cat_lengthscales) > 0:
                    for i, ls_val in enumerate(cat_lengthscales):
                        gp_metric[f"cat_lengthscale_{i}"] = ls_val
            else:
                # Single value case (scalar)
                print(f"[DEBUG train_eval] cat_lengthscales is a scalar: {cat_lengthscales}")
                gp_metric["cat_lengthscale_0"] = cat_lengthscales

        # Add individual source_lengthscales as separate metrics
        source_lengthscales = best_model_metrics.get("source_lengthscales")
        if source_lengthscales is not None:
            if isinstance(source_lengthscales, (list, tuple)):
                print(
                    (
                        f"[DEBUG train_eval] Found {len(source_lengthscales)} "
                        f"source_lengthscales: {source_lengthscales[:5]}..."
                    )
                    if len(source_lengthscales) > 5
                    else (
                        f"[DEBUG train_eval] Found {len(source_lengthscales)} "
                        f"source_lengthscales: {source_lengthscales}"
                    )
                )
                if len(source_lengthscales) > 0:
                    for i, ls_val in enumerate(source_lengthscales):
                        gp_metric[f"source_lengthscale_{i}"] = ls_val
            else:
                # Single value case (scalar)
                print(f"[DEBUG train_eval] source_lengthscales is a scalar: {source_lengthscales}")
                gp_metric["source_lengthscale_0"] = source_lengthscales
    else:
        print("[DEBUG train_eval] No best_model_metrics found at all!")

        # Still add noise metrics even if best_model_metrics is None
        # Helper to add array/scalar metrics
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

    # Compute per-source metrics if source_cols is provided
    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        # Convert single-element list to integer for consistency
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

        # Add per-source metrics directly to the main metrics dictionary
        for source_name, source_metrics in gp_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                gp_metric[f"{source_name}_{metric_name}"] = metric_value

    # Add y_train_mean and y_train_std to metrics for this run
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            # Method 2: per-source means/stds
            for source_key, mean_val in y_train_mean.items():
                gp_metric[f"y_train_mean_source_{source_key}"] = float(
                    mean_val.item() if hasattr(mean_val, "item") else mean_val
                )
            for source_key, std_val in y_train_std.items():
                gp_metric[f"y_train_std_source_{source_key}"] = float(
                    std_val.item() if hasattr(std_val, "item") else std_val
                )
        else:
            # Methods 0 and 1: single mean/std
            # Handle tensors that may need squeezing (e.g., from LogScaler with keepdim=True)
            if hasattr(y_train_mean, "item"):
                mean_val = y_train_mean.item() if y_train_mean.numel() == 1 else y_train_mean.squeeze().item()
            else:
                mean_val = y_train_mean
            if hasattr(y_train_std, "item"):
                std_val = y_train_std.item() if y_train_std.numel() == 1 else y_train_std.squeeze().item()
            else:
                std_val = y_train_std
            gp_metric["y_train_mean"] = float(mean_val)
            gp_metric["y_train_std"] = float(std_val)

    # Process trainer info if requested
    gp_trainer_info = None
    if trainer_info:
        # Collect stored parameters from all runs in train_results
        all_stored_params = []
        for result in train_results:
            callback_data = result.get("callback_data", {})
            # Look for FinalParameterStorageCallback data
            for cb_key, stored_params_list in callback_data.items():
                if "FinalParameterStorage" in cb_key or "ParameterStorage" in cb_key:
                    all_stored_params.extend(stored_params_list)
        
        if all_stored_params:
            gp_trainer_info = _process_trainer_info(all_stored_params, train_results, num_epochs)
        else:
            # Return empty structure if no parameters stored
            gp_trainer_info = {
                "runs": [],
                "best_parameters": None,
                "average_final_parameters": {},
            }

    # Always return 4 values (gp_trainer_info will be None if trainer_info is False)
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
    log_scale_C: float = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    y_train_min=None,  # Unused, kept for backward compatibility
    source_cols=None,
):
    """
    Train/evaluate TabPFN on provided split and return metrics, preds, std.

    Supports both TabPFNRegressor (sklearn-like API) and VanillaDirectTabPFNRegressor (custom API).

    If source_cols is provided, per-source metrics will be computed and added to the
    main metrics dictionary with source-specific prefixes.

    Args:
        regressor: TabPFNRegressor or VanillaDirectTabPFNRegressor instance. 
                  If None, creates a new VanillaDirectTabPFNRegressor.
        source_cols: Column index(es) for source identification. If provided, per-source
                    metrics will be computed and added to the main metrics.
                    - If int: single source column (data not encoded)
                    - If list with 1 element: converted to int (data not encoded)
                    - If list with multiple elements: multiple source columns (one-hot encoded data)

    Returns:
        metrics: dict of computed metrics and per-source metrics if source_cols is provided
        y_pred_test: numpy array of predictions
        output_std_test: numpy array of predictive std
    """
    import numpy as np
    import torch

    if regressor is None:
        from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
        regressor = VanillaDirectTabPFNRegressor(device=amp_device)

    # Check if regressor is standard TabPFNRegressor (sklearn-like API) or wrapper (custom API)
    try:
        from tabpfn import TabPFNRegressor as StandardTabPFNRegressor
        is_standard_tabpfn = isinstance(regressor, StandardTabPFNRegressor)
    except ImportError:
        is_standard_tabpfn = False
    
    if is_standard_tabpfn:
        # Standard TabPFN library - use sklearn-like API
        # Convert to numpy arrays
        X_train_np = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        X_test_np = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_train_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_test_np = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        
        # Ensure 1D arrays for y
        if y_train_np.ndim > 1:
            y_train_np = y_train_np.ravel()
        if y_test_np.ndim > 1:
            y_test_np = y_test_np.ravel()
        
        # Fit if not already fitted
        if not hasattr(regressor, 'feature_names_in_'):
            regressor.fit(X_train_np, y_train_np)
        
        t_start = time.time()
        # Predict using "main" to get both mean and variance in a dict
        predictions = regressor.predict(X_test_np, output_type="main")
        
        # Extract mean and variance from the dict
        y_pred_tabpfn = predictions.get("mean", predictions.get("median"))
        y_var_tabpfn = predictions.get("variance")
        
        # If variance not available, try to get it separately or compute from quantiles
        if y_var_tabpfn is None:
            # Try to get variance directly (might not be supported in all versions)
            try:
                y_var_tabpfn = regressor.predict(X_test_np, output_type="variance")
            except (ValueError, KeyError):
                # Fallback: estimate variance from quantiles if available
                if "quantiles" in predictions:
                    quantiles = predictions["quantiles"]
                    if isinstance(quantiles, list) and len(quantiles) >= 2:
                        # Use interquartile range as proxy for std: IQR/1.35 ≈ std for normal dist
                        q75_idx = len(quantiles) - 1
                        q25_idx = 0
                        iqr = quantiles[q75_idx] - quantiles[q25_idx]
                        y_var_tabpfn = (iqr / 1.35) ** 2
                    else:
                        # Last resort: use a small default variance
                        y_var_tabpfn = np.ones_like(y_pred_tabpfn) * 0.01
                else:
                    y_var_tabpfn = np.ones_like(y_pred_tabpfn) * 0.01
        
        # Convert to numpy if needed
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
        
        y_pred_test = y_pred_tabpfn
        output_std_test = np.sqrt(y_var_tabpfn)
    else:
        # Wrapper API (VanillaDirectTabPFNRegressor) - use custom forward/predict_mean/predict_variance API
        X_all = np.concatenate([X_train, X_test], axis=0)
        Y_all = np.concatenate([y_train, np.zeros_like(y_test)], axis=0)

        X_all = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1)
        Y_all = torch.tensor(Y_all, dtype=torch.float32).reshape(-1, 1, 1)

        single_eval_pos = len(X_train)
        t_start = time.time()
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_all, Y_all, single_eval_pos)
            logits = out["standard"]
            y_mean = regressor.predict_mean(logits)
            y_var = regressor.predict_variance(logits)

        y_pred = y_mean.detach().cpu().numpy().reshape(-1)
        output_std = (y_var.detach().cpu().numpy().reshape(-1)) ** 0.5

        # Metrics only on the test segment
        y_pred_test = y_pred[-len(y_test) :]
        output_std_test = output_std[-len(y_test) :]

    # If train mean/std are provided (e.g., standardized training targets),
    # denormalize predictions and std before computing metrics to compare on original scale
    if (y_train_mean is not None) and (y_train_std is not None):
        # Check if y_train_mean/std are dictionaries (method 2: per-source standardization)
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            # Extract source indices from X_test for per-source denormalization
            if source_cols is not None:
                # Convert X_test to tensor if it's numpy
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

                # Convert to numpy for indexing
                source_indices_test_np = source_indices_test.detach().cpu().numpy()

                # Denormalize per source
                y_pred_test_denorm = y_pred_test.copy()
                output_std_test_denorm = output_std_test.copy()
                unique_sources = np.unique(source_indices_test_np)

                for source_idx in unique_sources:
                    source_mask = source_indices_test_np == source_idx
                    source_key = int(source_idx)

                    # Use the mean/std for this source, or fallback to source 0 or first available
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
                        # Unstandardize in log space
                        log_y_pred = (y_pred_test[source_mask] * std) + mean
                        log_y_std = output_std_test[source_mask] * std
                        # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                        max_log_val = 700.0
                        log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                        # Convert from log space to original space: exp(log_data) - C
                        # This is the inverse of log(y + C)
                        exp_log_y = np.exp(log_y_pred)
                        y_pred_test_denorm[source_mask] = exp_log_y - log_scale_C
                        # Use delta method: std_original ≈ exp(log_mean) * log_std
                        output_std_test_denorm[source_mask] = exp_log_y * log_y_std
                    else:
                        # Normal unstandardization (not log scale)
                        y_pred_test_denorm[source_mask] = (y_pred_test[source_mask] * std) + mean
                        output_std_test_denorm[source_mask] = output_std_test[source_mask] * std

                # Update predictions and std after processing all sources
                y_pred_test = y_pred_test_denorm
                output_std_test = output_std_test_denorm
            else:
                # If source_cols not provided but we have dicts, use first available source
                first_key = list(y_train_mean.keys())[0]
                mean = float(y_train_mean[first_key])
                std = float(y_train_std[first_key])
                if standardize_y_log_scale:
                    # Unstandardize in log space
                    log_y_pred = (y_pred_test * std) + mean
                    log_y_std = output_std_test * std
                    # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                    max_log_val = 700.0
                    log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                    # Convert from log space to original space: exp(log_data) - C
                    # This is the inverse of log(y + C)
                    exp_log_y = np.exp(log_y_pred)
                    y_pred_test = exp_log_y - log_scale_C
                    # Use delta method: std_original ≈ exp(log_mean) * log_std
                    output_std_test = exp_log_y * log_y_std
                else:
                    y_pred_test = (y_pred_test * std) + mean
                    output_std_test = output_std_test * std
        else:
            # Single mean/std for all data (methods 0 and 1)
            if standardize_y_log_scale:
                # Unstandardize in log space
                log_y_pred = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                log_y_std = output_std_test * float(y_train_std)
                # Clamp to prevent overflow in exp (exp(700) ≈ inf for float32)
                max_log_val = 700.0
                log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                # Convert from log space to original space: exp(log_data) - C
                # This is the inverse of log(y + C)
                exp_log_y = np.exp(log_y_pred)
                y_pred_test = exp_log_y - log_scale_C
                # Use delta method: std_original ≈ exp(log_mean) * log_std
                output_std_test = exp_log_y * log_y_std
            else:
                y_pred_test = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                output_std_test = output_std_test * float(y_train_std)

    metrics = compute_metrics(y_test, y_pred_test, output_std_test, start_time=t_start)

    # Compute per-source metrics if source_cols is provided
    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        # Convert single-element list to integer for consistency
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]

        pfn_per_source_metric = compute_per_source_metrics(
            y_test, y_pred_test, output_std_test, X_test, source_columns=source_cols
        )

        # Add per-source metrics directly to the main metrics dictionary
        for source_name, source_metrics in pfn_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                metrics[f"{source_name}_{metric_name}"] = metric_value

    return metrics, y_pred_test, output_std_test
