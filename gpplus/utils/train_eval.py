import time
import torch
import numpy as np

from gpplus.training import GPTrainer
from gpplus.training.eval import evaluate_gp_model
from gpplus.training.eval2 import evaluate_gp_model as evaluate_gp_model2
from gpplus.utils.metrics_functions import compute_metrics, compute_per_source_metrics
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from gpplus.training.callbacks import FinalParameterStorageCallback
from gpplus.training.optimizers import LBFGSScipy

def train_eval_gp(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    seed: int,
    num_runs: int,
    lr: float,
    convergence_patience: int,
    optimizer_class = torch.optim.Adam,
    initializer_class = None,
    device: str = 'cpu',
    # dtype: torch.dtype = torch.float64,
    y_train_mean: torch.Tensor | None = None,
    y_train_std: torch.Tensor | None = None,
    source_cols: int | list[int] | None = None,
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
    if optimizer_class == LBFGSScipy or (hasattr(optimizer_class, '__name__') and optimizer_class.__name__ == 'LBFGSScipy'):
        optimizer_kwargs = {
            'max_iter': 2000,
            'max_eval': 5000,
            'tolerance_grad': 1e-5,
            'tolerance_change': 1e-9,
            'history_size': 10,
        }
    else:
        optimizer_kwargs = {"lr": lr}
    
    callbacks = [FinalParameterStorageCallback(save_file="gp_parameters.json", verbose=False)]
    
    trainer = GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs,
        optimizer_kwargs=optimizer_kwargs,
        convergence_patience=convergence_patience,
        callbacks=callbacks,
        optimizer_class=optimizer_class,
        device=device,
        initializer_class=initializer_class,
    )

    # Measure training time
    t_train_start = time.time()
    train_results = trainer.train()
    training_time = time.time() - t_train_start

    # Measure prediction time
    t_pred_start = time.time()
    y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
    prediction_time = time.time() - t_pred_start

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
                    
                    y_pred_denorm[source_mask] = (y_pred[source_mask] * std) + mean
                    output_std_denorm[source_mask] = output_std[source_mask] * std
                
                y_pred = y_pred_denorm
                output_std = output_std_denorm
            else:
                # If source_cols not provided but we have dicts, use first available source
                first_key = list(y_train_mean.keys())[0]
                y_pred = (y_pred * y_train_std[first_key]) + y_train_mean[first_key]
                output_std = output_std * y_train_std[first_key]
        else:
            # Single mean/std for all data (methods 0 and 1)
            y_pred = (y_pred * y_train_std) + y_train_mean
            output_std = output_std * y_train_std

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)

    gp_metric = compute_metrics(y_test, y_pred_np, output_std_np, training_time=training_time, prediction_time=prediction_time)

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
                else:
                    std_to_use = list(y_train_std.values())[0]
                noise_std_original_scale = noise_std * std_to_use
            else:
                noise_std_original_scale = noise_std * y_train_std
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
    best_run = min(
        [r for r in train_results if r.get("loss") is not None],
        key=lambda x: x.get("loss", float("inf")),
        default=None
    ) if train_results else None
    
    # Determine num_epochs from trainer
    num_epochs_actual = trainer.num_epochs if hasattr(trainer, 'num_epochs') else None
    
    if hasattr(trainer, 'cholesky_jitter'):
        # Extract metrics directly from the loaded best model (which is already in model after training)
        # Create a temporary callback instance to use the extraction method
        temp_callback = FinalParameterStorageCallback(verbose=False)
        extracted_params = temp_callback._extract_final_parameters(
            model,
            epoch=best_run.get("run_index", 0) if best_run else 0,  # Use run_index as proxy for epoch
            best_loss=best_run.get("loss") if best_run else None,
            cholesky_jitter=trainer.cholesky_jitter,
            best_epoch=None  # Not available from results
        )
        if extracted_params:
            lengthscales_extracted = extracted_params.get("lengthscales")
            cat_lengthscales_extracted = extracted_params.get("cat_lengthscales")
            source_lengthscales_extracted = extracted_params.get("source_lengthscales")
            # Convert to expected format
            best_model_metrics = {
                "num_epochs": num_epochs_actual if num_epochs_actual is not None else extracted_params.get("num_epochs"),
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
            if hasattr(value, 'numpy'):
                value = value.numpy()
            # Check if it's a single-element array
            if hasattr(value, 'size') and value.size == 1:
                gp_metric[name] = float(value.item() if hasattr(value, 'item') else value.flat[0])
            elif hasattr(value, '__len__') and len(value) > 1:
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
        gp_metric.update({
            "num_epochs": best_model_metrics.get("num_epochs"),
            "best_epoch": best_model_metrics.get("best_epoch"),
        })
        # Add individual lengthscales as separate metrics
        lengthscales = best_model_metrics.get("lengthscales")
        # Debug output
        if lengthscales is None:
            print(f"[DEBUG train_eval] lengthscales is None. best_model_metrics keys: {list(best_model_metrics.keys())}")
            print(f"[DEBUG train_eval] best_model_metrics content: {best_model_metrics}")
        elif isinstance(lengthscales, (list, tuple)):
            print(f"[DEBUG train_eval] Found {len(lengthscales)} cont_lengthscales: {lengthscales[:5]}..." if len(lengthscales) > 5 else f"[DEBUG train_eval] Found {len(lengthscales)} cont_lengthscales: {lengthscales}")
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
                print(f"[DEBUG train_eval] Found {len(cat_lengthscales)} cat_lengthscales: {cat_lengthscales[:5]}..." if len(cat_lengthscales) > 5 else f"[DEBUG train_eval] Found {len(cat_lengthscales)} cat_lengthscales: {cat_lengthscales}")
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
                print(f"[DEBUG train_eval] Found {len(source_lengthscales)} source_lengthscales: {source_lengthscales[:5]}..." if len(source_lengthscales) > 5 else f"[DEBUG train_eval] Found {len(source_lengthscales)} source_lengthscales: {source_lengthscales}")
                if len(source_lengthscales) > 0:
                    for i, ls_val in enumerate(source_lengthscales):
                        gp_metric[f"source_lengthscale_{i}"] = ls_val
            else:
                # Single value case (scalar)
                print(f"[DEBUG train_eval] source_lengthscales is a scalar: {source_lengthscales}")
                gp_metric["source_lengthscale_0"] = source_lengthscales
    else:
        print(f"[DEBUG train_eval] No best_model_metrics found at all!")
        # Still add noise metrics even if best_model_metrics is None
        # Helper to add array/scalar metrics
        def add_metric(name, value):
            if value is None:
                return
            if hasattr(value, 'size'):
                if value.size == 1:
                    gp_metric[name] = float(value.item() if hasattr(value, 'item') else value)
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
    if (
        isinstance(source_cols, int)
        or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0)
    ):
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
            prediction_time=prediction_time
        )
        
        # Add per-source metrics directly to the main metrics dictionary
        for source_name, source_metrics in gp_per_source_metric['per_source'].items():
            for metric_name, metric_value in source_metrics.items():
                gp_metric[f"{source_name}_{metric_name}"] = metric_value

    # Add y_train_mean and y_train_std to metrics for this run
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            # Method 2: per-source means/stds
            for source_key, mean_val in y_train_mean.items():
                gp_metric[f"y_train_mean_source_{source_key}"] = float(mean_val.item() if hasattr(mean_val, 'item') else mean_val)
            for source_key, std_val in y_train_std.items():
                gp_metric[f"y_train_std_source_{source_key}"] = float(std_val.item() if hasattr(std_val, 'item') else std_val)
        else:
            # Methods 0 and 1: single mean/std
            gp_metric["y_train_mean"] = float(y_train_mean.item() if hasattr(y_train_mean, 'item') else y_train_mean)
            gp_metric["y_train_std"] = float(y_train_std.item() if hasattr(y_train_std, 'item') else y_train_std)

    return gp_metric, y_pred_np, output_std_np


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
    source_cols=None,
):
    """
    Train/evaluate TabPFN on provided split and return metrics, preds, std.

    Expects regressor with forward/predict_mean/predict_variance API.
    
    If source_cols is provided, per-source metrics will be computed and added to the
    main metrics dictionary with source-specific prefixes.

    Args:
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
        regressor = VanillaDirectTabPFNRegressor(device=amp_device)

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
    y_pred_test = y_pred[-len(y_test):]
    output_std_test = output_std[-len(y_test):]

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
                    
                    y_pred_test_denorm[source_mask] = (y_pred_test[source_mask] * std) + mean
                    output_std_test_denorm[source_mask] = output_std_test[source_mask] * std
                
                y_pred_test = y_pred_test_denorm
                output_std_test = output_std_test_denorm
            else:
                # If source_cols not provided but we have dicts, use first available source
                first_key = list(y_train_mean.keys())[0]
                y_pred_test = (y_pred_test * float(y_train_std[first_key])) + float(y_train_mean[first_key])
                output_std_test = output_std_test * float(y_train_std[first_key])
        else:
            # Single mean/std for all data (methods 0 and 1)
            y_pred_test = (y_pred_test * float(y_train_std)) + float(y_train_mean)
            output_std_test = output_std_test * float(y_train_std)

    metrics = compute_metrics(y_test, y_pred_test, output_std_test, start_time=t_start)

    # Compute per-source metrics if source_cols is provided
    if (
        isinstance(source_cols, int)
        or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0)
    ):
        # Convert single-element list to integer for consistency
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]
            
        pfn_per_source_metric = compute_per_source_metrics(
            y_test, 
            y_pred_test, 
            output_std_test, 
            X_test,
            source_columns=source_cols
        )
        
        # Add per-source metrics directly to the main metrics dictionary
        for source_name, source_metrics in pfn_per_source_metric['per_source'].items():
            for metric_name, metric_value in source_metrics.items():
                metrics[f"{source_name}_{metric_name}"] = metric_value

    return metrics, y_pred_test, output_std_test

