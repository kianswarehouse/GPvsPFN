
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error, r2_score

# Use a non-interactive backend to avoid Tkinter dependency in non-main threads
try:
    import matplotlib
    matplotlib.use('Agg', force=True)
except Exception:
    pass


def compute_metrics(y_true, y_hat, output_std=None, start_time=None, training_time=None, prediction_time=None):
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)

    Returns:
        dict: Dictionary with computed metrics including time information
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Handle time metrics
    if training_time is not None and prediction_time is not None:
        # New approach: separate training and prediction times
        total_time = training_time + prediction_time
        metrics = {
            "Total_Time": total_time,
            "Training_Time": training_time,
            "Prediction_Time": prediction_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    elif start_time is not None:
        # Legacy approach: single start_time
        elapsed_time = time.time() - start_time
        metrics = {
            "Time": elapsed_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    else:
        # No time information
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics

    return metrics


def adjust_predictive_variance_for_test_noise(output_std, test_noise_std):
    """
    Adjust predictive standard deviation to account for additional test noise.
    
    When test data has noise added that is not accounted for in the model's
    predictive variance, this function adds the test noise variance to the
    predictive variance to get the total uncertainty.
    
    Args:
        output_std: Predictive standard deviation from the model (includes training noise)
        test_noise_std: Standard deviation of the noise added to test targets
        
    Returns:
        Adjusted standard deviation: sqrt(predictive_variance + test_noise_variance)
        
    Example:
        If model predicts with std=0.1 and test noise has std=0.05:
        >>> adjusted_std = adjust_predictive_variance_for_test_noise(0.1, 0.05)
        >>> # adjusted_std = sqrt(0.1^2 + 0.05^2) = sqrt(0.01 + 0.0025) ≈ 0.112
        
    Note:
        This is useful when evaluating with noisy test data. The model's predictive
        variance includes training noise, but not test noise. Adding test noise variance
        gives the total uncertainty for comparing against noisy test targets.
    """
    if isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy()
    if isinstance(test_noise_std, torch.Tensor):
        test_noise_std = test_noise_std.detach().cpu().numpy()
    
    # Convert to numpy arrays if needed
    output_std = np.asarray(output_std)
    test_noise_std = np.asarray(test_noise_std)
    
    # Add variances (var = std^2)
    adjusted_variance = output_std**2 + test_noise_std**2
    adjusted_std = np.sqrt(adjusted_variance)
    
    return adjusted_std


def format_metric_value(key: str, value: float, precision: int = 4) -> str:
    """
    Format a metric value appropriately based on its key.
    
    Args:
        key: The metric key (e.g., 'jitter', 'noise', 'RRMSE')
        value: The value to format
        precision: Number of decimal places (for non-scientific notation)
    
    Returns:
        Formatted string representation of the value
    """
    if key in ['jitter', 'noise']:
        # Use scientific notation for jitter and noise
        return f"{value:.6e}"
    elif key in ['num_epochs', 'best_epoch']:
        # Integer values
        return f"{int(value)}"
    else:
        # Default formatting
        return f"{value:.{precision}f}"


def analyze_metrics(metrics_list, print_summary: bool = False, label: str = None, title: str = None):
    """
    Summarize metrics across seeds for RRMSE and NIS, including per-source statistics.

    Args:
        metrics_list: list of dicts, each containing metric values for a seed

    Returns:
        dict with per-metric summary: {metric: {mean, std, median, min, max}}
        and per-source summaries for RRMSE and NIS
    """
    import numpy as np
    import pandas as pd

    if metrics_list is None or len(metrics_list) == 0:
        return {}

    df = pd.DataFrame(metrics_list)

    # Mean/std for all available metric columns
    # Detailed stats for specific metrics
    detailed = {}
    for m in ["RRMSE", "NIS", "Time", "Total_Time", "Training_Time", "Prediction_Time", 
              "num_epochs", "best_epoch", "jitter", "raw_noise", "outputscale"]:
        if m in df.columns:
            vals = df[m].dropna().values
            if len(vals) == 0:
                continue
            vals = vals.astype(float)
            detailed[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": int(len(vals)),
            }
    
    # Handle individual lengthscale metrics (lengthscale_0, lengthscale_1, etc.)
    lengthscale_columns = [col for col in df.columns if col.startswith("lengthscale_")]
    for lengthscale_col in sorted(lengthscale_columns):  # Sort to ensure consistent ordering
        vals = df[lengthscale_col].dropna().values
        if len(vals) == 0:
            continue
        vals = vals.astype(float)
        detailed[lengthscale_col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": int(len(vals)),
        }

    # Extract per-source metrics for RRMSE and NIS
    per_source_stats = {}
    source_columns = [col for col in df.columns if col.startswith('source_') and ('_RRMSE' in col or '_NIS' in col)]
    
    if source_columns:
        # Group by source
        sources = {}
        for col in source_columns:
            source_name = col.split('_')[0] + '_' + col.split('_')[1]  # e.g., 'source_0'
            metric_name = col.split('_', 2)[2]  # e.g., 'RRMSE' or 'NIS'
            
            if source_name not in sources:
                sources[source_name] = {}
            sources[source_name][metric_name] = col
        
        # Compute statistics for each source
        for source_name, metrics in sources.items():
            per_source_stats[source_name] = {}
            for metric_name, col_name in metrics.items():
                if col_name in df.columns:
                    vals = df[col_name].dropna().values
                    if len(vals) > 0:
                        vals = vals.astype(float)
                        per_source_stats[source_name][metric_name] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "median": float(np.median(vals)),
                            "min": float(np.min(vals)),
                            "max": float(np.max(vals)),
                            "count": int(len(vals)),
                        }

    if print_summary and len(detailed) > 0:
        header = f"{label} summary" if label else "Summary"
        label_print = (label or 'Summary')

        if title:
            print(f"\n{label_print} over {len(metrics_list)} seeds for {title} (RRMSE, NIS):")
        else:
            print(f"\n{label_print} over {len(metrics_list)} seeds (RRMSE, NIS):")

        for m, s in detailed.items():
            # Format output based on metric type
            if m.startswith("lengthscale_"):
                # For individual lengthscales
                print(
                    f"  {m}: median={s['median']:.6f} | min={s['min']:.6f} | max={s['max']:.6f} | "
                    f"mean={s['mean']:.6f} ± {s['std']:.6f} (n={s['count']})"
                )
            elif m in ["num_epochs", "best_epoch"]:
                # For integer metrics, show as integers
                print(
                    f"  {m}: median={s['median']:.0f} | min={s['min']:.0f} | max={s['max']:.0f} | "
                    f"mean={s['mean']:.1f} ± {s['std']:.1f} (n={s['count']})"
                )
            elif m in ["jitter", "noise"]:
                # Use scientific notation for jitter and noise
                print(
                    f"  {m}: median={s['median']:.6e} | min={s['min']:.6e} | max={s['max']:.6e} | "
                    f"mean={s['mean']:.6e} ± {s['std']:.6e} (n={s['count']})"
                )
            else:
                print(
                    f"  {m}: median={s['median']:.6f} | min={s['min']:.6f} | max={s['max']:.6f} | "
                    f"mean={s['mean']:.6f} ± {s['std']:.6f} (n={s['count']})"
                )
        
        # Print per-source statistics
        if per_source_stats:
            print(f"\n{label_print} Per-Source Statistics:")
            for source_name, source_metrics in per_source_stats.items():
                print(f"  {source_name}:")
                for metric_name, stats in source_metrics.items():
                    print(
                        f"    {metric_name}: median={stats['median']:.6f} | min={stats['min']:.6f} | max={stats['max']:.6f} | "
                        f"mean={stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})"
                    )

    # Add per-source stats to the return value
    if per_source_stats:
        detailed['per_source'] = per_source_stats

    return detailed


def plot_metrics(*args, labels: list = None, title: str = None, save_path: str = None, subplots: bool = True):
    """
    Plot per-seed metric VALUES (not aggregates) for multiple runs as violin plots.

    Args:
        metrics_lists: list of lists, where each inner list is a metrics_list
                       (the same structure you pass to analyze_metrics), i.e.,
                       a list of dicts with keys like 'RRMSE', 'NIS', 'Time'.
        labels: optional list of names for each metrics_list; defaults to
                ["run_0", ...].
        subplots: if True (default), returns both individual plots AND combined plots.
                 if False, returns only individual plots.

    Returns:
        dict: Dictionary containing 'individual' and optionally 'combined' figure objects.
              - 'individual': dict with 'RRMSE' and 'NIS' figure objects
              - 'combined': figure object with both metrics in subplots (only when subplots=True)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Debug: print save_path if provided
    if save_path is not None:
        print(f"[DEBUG plot_metrics] save_path provided: {save_path}")
    else:
        print(f"[DEBUG plot_metrics] save_path is None - plots will not be saved")

    # Normalize inputs: allow plot_metric_values(run1, run2, ...) or plot_metric_values([run1, run2, ...])
    if len(args) == 1 and isinstance(args[0], list) and (
        len(args[0]) == 0 or isinstance(args[0][0], (dict, list))
    ):
        metrics_lists = args[0]
    else:
        metrics_lists = list(args)

    if labels is None:
        labels = [f"run_{i}" for i in range(len(metrics_lists))]

    def extract(vals_list, key):
        out = []
        for ml in metrics_lists:
            # ml should be list[dict]
            if not isinstance(ml, list):
                out.append(np.array([], dtype=float))
                continue
            arr = [d[key] for d in ml if isinstance(d, dict) and key in d and d[key] is not None]
            out.append(np.array(arr, dtype=float) if len(arr) > 0 else np.array([], dtype=float))
        return out

    # Determine a representative seed count (use min across lists to be safe)
    seed_counts = []
    for ml in metrics_lists:
        seed_counts.append(len(ml) if isinstance(ml, list) else 0)
    n_seeds = min(seed_counts) if len(seed_counts) > 0 else 0

    def create_violin_plot(ax, data, metric, labels, n_seeds):
        """Helper function to create a violin plot on the given axis."""
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#888888')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Overlay mean (blue) and median (red) lines
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(mean_v, i - 0.25, i + 0.25, colors='blue', linewidth=2)
            ax.hlines(med_v, i - 0.25, i + 0.25, colors='red', linewidth=2)

        # Legend: blue = mean, red = median
        try:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color='blue', lw=2, label='Mean'),
                Line2D([0], [0], color='red', lw=2, label='Median'),
            ]
            ax.legend(handles=legend_handles, loc='upper right', frameon=False)
        except Exception:
            pass

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} distribution (n={n_seeds})")
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    def save_figure(fig, metric_name, save_path, title):
        """Helper function to save a figure."""
        if save_path is not None:
            from pathlib import Path
            p = Path(save_path)
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[WARNING] Failed to create directory {save_path}: {e}")
                return
            fname = f"{metric_name}" + (f"_{title}" if title else "") + ".png"
            full_path = p / fname
            try:
                fig.savefig(str(full_path), dpi=300, bbox_inches="tight")
                print(f"[INFO] Saved plot to: {full_path}")
                # Close only when we actually save, to avoid leaking figures
                try:
                    plt.close(fig)
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARNING] Failed to save plot to {full_path}: {e}")
                import traceback
                traceback.print_exc()

    # Always create individual plots
    individual_figs = {}
    for metric in ["RRMSE", "NIS"]:
        data = extract(metrics_lists, metric)
        fig, ax = plt.subplots(figsize=(7, 4))
        create_violin_plot(ax, data, metric, labels, n_seeds)
        
        if title:
            try:
                fig.suptitle(title)
            except Exception:
                pass
        
        plt.tight_layout()
        save_figure(fig, metric.lower(), save_path, title)
        individual_figs[metric] = fig

    result = {'individual': individual_figs}

    # Create combined plot if subplots=True
    if subplots:
        # Create one figure with two subplots
        combined_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Plot RRMSE
        rrmse_data = extract(metrics_lists, "RRMSE")
        create_violin_plot(ax1, rrmse_data, "RRMSE", labels, n_seeds)
        
        # Plot NIS
        nis_data = extract(metrics_lists, "NIS")
        create_violin_plot(ax2, nis_data, "NIS", labels, n_seeds)
        
        # Set overall title if provided
        if title:
            try:
                combined_fig.suptitle(title)
            except Exception:
                pass
        
        plt.tight_layout()
        save_figure(combined_fig, "metrics_combined", save_path, title)
        result['combined'] = combined_fig

    return result

def compute_per_source_metrics(y_true, y_hat, output_std, X_test, source_columns, start_time=None, training_time=None, prediction_time=None):
    """
    Compute metrics for each source separately.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        X_test: Test features (2D array) containing source information
        source_columns: Either a single column index (int) or list of column indices for source identification
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)

    Returns:
        dict: Dictionary with overall metrics and per-source metrics including time information
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Store the overall standard deviation for consistent normalization
    overall_std = y_true.std()

    # Handle source_columns parameter
    if isinstance(source_columns, int):
        # Single column case - filter by value in that column
        source_values = np.unique(X_test[:, source_columns])
        num_sources = len(source_values)
        source_indices = {}
        for i, val in enumerate(source_values):
            source_indices[f"source_{i}"] = X_test[:, source_columns] == val
    else:
        # Multiple columns case - one-hot encoded sources
        # Convert to list if it's a numpy array
        if isinstance(source_columns, np.ndarray):
            source_columns = source_columns.tolist()

        num_sources = len(source_columns)
        source_indices = {}
        for i in range(num_sources):
            source_indices[f"source_{i}"] = X_test[:, source_columns[i]] == 1

    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_hat, output_std, start_time, training_time, prediction_time)
    # Add sample size to overall metrics (as integer)
    overall_metrics["num_samples"] = int(len(y_true))

    # Compute per-source metrics
    per_source_metrics = {}
    for source_name, source_mask in source_indices.items():
        if np.sum(source_mask) > 0:  # Only compute if source has data
            source_y_true = y_true[source_mask]
            source_y_hat = y_hat[source_mask]
            source_output_std = output_std[source_mask] if output_std is not None else None

            # Compute source metrics with source-specific normalization
            # Compute source metrics with source-specific normalization
            source_rmse = np.sqrt(mean_squared_error(source_y_true, source_y_hat))
            source_std = source_y_true.std()
            source_rrmse = source_rmse / source_std if source_std > 0 else np.inf

            # Handle time metrics for per-source
            if training_time is not None and prediction_time is not None:
                # New approach: separate training and prediction times
                total_time = training_time + prediction_time
                source_metrics = {
                    "Total_Time": total_time,
                    "Training_Time": training_time,
                    "Prediction_Time": prediction_time,
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            elif start_time is not None:
                # Legacy approach: single start_time
                elapsed_time = time.time() - start_time
                source_metrics = {
                    "Time": elapsed_time,
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            else:
                # No time information
                source_metrics = {
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }

            # Add NIS if output_std is provided
            if source_output_std is not None:
                z = 1.96
                L = source_y_hat - z * source_output_std
                U = source_y_hat + z * source_output_std
                width = U - L
                below = (L - source_y_true) * (source_y_true < L)
                above = (source_y_true - U) * (source_y_true > U)
                interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
                source_nis = interval_score.mean() / source_y_true.std()  # Use per-source target std for NIS
                source_metrics["NIS"] = source_nis

            per_source_metrics[source_name] = source_metrics

    # Combine overall and per-source metrics
    all_metrics = {"overall": overall_metrics, "per_source": per_source_metrics, "num_sources": num_sources}

    return all_metrics


def extract_parameter_statistics(gp_parameters_file="gp_parameters.json"):
    """
    Extract parameter statistics from the gp_parameters.json file.
    
    Args:
        gp_parameters_file: Path to the gp_parameters.json file
        
    Returns:
        dict: Parameter statistics including initial, final, and deltas for each parameter
    """
    import json
    import numpy as np
    from pathlib import Path
    
    try:
        # Read the gp_parameters.json file
        param_file = Path(gp_parameters_file)
        if not param_file.exists():
            return {"error": f"Parameter file {gp_parameters_file} not found"}
        
        with open(param_file, 'r') as f:
            parameters_data = json.load(f)
        
        if not parameters_data:
            return {"error": "No parameter data found"}
        
        # Extract parameter statistics
        param_stats = {
            "raw_noise": {
                "initial": [],
                "final": [],
                "deltas": []
            },
            "raw_outputscale": {
                "initial": [],
                "final": [],
                "deltas": []
            },
            "raw_lengthscales": {
                "initial": [],
                "final": [],
                "deltas": []
            }
        }
        
        # Collect all parameter values across runs
        for run_data in parameters_data:
            for param_name in ["raw_noise", "raw_outputscale", "raw_lengthscales"]:
                if param_name in run_data.get("initial", {}):
                    initial_val = run_data["initial"][param_name]
                    final_val = run_data["final"][param_name]
                    delta_val = run_data["deltas"][param_name]
                    
                    # Handle different data types
                    if param_name == "raw_lengthscales":
                        # For lengthscales, store as lists
                        param_stats[param_name]["initial"].append(initial_val if initial_val is not None else [])
                        param_stats[param_name]["final"].append(final_val if final_val is not None else [])
                        param_stats[param_name]["deltas"].append(delta_val if delta_val is not None else [])
                    else:
                        # For scalar parameters
                        param_stats[param_name]["initial"].append(initial_val if initial_val is not None else 0.0)
                        param_stats[param_name]["final"].append(final_val if final_val is not None else 0.0)
                        param_stats[param_name]["deltas"].append(delta_val if delta_val is not None else 0.0)
        
        # Compute summary statistics for each parameter
        summary_stats = {}
        for param_name, param_data in param_stats.items():
            summary_stats[param_name] = {}
            
            for stat_type in ["initial", "final", "deltas"]:
                values = param_data[stat_type]
                
                if param_name == "raw_lengthscales":
                    # For lengthscales, compute stats across all dimensions
                    if values and len(values) > 0 and len(values[0]) > 0:
                        # Flatten all lengthscale values
                        flat_values = []
                        for val_list in values:
                            if val_list:  # Check if not empty
                                flat_values.extend(val_list)
                        
                        if flat_values:
                            summary_stats[param_name][stat_type] = {
                                "mean": float(np.mean(flat_values)),
                                "std": float(np.std(flat_values, ddof=1)) if len(flat_values) > 1 else 0.0,
                                "median": float(np.median(flat_values)),
                                "min": float(np.min(flat_values)),
                                "max": float(np.max(flat_values)),
                                "count": len(flat_values),
                                "raw_values": values  # Keep raw values for reference
                            }
                        else:
                            summary_stats[param_name][stat_type] = {
                                "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                                "raw_values": values
                            }
                    else:
                        summary_stats[param_name][stat_type] = {
                            "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                            "raw_values": values
                        }
                else:
                    # For scalar parameters
                    if values:
                        values_array = np.array(values)
                        summary_stats[param_name][stat_type] = {
                            "mean": float(np.mean(values_array)),
                            "std": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                            "median": float(np.median(values_array)),
                            "min": float(np.min(values_array)),
                            "max": float(np.max(values_array)),
                            "count": len(values_array),
                            "raw_values": values
                        }
                    else:
                        summary_stats[param_name][stat_type] = {
                            "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0,
                            "raw_values": values
                        }
        
        # Add metadata
        summary_stats["metadata"] = {
            "total_runs": len(parameters_data),
            "parameter_file": str(param_file),
            "kernel_types": list(set([run.get("initial", {}).get("kernel_type", "Unknown") for run in parameters_data])),
            "input_dims": list(set([run.get("initial", {}).get("input_dim", "Unknown") for run in parameters_data]))
        }
        
        return summary_stats
        
    except Exception as e:
        return {"error": f"Failed to extract parameter statistics: {str(e)}"}