"""
Plot epoch vs training metrics from GP_Trainer_Analysis.json files.

Reads epoch_metrics from each run (list of {epoch, loss, NLL, NIS, LOO_NLL, KF, Residual_MSE})
and plots one subplot per metric: epoch (x) vs metric value (y).
All runs as thin grey lines; chosen run (lowest loss per fold) in green; best RRMSE run in red.

Usage:
  python plot_epoch_metrics.py path/to/gpVpfn_*_GP_Trainer_Analysis.json   # single file
  python plot_epoch_metrics.py results_v2.0/NLL3                          # all trainer_analysis JSONs under NLL3
  python plot_epoch_metrics.py                                            # default: results_v2.0/NLL3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_trainer_analysis_hyperparams import (
    extract_runs_and_chosen,
    find_trainer_analysis_jsons,
    load_trainer_analysis,
)

# Metrics to plot (keys in each epoch_metrics / lbfgs_inner_metrics entry)
EPOCH_METRIC_KEYS = ["loss", "NLL", "NIS", "LOO_NLL", "KF", "Residual_MSE"]
ITER_METRIC_KEYS = EPOCH_METRIC_KEYS


def _get_metric_arrays(
    run: dict,
    entry_key: str,
    x_field: str,
    metric_keys: list[str],
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Generic helper: for a single run, extract (x, values) per metric from a list of dicts
    stored under run[entry_key]. Returns dict[metric_key] = (xs, ys).
    """
    em = run.get(entry_key) or []
    if not em:
        return {}
    out: dict[str, tuple[list[float], list[float]]] = {}
    for key in metric_keys:
        xs: list[float] = []
        ys: list[float] = []
        for e in em:
            x = e.get(x_field)
            v = e.get(key)
            if x is not None and v is not None:
                try:
                    xs.append(float(x))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    pass
        if xs and ys:
            out[key] = (xs, ys)
    return out


def _get_epoch_metric_arrays(run: dict) -> dict[str, tuple[list[float], list[float]]]:
    """Epoch-level metrics: use 'epoch_metrics' and x field 'epoch'."""
    return _get_metric_arrays(run, entry_key="epoch_metrics", x_field="epoch", metric_keys=EPOCH_METRIC_KEYS)


def _get_iter_metric_arrays(run: dict) -> dict[str, tuple[list[float], list[float]]]:
    """Inner-iteration metrics: use 'lbfgs_inner_metrics' and x field 'lbfgs_iter'."""
    return _get_metric_arrays(run, entry_key="lbfgs_inner_metrics", x_field="lbfgs_iter", metric_keys=ITER_METRIC_KEYS)


def plot_epoch_metrics_from_data(
    data: dict,
    out_dir: Path,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
    title: str | None = None,
) -> Path:
    """
    Plot epoch vs training metrics from in-memory trainer_info data (same structure as saved JSON).
    Use this right after saving the trainer_analysis JSON so epoch-metrics plots are created automatically.
    Returns the path where the figure was saved. Raises ValueError if no runs have epoch_metrics.
    """
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("epoch_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with epoch_metrics in trainer_info data")

    title = title or data.get("title", "trainer_analysis")
    return _plot_epoch_metrics_impl(data, out_dir, title, figsize_per_subplot)


def _plot_epoch_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """Shared implementation: data already loaded, title and out_dir set."""
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("epoch_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with epoch_metrics")

    metric_keys = []
    for key in EPOCH_METRIC_KEYS:
        if any(_get_epoch_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No epoch metric data found")

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        for r in all_runs:
            arrs = _get_epoch_metric_arrays(r)
            ep_v, val_v = arrs.get(key, ([], []))
            if not ep_v or not val_v:
                continue
            ep_v = np.array(ep_v)
            val_v = np.array(val_v)
            if r.get("_best_rrmse"):
                ax.plot(ep_v, val_v, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (run)" if not labeled_best else None)
                labeled_best = True
            elif r.get("_chosen"):
                ax.plot(ep_v, val_v, c="C2", alpha=0.85, linewidth=2, zorder=5,
                        label="Chosen (run)" if not labeled_chosen else None)
                labeled_chosen = True
            else:
                ax.plot(ep_v, val_v, c="gray", alpha=0.45, linewidth=0.9, zorder=1,
                        label="All runs" if not labeled_all else None)
                labeled_all = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Epoch vs metrics — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"epoch_metrics_{title}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_epoch_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Load a GP_Trainer_Analysis.json, extract epoch_metrics from runs, and plot epoch vs each metric.
    Returns the path where the figure was saved.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    return _plot_epoch_metrics_impl(data, Path(out_dir), title, figsize_per_subplot)


def _plot_iter_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """Shared implementation for inner-iteration (lbfgs_iter) metrics."""
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("lbfgs_inner_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with lbfgs_inner_metrics")

    metric_keys: list[str] = []
    for key in ITER_METRIC_KEYS:
        if any(_get_iter_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No inner-iteration metric data found")

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        for r in all_runs:
            arrs = _get_iter_metric_arrays(r)
            it_v, val_v = arrs.get(key, ([], []))
            if not it_v or not val_v:
                continue
            it_v = np.array(it_v)
            val_v = np.array(val_v)
            if r.get("_best_rrmse"):
                ax.plot(it_v, val_v, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (run)" if not labeled_best else None)
                labeled_best = True
            elif r.get("_chosen"):
                ax.plot(it_v, val_v, c="C2", alpha=0.85, linewidth=2, zorder=5,
                        label="Chosen (run)" if not labeled_chosen else None)
                labeled_chosen = True
            else:
                ax.plot(it_v, val_v, c="gray", alpha=0.45, linewidth=0.9, zorder=1,
                        label="All runs" if not labeled_all else None)
                labeled_all = True
        ax.set_xlabel("LBFGS iteration")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Iteration vs metrics — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"iter_metrics_{title}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_iter_metrics_from_data(
    data: dict,
    out_dir: Path,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
    title: str | None = None,
) -> Path:
    """
    Plot iteration vs metrics from in-memory trainer_info data (same structure as saved JSON).
    Produces both iter_metrics_*.png (full y-range) and iter2_metrics_*.png (y scaled to chosen runs).
    Use this right after saving the trainer_analysis JSON. Returns the path to the iter_metrics figure.
    Raises ValueError if no runs have lbfgs_inner_metrics.
    """
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("lbfgs_inner_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with lbfgs_inner_metrics in trainer_info data")
    title = title or data.get("title", "trainer_analysis")
    out_dir = Path(out_dir)
    _plot_iter_metrics_impl(data, out_dir, title, figsize_per_subplot)
    _plot_iter2_metrics_impl(data, out_dir, title, figsize_per_subplot)
    return out_dir / f"iter_metrics_{title}.png"


def plot_iter_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Load a GP_Trainer_Analysis.json, extract lbfgs_inner_metrics from runs, and
    plot iteration vs each metric.
    Returns the path where the figure was saved.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    return _plot_iter_metrics_impl(data, Path(out_dir), title, figsize_per_subplot)


def _plot_iter2_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """
    Same as iter_metrics but y-axis scaled to min/max of chosen runs (and best RRMSE run)
    so convergence is visible instead of squashed by high-loss runs.
    """
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("lbfgs_inner_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with lbfgs_inner_metrics")

    metric_keys: list[str] = []
    for key in ITER_METRIC_KEYS:
        if any(_get_iter_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No inner-iteration metric data found")

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        chosen_vals: list[float] = []
        for r in all_runs:
            arrs = _get_iter_metric_arrays(r)
            it_v, val_v = arrs.get(key, ([], []))
            if not it_v or not val_v:
                continue
            it_v = np.array(it_v)
            val_v = np.array(val_v)
            if r.get("_best_rrmse"):
                ax.plot(it_v, val_v, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (run)" if not labeled_best else None)
                labeled_best = True
                chosen_vals.extend(val_v.tolist())
            elif r.get("_chosen"):
                ax.plot(it_v, val_v, c="C2", alpha=0.85, linewidth=2, zorder=5,
                        label="Chosen (run)" if not labeled_chosen else None)
                labeled_chosen = True
                chosen_vals.extend(val_v.tolist())
            else:
                ax.plot(it_v, val_v, c="gray", alpha=0.45, linewidth=0.9, zorder=1,
                        label="All runs" if not labeled_all else None)
                labeled_all = True
        ax.set_xlabel("LBFGS iteration")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Scale y-axis to chosen (and best RRMSE) runs only
        if chosen_vals:
            arr = np.array(chosen_vals)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                y_min, y_max = float(np.min(arr)), float(np.max(arr))
                pad = max((y_max - y_min) * 0.08, 1e-12)
                ax.set_ylim(y_min - pad, y_max + pad)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Iteration vs metrics (y: chosen runs) — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"iter2_metrics_{title}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_iter2_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Same as plot_iter_metrics but y-axis is scaled to min/max of chosen runs (and best RRMSE run)
    so loss convergence is visible.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    return _plot_iter2_metrics_impl(data, Path(out_dir), title, figsize_per_subplot)


def main() -> None:
    import argparse
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "results_v2.0" / "NLL3"
    parser = argparse.ArgumentParser(
        description="Plot epoch vs training metrics (loss, NLL, NIS, LOO_NLL, KF, Residual_MSE) from trainer_analysis JSON.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=f"Path to a *_GP_Trainer_Analysis.json file or a directory to search (default: {default_root}).",
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=None,
        help="Directory to save the figure (single-file mode only). Default: <json_dir>/plots",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve() if args.path else default_root
    if not path.exists():
        print(f"Path not found: {path}")
        return

    if path.is_dir():
        jsons = find_trainer_analysis_jsons(path)
        if not jsons:
            print(f"No *_GP_Trainer_Analysis.json files found under {path}")
            return
        print(f"Found {len(jsons)} trainer analysis JSON(s) under {path}\n")
        for json_path in sorted(jsons):
            out_dir = json_path.parent / "plots"
            try:
                save_path = plot_epoch_metrics(json_path, out_dir=out_dir)
                print(f"  Saved: {save_path}")
            except ValueError as e:
                print(f"  Skip {json_path.name}: {e}")
        print("\nDone.")
    else:
        if not path.is_file():
            print(f"Not a file: {path}")
            return
        out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        save_path = plot_epoch_metrics(path, out_dir=out_dir)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
