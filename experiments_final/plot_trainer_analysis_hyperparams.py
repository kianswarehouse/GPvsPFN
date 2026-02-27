"""
Plot hyperparameter vs total optimization steps from GP_Trainer_Analysis.json files.

For each trainer_analysis JSON:
- One subplot per hyperparameter: raw_noise, raw_outputscale, then one per dimension for
  raw_lengthscale_0, raw_lengthscale_1, ... and raw_cat_lengthscale_0, raw_cat_lengthscale_1, ...
- X = total steps (0 = start, value = when run ended), Y = hyperparameter value.
- All runs: small gray points; lines from (0, initial) to (num_steps, final).
- Chosen runs (lowest loss per fold): thicker green line; circle = start, star = end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _scalar_from_param(value: Any) -> float | None:
    """Extract a single float from initial_parameters/final_parameters entry."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value)
    if isinstance(value, (list, tuple)):
        arr = [float(x) for x in value if x is not None]
        return float(np.mean(arr)) if arr else None
    return None


def _get_hyperparameter_value(params: dict, key: str) -> float | None:
    """Get scalar value for a hyperparameter key (scalar keys or raw_lengthscale_i, raw_cat_lengthscale_i, mean_weight_i)."""
    if key.startswith("raw_lengthscale_"):
        try:
            idx = int(key.split("_")[-1])
        except ValueError:
            return None
        raw = params.get("raw_lengthscales")
        if not raw or not isinstance(raw, (list, tuple)) or len(raw) <= idx:
            return None
        val = raw[idx]
        return float(val) if val is not None and (isinstance(val, (int, float)) or hasattr(val, "item")) else None
    if key.startswith("raw_cat_lengthscale_"):
        try:
            idx = int(key.split("_")[-1])
        except ValueError:
            return None
        raw = params.get("raw_cat_lengthscales")
        if not raw or not isinstance(raw, (list, tuple)) or len(raw) <= idx:
            return None
        val = raw[idx]
        return float(val) if val is not None and (isinstance(val, (int, float)) or hasattr(val, "item")) else None
    if key.startswith("mean_weight_"):
        try:
            idx = int(key.split("_")[-1])
        except ValueError:
            return None
        mw = params.get("mean_weights")
        if not mw or not isinstance(mw, (list, tuple)) or len(mw) <= idx:
            return None
        val = mw[idx]
        return float(val) if val is not None and (isinstance(val, (int, float)) or hasattr(val, "item")) else None
    return _scalar_from_param(params.get(key))


def _infer_encoder_embedding_keys(all_runs: list[dict]) -> list[str]:
    """Infer encoder_embedding_* keys from initial_parameters (list-of-lists matrices)."""
    seen: set[str] = set()
    for r in all_runs:
        init = r.get("initial_parameters") or {}
        fin = r.get("final_parameters") or {}
        for key in list(init.keys()) + list(fin.keys()):
            if key.startswith("encoder_embedding_") and key not in seen:
                val = init.get(key) or fin.get(key)
                if val is not None and isinstance(val, (list, tuple)) and len(val) > 0:
                    if isinstance(val[0], (list, tuple)):
                        seen.add(key)
                    elif isinstance(val[0], (int, float)):
                        seen.add(key)
    return sorted(seen)


def _embedding_matrix_from_params(params: dict, key: str) -> np.ndarray | None:
    """Extract embedding matrix (n_categories, z_dim) from params[key]; return as float ndarray or None."""
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, (list, tuple)) or len(val) == 0:
        return None
    try:
        arr = np.array(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            return None
        return arr
    except (ValueError, TypeError):
        return None


def _infer_hyperparameter_keys(all_runs: list[dict]) -> list[str]:
    """Infer which raw hyperparameter keys exist; one key per scalar, one per lengthscale dimension.
    Includes raw_power for PowerExponentialKernel; mean_weight_i and mean_bias for LinearMean."""
    has_noise = has_outputscale = has_constant = has_raw_power = False
    max_cont, max_cat = 0, 0
    max_mean_weights = 0
    has_mean_bias = False
    for r in all_runs:
        init = r.get("initial_parameters") or {}
        if "raw_noise" in init:
            has_noise = True
        if "raw_outputscale" in init:
            has_outputscale = True
        if "raw_constant" in init:
            has_constant = True
        if "raw_power" in init:
            has_raw_power = True
        mw = init.get("mean_weights")
        if mw is not None and isinstance(mw, (list, tuple)):
            max_mean_weights = max(max_mean_weights, len(mw))
        if init.get("mean_bias") is not None:
            has_mean_bias = True
        raw_ls = init.get("raw_lengthscales")
        if raw_ls is not None and isinstance(raw_ls, (list, tuple)):
            max_cont = max(max_cont, len(raw_ls))
        raw_cat = init.get("raw_cat_lengthscales")
        if raw_cat is not None and isinstance(raw_cat, (list, tuple)):
            max_cat = max(max_cat, len(raw_cat))
    keys = []
    if has_noise:
        keys.append("raw_noise")
    if has_outputscale:
        keys.append("raw_outputscale")
    if has_constant:
        keys.append("raw_constant")
    if has_raw_power:
        keys.append("raw_power")
    keys.extend(f"mean_weight_{i}" for i in range(max_mean_weights))
    if has_mean_bias:
        keys.append("mean_bias")
    keys.extend(f"raw_lengthscale_{i}" for i in range(max_cont))
    keys.extend(f"raw_cat_lengthscale_{i}" for i in range(max_cat))
    return keys


def load_trainer_analysis(path: Path) -> dict:
    """Load a single GP_Trainer_Analysis.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _trainer_info_fold_iter(data: dict):
    """Yield (fold_idx_0based, fold_entry) for trainer_info as either list or dict (fold_1, fold_2, ...)."""
    trainer_info = data.get("trainer_info")
    if trainer_info is None:
        return
    if isinstance(trainer_info, dict):
        def _fold_sort_key(k):
            if isinstance(k, str) and k.startswith("fold_"):
                try:
                    return int(k.replace("fold_", ""))
                except ValueError:
                    return 0
            return 0
        for fold_key in sorted(trainer_info.keys(), key=_fold_sort_key):
            fold_entry = trainer_info[fold_key]
            try:
                fold_idx = int(fold_key.replace("fold_", "")) - 1 if (isinstance(fold_key, str) and fold_key.startswith("fold_")) else 0
            except ValueError:
                fold_idx = 0
            yield fold_idx, fold_entry
    else:
        for fold_idx, fold_entry in enumerate(trainer_info):
            yield fold_idx, fold_entry


def extract_runs_and_chosen(data: dict) -> tuple[list[dict], list[dict]]:
    """
    From trainer_info (list of folds or dict fold_1, fold_2, ...), return flat list of runs and list of chosen runs (one per fold).
    Chosen = run with minimum loss in that fold.
    Also marks _best_rrmse on the chosen run from the fold that achieved the best RRMSE over all folds.
    """
    all_runs: list[dict] = []
    chosen_list: list[dict] = []
    fold_rrmse: list[float] = []

    for fold_idx, fold_entry in _trainer_info_fold_iter(data):
        runs = fold_entry.get("runs", [])
        if not runs:
            continue
        valid = [r for r in runs if r.get("loss") is not None and np.isfinite(r.get("loss", np.nan))]
        best = min(valid, key=lambda r: r["loss"]) if valid else runs[0]
        chosen_list.append(best)
        metrics = fold_entry.get("metrics") or {}
        rrmse = metrics.get("RRMSE")
        fold_rrmse.append(float(rrmse) if rrmse is not None and np.isfinite(rrmse) else float("inf"))
        for r in runs:
            rec = dict(r)
            rec["_fold_idx"] = fold_idx
            rec["_chosen"] = r is best or (
                r.get("loss") == best.get("loss") and r.get("num_epochs") == best.get("num_epochs")
            )
            rec["_best_rrmse"] = False  # set below
            all_runs.append(rec)

    # Mark the chosen run from the fold with best (lowest) RRMSE
    if fold_rrmse and chosen_list:
        best_fold_idx = int(np.argmin(fold_rrmse))
        for r in all_runs:
            if r.get("_chosen") and r.get("_fold_idx") == best_fold_idx:
                r["_best_rrmse"] = True
                break

    return all_runs, chosen_list


def build_arrays_per_hyperparameter(
    all_runs: list[dict], param_keys: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """
    For each hyperparameter key, build:
    - num_iters (treated as total LBFGS iterations), best_iter, initial_value, final_value,
      is_chosen, is_best_rrmse.
    best_iter = LBFGS iteration when best loss occurred (checkpoint we use);
    num_iters = total number of LBFGS iterations when the run ended.
    If no inner-iteration metrics are available, falls back to epoch-based counts.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in param_keys:
        iters, best_iters, init_vals, final_vals, is_chosen, is_best_rrmse = [], [], [], [], [], []
        for r in all_runs:
            init = r.get("initial_parameters") or {}
            fin = r.get("final_parameters") or {}
            vi = _get_hyperparameter_value(init, key)
            vf = _get_hyperparameter_value(fin, key)
            if vi is None and vf is None:
                continue
            # Prefer LBFGS inner-iteration metrics if available; otherwise fall back to epochs.
            inner = r.get("lbfgs_inner_metrics") or []
            if inner:
                # Total iterations = max lbfgs_iter; best_iter = lbfgs_iter at minimum loss.
                valid_inner = [m for m in inner if m.get("lbfgs_iter") is not None and np.isfinite(m.get("loss", np.nan))]
                if valid_inner:
                    num_iters = float(max(m["lbfgs_iter"] for m in valid_inner))
                    best_entry = min(valid_inner, key=lambda m: m.get("loss", np.inf))
                    best_iter = float(best_entry.get("lbfgs_iter", num_iters))
                else:
                    num_iters = float(len(inner))
                    best_iter = np.nan
            else:
                # Fallback: epochs
                num_iters = float(r.get("num_epochs", 0))
                be = r.get("best_epoch") if r.get("best_epoch") is not None else fin.get("best_epoch")
                if be is not None and isinstance(be, (int, float)) and np.isfinite(be):
                    best_iter = float(be)
                else:
                    best_iter = np.nan

            iters.append(num_iters)
            best_iters.append(best_iter)
            init_vals.append(vi if vi is not None else np.nan)
            final_vals.append(vf if vf is not None else np.nan)
            is_chosen.append(r.get("_chosen", False))
            is_best_rrmse.append(r.get("_best_rrmse", False))
        out[key] = {
            "num_iters": np.array(iters, dtype=float),
            "best_iter": np.array(best_iters, dtype=float),
            "initial_value": np.array(init_vals, dtype=float),
            "final_value": np.array(final_vals, dtype=float),
            "is_chosen": np.array(is_chosen, dtype=bool),
            "is_best_rrmse": np.array(is_best_rrmse, dtype=bool),
        }
    return out


def plot_hyperparameter_epochs(
    arrays_per_param: dict[str, dict[str, np.ndarray]],
    title: str,
    save_path: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 4),
) -> None:
    """
    One subplot per hyperparameter. X = total steps (0 = start, num_steps = end).
    Y = hyperparameter value.
    Each run shows 2 or 3 points: start (step 0), best step (if distinct), and final
    (total steps).
    Same for gray, green (chosen), and red (best RRMSE) runs.
    """
    def _run_line_points(epochs_i: float, best_epoch_i: float, init_i: float, final_i: float):
        """Return (x_list, y_list) for polyline: 2 points if best==start or best==end, else 3 points."""
        x0, x_end = 0.0, float(epochs_i)
        be = float(best_epoch_i) if np.isfinite(best_epoch_i) else None
        if be is not None and 0 < be < x_end:
            return [x0, be, x_end], [init_i, final_i, final_i]
        return [x0, x_end], [init_i, final_i]

    def _run_marker_points(epochs_i: float, best_epoch_i: float, init_i: float, final_i: float):
        """Return (x_array, y_array) for scatter: 2 or 3 points (start, best if distinct, end)."""
        x0, x_end = 0.0, float(epochs_i)
        be = float(best_epoch_i) if np.isfinite(best_epoch_i) else None
        if be is not None and 0 < be < x_end:
            return np.array([x0, be, x_end]), np.array([init_i, final_i, final_i])
        return np.array([x0, x_end]), np.array([init_i, final_i])

    param_keys = [k for k in arrays_per_param if np.any(np.isfinite(arrays_per_param[k]["initial_value"]))]
    n = len(param_keys)
    if n == 0:
        return
    ncols = 3 if n > 6 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    for idx, key in enumerate(param_keys):
        ax = axes[idx]
        d = arrays_per_param[key]
        epochs = d["num_iters"]
        best_epochs = d.get("best_iter", np.full_like(epochs, np.nan))
        init_v = d["initial_value"]
        final_v = d["final_value"]
        chosen = d["is_chosen"]
        best_rrmse = d.get("is_best_rrmse", np.zeros(len(epochs), dtype=bool))

        # Lines and points: 2 or 3 points per run (start, best epoch if distinct, final)
        labeled_chosen_line = False
        labeled_all_line = False
        labeled_best_rrmse_line = False
        all_x, all_y = [], []
        chosen_x, chosen_y = [], []
        best_x, best_y = [], []

        for i in range(len(epochs)):
            if not np.isfinite(init_v[i]) and not np.isfinite(final_v[i]):
                continue
            init_i = init_v[i] if np.isfinite(init_v[i]) else final_v[i]
            final_i = final_v[i] if np.isfinite(final_v[i]) else init_v[i]
            x_pts, y_pts = _run_line_points(epochs[i], best_epochs[i], init_i, final_i)
            mx, my = _run_marker_points(epochs[i], best_epochs[i], init_i, final_i)
            if best_rrmse[i]:
                ax.plot(x_pts, y_pts, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (line)" if not labeled_best_rrmse_line else None)
                labeled_best_rrmse_line = True
                best_x.append(mx)
                best_y.append(my)
            elif chosen[i]:
                ax.plot(x_pts, y_pts, c="C2", alpha=0.8, linewidth=2, zorder=2,
                        label="Chosen (line)" if not labeled_chosen_line else None)
                labeled_chosen_line = True
                chosen_x.append(mx)
                chosen_y.append(my)
            else:
                ax.plot(x_pts, y_pts, c="gray", alpha=0.4, linewidth=0.8, zorder=1,
                        label="All runs (line)" if not labeled_all_line else None)
                labeled_all_line = True
                all_x.append(mx)
                all_y.append(my)

        # Scatter: 2 or 3 points per run — start (circle), best epoch if distinct, end (star)
        if all_x:
            ax.scatter(
                np.concatenate(all_x), np.concatenate(all_y),
                s=20, c="gray", alpha=0.6, label="All runs", zorder=3,
            )
        if chosen_x:
            starts_x = np.concatenate([a[0:1] for a in chosen_x])
            starts_y = np.concatenate([a[0:1] for a in chosen_y])
            ends_x = np.concatenate([a[1:] for a in chosen_x])
            ends_y = np.concatenate([a[1:] for a in chosen_y])
            ax.scatter(starts_x, starts_y, s=120, c="C2", alpha=0.9, marker="o",
                       edgecolors="darkgreen", linewidths=1.5, label="Chosen (start)", zorder=4)
            ax.scatter(ends_x, ends_y, s=180, c="C0", alpha=0.9, marker="*",
                       edgecolors="darkblue", linewidths=1, label="Chosen (end)", zorder=5)
        if best_x:
            starts_x = np.concatenate([a[0:1] for a in best_x])
            starts_y = np.concatenate([a[0:1] for a in best_y])
            ends_x = np.concatenate([a[1:] for a in best_x])
            ends_y = np.concatenate([a[1:] for a in best_y])
            ax.scatter(starts_x, starts_y, s=150, c="red", alpha=0.95, marker="o",
                       edgecolors="darkred", linewidths=2, label="Best RRMSE (start)", zorder=7)
            ax.scatter(ends_x, ends_y, s=220, c="red", alpha=0.95, marker="*",
                       edgecolors="darkred", linewidths=1.5, label="Best RRMSE (end)", zorder=8)

        ax.set_xlabel("Iterations")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.set_title(key.replace("_", " ").title())
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-0.2)  # small gap before step 0 (max 0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # steps are integers only

        # Y-axis: bounds from FINAL CHOSEN RUNS only (one per fold, lowest loss; initial + final values) + padding
        final_chosen_init = init_v[chosen]
        final_chosen_final = final_v[chosen]
        final_chosen_vals = np.concatenate([final_chosen_init, final_chosen_final])
        chosen_finite = final_chosen_vals[np.isfinite(final_chosen_vals)]
        if len(chosen_finite) > 0:
            y_min, y_max = np.min(chosen_finite), np.max(chosen_finite)
            pad = max((y_max - y_min) * 0.1, 0.2)
            ax.set_ylim(y_min - pad, y_max + pad)
        else:
            # Fallback: use all runs only if no chosen runs have finite values
            all_vals = np.concatenate([init_v, final_v])
            finite = all_vals[np.isfinite(all_vals)]
            if len(finite) > 0:
                y_min, y_max = np.min(finite), np.max(finite)
                pad = max((y_max - y_min) * 0.1, 0.2)
                ax.set_ylim(y_min - pad, y_max + pad)

    for j in range(len(param_keys), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _collect_embedding_by_category(
    runs: list[dict], key: str
) -> tuple[bool, int, dict[int, tuple[list[float], list[float], list[float], list[float]]]]:
    """
    For one encoder key, collect per-category (variable) initial/final (x,y) across given runs.
    Returns (use_2d, n_categories, {k: (x_i, y_i, x_f, y_f)}).
    n_categories is read from the embedding matrix shape (number of rows).
    """
    use_2d = True
    n_cat_seen: int | None = None
    by_cat: dict[int, tuple[list[float], list[float], list[float], list[float]]] = {}

    for r in runs:
        init_params = r.get("initial_parameters") or {}
        final_params = r.get("final_parameters") or {}
        M_init = _embedding_matrix_from_params(init_params, key)
        M_final = _embedding_matrix_from_params(final_params, key)
        if M_init is None and M_final is None:
            continue
        if M_init is None:
            M_init = M_final
        if M_final is None:
            M_final = M_init
        n_cat = min(M_init.shape[0], M_final.shape[0])
        if n_cat_seen is None:
            n_cat_seen = n_cat
            use_2d = bool(M_init.shape[1] >= 2 and M_final.shape[1] >= 2)
            for k in range(n_cat):
                by_cat[k] = ([], [], [], [])
        else:
            for k in range(n_cat):
                if k not in by_cat:
                    by_cat[k] = ([], [], [], [])

        for k in range(n_cat):
            x_i, y_i, x_f, y_f = by_cat[k]
            if use_2d:
                x_i.append(float(M_init[k, 0]))
                y_i.append(float(M_init[k, 1]))
                x_f.append(float(M_final[k, 0]))
                y_f.append(float(M_final[k, 1]))
            else:
                x_i.append(float(k))
                y_i.append(float(M_init[k, 0]) if M_init.shape[1] > 0 else 0.0)
                x_f.append(float(k))
                y_f.append(float(M_final[k, 0]) if M_final.shape[1] > 0 else 0.0)

    return use_2d, n_cat_seen or 0, by_cat


# Very distinct colors per variable (red, green, orange, purple, teal, brown, magenta, gold, …)
# Avoid blue/light-blue pairs so each variable is easy to tell apart.
_VAR_COLORS = [
    "#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085",
    "#d35400", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71",
    "#f39c12", "#c0392b", "#2980b9", "#d35400",
]


def _lighten(rgb_hex: str, factor: float = 0.65) -> tuple[float, float, float]:
    """Lighten a hex color by blending with white (init marker)."""
    r = int(rgb_hex[1:3], 16) / 255.0
    g = int(rgb_hex[3:5], 16) / 255.0
    b = int(rgb_hex[5:7], 16) / 255.0
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b)


def _draw_embedding_subplot(
    ax,
    key: str,
    use_2d: bool,
    by_cat: dict[int, tuple[list[float], list[float], list[float], list[float]]],
    zoom: bool = False,
    percentile: float = 2.0,
) -> None:
    """Draw one encoder subplot: per-variable init/final, lines init→final, distinct colors per variable."""
    if not by_cat:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(key.replace("_", " ").title())
        return

    xlabel, ylabel = ("dim 0", "dim 1") if use_2d else ("category", "value")
    all_x: list[float] = []
    all_y: list[float] = []

    for k in sorted(by_cat.keys()):
        x_i, y_i, x_f, y_f = by_cat[k]
        if not x_i and not x_f:
            continue
        base_hex = _VAR_COLORS[k % len(_VAR_COLORS)]
        base_rgb = tuple(int(base_hex[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        light_rgb = _lighten(base_hex, factor=0.65)

        # Line from each init to final (one per chosen run)
        for j in range(len(x_i)):
            ax.plot(
                [x_i[j], x_f[j]], [y_i[j], y_f[j]],
                color=base_rgb, alpha=0.75, linewidth=1.5, zorder=1,
            )
        ax.scatter(
            x_i, y_i,
            c=[light_rgb], s=44, marker="o", edgecolors=base_rgb, linewidths=1.2,
            label=f"Var {k} init", zorder=2,
        )
        ax.scatter(
            x_f, y_f,
            c=[base_rgb], s=90, marker="*", edgecolors="k", linewidths=0.5,
            label=f"Var {k} final", zorder=3,
        )
        all_x.extend(x_i)
        all_x.extend(x_f)
        all_y.extend(y_i)
        all_y.extend(y_f)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(key.replace("_", " ").title())
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    if zoom and all_x and all_y:
        all_x_arr = np.array(all_x)
        all_y_arr = np.array(all_y)
        x_lo, x_hi = np.percentile(all_x_arr, [percentile, 100 - percentile])
        y_lo, y_hi = np.percentile(all_y_arr, [percentile, 100 - percentile])
        pad_x = max((x_hi - x_lo) * 0.05, 1e-6)
        pad_y = max((y_hi - y_lo) * 0.05, 1e-6)
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)


def plot_embedding_2d(
    data: dict,
    title: str,
    out_dir: Path,
    save_path: Path | None = None,
) -> None:
    """
    For each encoder_embedding_* key, plot initial vs final embedding positions in 2D
    for all runs. One subplot per encoder. Number of variables (categories) per encoder
    is read from the embedding matrix shape (number of rows). Each variable is drawn
    with a distinct color; initial = circles, final = stars.
    Saves two figures: full range and zoomed (percentile-based limits to center on bulk, hide outliers).
    """
    all_runs, chosen_list = extract_runs_and_chosen(data)
    if not all_runs:
        return
    chosen_runs = [r for r in all_runs if r.get("_chosen")]
    if not chosen_runs:
        chosen_runs = chosen_list
    if not chosen_runs:
        return
    encoder_keys = _infer_encoder_embedding_keys(chosen_runs)
    if not encoder_keys:
        return

    # Build per-encoder, per-category data from chosen runs only
    enc_data: list[tuple[bool, int, dict]] = []
    for key in encoder_keys:
        use_2d, n_cat, by_cat = _collect_embedding_by_category(chosen_runs, key)
        enc_data.append((use_2d, n_cat, by_cat))

    n_enc = len(encoder_keys)
    ncols = min(n_enc, 3)
    nrows = (n_enc + ncols - 1) // ncols
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chosen = len(chosen_runs)

    for zoom, suffix in [(False, ""), (True, "_zoomed")]:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            squeeze=False,
        )
        axes = axes.flat
        for idx, key in enumerate(encoder_keys):
            ax = axes[idx]
            use_2d, _, by_cat = enc_data[idx]
            _draw_embedding_subplot(ax, key, use_2d, by_cat, zoom=zoom)
        for j in range(len(encoder_keys), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(
            f"{title}\nEmbedding: initial and final for chosen runs (n={n_chosen}) — init (○) → final (★)" + (
                " — zoomed" if zoom else ""
            ),
            fontsize=10,
        )
        fig.tight_layout()
        path = out_dir / f"trainer_embeddings_{title}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if save_path is not None and not zoom:
            pass  # caller’s save_path only used for non-zoomed if we wanted a single custom path


def plot_trainer_analysis_from_data(data: dict, out_dir: Path) -> None:
    """
    Generate hyperparameter vs total-step plot from in-memory trainer_info data
    (same structure as saved JSON).
    Use this right after saving the trainer_analysis JSON so plots are created automatically.
    If encoder_embedding_* keys exist, also generates 2D embedding plots (initial vs final).
    """
    all_runs, _ = extract_runs_and_chosen(data)
    if not all_runs:
        return
    out_dir = Path(out_dir)
    title = data.get("title", "trainer_analysis")

    param_keys = _infer_hyperparameter_keys(all_runs)
    if param_keys:
        arrays = build_arrays_per_hyperparameter(all_runs, param_keys)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"trainer_hyperparams_{title}.png"
        plot_hyperparameter_epochs(arrays, title=title, save_path=save_path)
        print(f"Trainer analysis plot saved to: {save_path}")

    encoder_keys = _infer_encoder_embedding_keys(all_runs)
    if encoder_keys:
        plot_embedding_2d(data, title=title, out_dir=out_dir)
        print(f"Trainer embedding plots saved to: {out_dir / ('trainer_embeddings_' + title + '.png')} and ..._zoomed.png")


def process_file(json_path: Path, out_dir: Path | None = None) -> None:
    """Load one trainer_analysis JSON, build arrays, plot and optionally save."""
    data = load_trainer_analysis(json_path)
    all_runs, chosen_list = extract_runs_and_chosen(data)
    if not all_runs:
        return
    title = data.get("title", json_path.stem)

    param_keys = _infer_hyperparameter_keys(all_runs)
    if param_keys:
        arrays = build_arrays_per_hyperparameter(all_runs, param_keys)
        save_path = out_dir / f"trainer_hyperparams_{title}.png" if out_dir is not None else None
        plot_hyperparameter_epochs(arrays, title=title, save_path=save_path)

    encoder_keys = _infer_encoder_embedding_keys(all_runs)
    if encoder_keys and out_dir is not None:
        plot_embedding_2d(data, title=title, out_dir=out_dir)


def find_trainer_analysis_jsons(root: Path) -> list[Path]:
    """Find all *_GP_Trainer_Analysis.json under root, in trainer_analysis dirs."""
    return list(root.rglob("trainer_analysis/*_GP_Trainer_Analysis.json"))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Plot hyperparameter vs epochs from trainer_analysis JSONs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="Paths to trainer_analysis dirs or JSON files. If empty, uses experiments_final/results_final.",
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Default: next to each JSON in a 'plots' sibling.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Discover and process all trainer_analysis JSONs under each given path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.paths:
        to_process: list[Path] = []
        for p in args.paths:
            path = Path(p).resolve()
            if path.is_file() and path.suffix == ".json" and "Trainer_Analysis" in path.name:
                to_process.append(path)
            elif path.is_dir():
                if args.all:
                    to_process.extend(path.rglob("trainer_analysis/*_GP_Trainer_Analysis.json"))
                else:
                    trainer_dir = path / "trainer_analysis"
                    if trainer_dir.is_dir():
                        to_process.extend(trainer_dir.glob("*_GP_Trainer_Analysis.json"))
                    # Also allow JSONs directly in the given directory (e.g. results/plotsfortyler)
                    to_process.extend(path.glob("*_GP_Trainer_Analysis.json"))
            else:
                to_process.append(path)
    else:
        results = root / "results_final"
        to_process = list(results.rglob("trainer_analysis/*_GP_Trainer_Analysis.json")) if results.is_dir() else []

    for json_path in to_process:
        if not json_path.is_file():
            continue
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = json_path.parent / "plots"
        print(f"Processing: {json_path}")
        process_file(json_path, out_dir=out_dir)


# ---------------------------------------------------------------------------
# "Just run this file": set a default path to process when no CLI args given.
# Leave empty to auto-discover all trainer_analysis JSONs under results_final.
# Example (one file):
#   DEFAULT_PATHS = ["results_final/1_12/buckling/trainer_analysis/gpVpfn_buckling_SF_2_80Dn_16runs_noiseTest0.005_noiseTrain0.005_GP_Trainer_Analysis.json"]
# Example (one folder):
#   DEFAULT_PATHS = ["results_final/1_12/buckling/trainer_analysis"]
# ---------------------------------------------------------------------------
DEFAULT_PATHS: list[str] = [
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/1_12/rastrigin",
                            "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/1_12/rosenbrock",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results/plotsfortyler",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/custom_GPvsPFN/buckling",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/custom_GPvsPFN/buckling/buckling",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/1_12/griewank",
                            ]


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent
    # If no CLI args but DEFAULT_PATHS set, use those (paths relative to script dir)
    if not sys.argv[1:] and DEFAULT_PATHS:
        sys.argv[1:] = [str(root / p) for p in DEFAULT_PATHS]
    main()
