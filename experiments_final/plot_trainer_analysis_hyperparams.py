"""
Plot hyperparameter vs total optimization steps from GP_Trainer_Analysis.json files.

For each trainer_analysis JSON:
- One subplot per hyperparameter: raw_noise, raw_outputscale, then one per dimension for
  raw_lengthscale_0, raw_lengthscale_1, ... and raw_cat_lengthscale_0, raw_cat_lengthscale_1, ...
- X = total steps (0 = start, value = when run ended), Y = hyperparameter value.
- All runs: small gray points; lines from (0, initial) to (num_steps, final).
- Chosen init (lowest loss per run): thicker green line; circle = start, star = end.
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


def _params_flat_view(params: dict) -> dict:
    """Return a flat view of params for plotting. Supports nested (covar_module, mean_module, likelihood) and flat formats."""
    if not params:
        return {}
    if "covar_module" in params or "likelihood" in params:
        flat: dict[str, Any] = {}
        lik = params.get("likelihood") or {}
        flat["raw_noise"] = lik.get("raw_noise")
        flat["noise"] = lik.get("noise")
        cov = params.get("covar_module") or {}
        flat["raw_outputscale"] = cov.get("raw_outputscale")
        flat["outputscale"] = cov.get("outputscale")
        base = cov.get("base_kernel") or cov
        base_type = base.get("type")

        # Collect continuous lengthscales (may be spread across sub-kernels for Additive/Product kernels)
        raw_ls_all: list[float] = []
        ls_all: list[float] = []
        raw_period_all: list[float] = []
        period_all: list[float] = []

        if base_type in ("AdditiveKernel", "ProductKernel") and isinstance(base.get("kernels"), list):
            for sub in base["kernels"]:
                if not isinstance(sub, dict):
                    continue
                sub_raw_ls = sub.get("raw_lengthscale")
                if isinstance(sub_raw_ls, (list, tuple)):
                    raw_ls_all.extend(float(v) for v in sub_raw_ls if v is not None)
                elif sub_raw_ls is not None:
                    raw_ls_all.append(float(sub_raw_ls))

                sub_ls = sub.get("lengthscale")
                if isinstance(sub_ls, (list, tuple)):
                    ls_all.extend(float(v) for v in sub_ls if v is not None)
                elif sub_ls is not None:
                    ls_all.append(float(sub_ls))

                sub_raw_period = sub.get("raw_period")
                if isinstance(sub_raw_period, (list, tuple)):
                    raw_period_all.extend(float(v) for v in sub_raw_period if v is not None)
                elif sub_raw_period is not None:
                    raw_period_all.append(float(sub_raw_period))

                sub_period = sub.get("period")
                if isinstance(sub_period, (list, tuple)):
                    period_all.extend(float(v) for v in sub_period if v is not None)
                elif sub_period is not None:
                    period_all.append(float(sub_period))
        else:
            raw_ls = base.get("raw_lengthscale")
            if isinstance(raw_ls, (list, tuple)):
                raw_ls_all = [float(v) for v in raw_ls if v is not None]
            elif raw_ls is not None:
                raw_ls_all = [float(raw_ls)]

            ls = base.get("lengthscale")
            if isinstance(ls, (list, tuple)):
                ls_all = [float(v) for v in ls if v is not None]
            elif ls is not None:
                ls_all = [float(ls)]

            raw_period = base.get("raw_period")
            if isinstance(raw_period, (list, tuple)):
                raw_period_all = [float(v) for v in raw_period if v is not None]
            elif raw_period is not None:
                raw_period_all = [float(raw_period)]

            period = base.get("period")
            if isinstance(period, (list, tuple)):
                period_all = [float(v) for v in period if v is not None]
            elif period is not None:
                period_all = [float(period)]

        flat["raw_lengthscales"] = raw_ls_all or None
        flat["lengthscales"] = ls_all or None
        flat["raw_periods"] = raw_period_all or None
        flat["periods"] = period_all or None
        # PowerExponentialKernel: expose raw_power/power at top level so they can be plotted
        if "raw_power" in base:
            flat["raw_power"] = base.get("raw_power")
        if "power" in base:
            flat["power"] = base.get("power")
        mean = params.get("mean_module") or {}
        flat["raw_constant"] = mean.get("raw_constant")
        flat["constant"] = mean.get("constant")
        flat["raw_cat_lengthscales"] = params.get("raw_cat_lengthscales")
        flat["raw_source_lengthscales"] = params.get("raw_source_lengthscales")
        for k, v in params.items():
            if k not in ("covar_module", "mean_module", "likelihood") and v is not None:
                flat[k] = v
        return flat
    return dict(params)


def _get_hyperparameter_value(params: dict, key: str) -> float | None:
    """Get scalar value for a hyperparameter key (scalar keys or raw_lengthscale_i, lengthscale_i, raw_cat_lengthscale_i, mean_weight_i)."""
    params = _params_flat_view(params)
    # Actual (transformed) keys used in the model
    if key == "noise":
        return _scalar_from_param(params.get("noise"))
    if key == "outputscale":
        return _scalar_from_param(params.get("outputscale"))
    if key == "constant":
        return _scalar_from_param(params.get("constant"))
    if key == "power":
        return _scalar_from_param(params.get("power"))
    if key.startswith("lengthscale_") and not key.startswith("lengthscale_s"):  # lengthscale_0, lengthscale_1, ...
        try:
            idx = int(key.split("_")[-1])
        except ValueError:
            return None
        ls = params.get("lengthscales")
        if not ls or not isinstance(ls, (list, tuple)) or len(ls) <= idx:
            return None
        val = ls[idx]
        return float(val) if val is not None and (isinstance(val, (int, float)) or hasattr(val, "item")) else None
    # Raw keys
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
    if key.startswith("raw_period_"):
        try:
            idx = int(key.split("_")[-1])
        except ValueError:
            return None
        raw_p = params.get("raw_periods")
        if not raw_p or not isinstance(raw_p, (list, tuple)) or len(raw_p) <= idx:
            return None
        val = raw_p[idx]
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
    max_cont, max_cat, max_periods = 0, 0, 0
    max_mean_weights = 0
    has_mean_bias = False
    for r in all_runs:
        init = _params_flat_view(r.get("initial_parameters") or {})
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
        raw_p = init.get("raw_periods")
        if raw_p is not None and isinstance(raw_p, (list, tuple)):
            max_periods = max(max_periods, len(raw_p))
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
    keys.extend(f"raw_period_{i}" for i in range(max_periods))
    return keys


def _infer_hyperparameter_keys_actual(all_runs: list[dict]) -> list[str]:
    """Infer which actual (transformed) hyperparameter keys exist for plotting.
    Same structure as raw but keys: noise, outputscale, constant, power, lengthscale_0, ...
    """
    has_noise = has_outputscale = has_constant = has_power = False
    max_cont, max_cat, max_periods = 0, 0, 0
    max_mean_weights = 0
    has_mean_bias = False
    for r in all_runs:
        init = _params_flat_view(r.get("initial_parameters") or {})
        if init.get("noise") is not None:
            has_noise = True
        if init.get("outputscale") is not None:
            has_outputscale = True
        if init.get("constant") is not None:
            has_constant = True
        if init.get("power") is not None:
            has_power = True
        mw = init.get("mean_weights")
        if mw is not None and isinstance(mw, (list, tuple)):
            max_mean_weights = max(max_mean_weights, len(mw))
        if init.get("mean_bias") is not None:
            has_mean_bias = True
        ls = init.get("lengthscales")
        if ls is not None and isinstance(ls, (list, tuple)):
            max_cont = max(max_cont, len(ls))
        ps = init.get("periods")
        if ps is not None and isinstance(ps, (list, tuple)):
            max_periods = max(max_periods, len(ps))
    keys = []
    if has_noise:
        keys.append("noise")
    if has_outputscale:
        keys.append("outputscale")
    if has_constant:
        keys.append("constant")
    if has_power:
        keys.append("power")
    keys.extend(f"mean_weight_{i}" for i in range(max_mean_weights))
    if has_mean_bias:
        keys.append("mean_bias")
    keys.extend(f"lengthscale_{i}" for i in range(max_cont))
    keys.extend(f"period_{i}" for i in range(max_periods))
    return keys


def load_trainer_analysis(path: Path) -> dict:
    """Load a single GP_Trainer_Analysis.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _trainer_info_run_iter(data: dict):
    """Yield (run_idx_0based, run_entry, group_key) for trainer_info. group_key is the dict key (run_1, run_2, ...) for naming."""
    trainer_info = data.get("trainer_info")
    if trainer_info is None:
        return
    if isinstance(trainer_info, dict):
        def _group_sort_key(k):
            if isinstance(k, str):
                if k.startswith("run_"):
                    try:
                        return int(k.replace("run_", ""))
                    except ValueError:
                        return 0
                if k.startswith("fold_"):
                    try:
                        return int(k.replace("fold_", ""))
                    except ValueError:
                        return 0
            return 0
        for group_key in sorted(trainer_info.keys(), key=_group_sort_key):
            run_entry = trainer_info[group_key]
            try:
                if isinstance(group_key, str) and group_key.startswith("run_"):
                    group_idx = int(group_key.replace("run_", "")) - 1
                elif isinstance(group_key, str) and group_key.startswith("fold_"):
                    group_idx = int(group_key.replace("fold_", "")) - 1
                else:
                    group_idx = 0
            except ValueError:
                group_idx = 0
            yield group_idx, run_entry, group_key
    else:
        for group_idx, run_entry in enumerate(trainer_info):
            group_key = f"run_{group_idx + 1}"
            yield group_idx, run_entry, group_key


def extract_runs_and_chosen(data: dict) -> tuple[list[dict], list[dict]]:
    """
    From trainer_info (dict run_1, run_2, ...), return flat list of inits and list of chosen inits (one per run).
    Chosen = init with minimum loss in that run. Sets _group_key (e.g. run_1) and _run_idx on each init.
    Also marks _best_rrmse on the chosen init from the run that achieved the best RRMSE.
    """
    all_inits: list[dict] = []
    chosen_list: list[dict] = []
    run_rrmse: list[float] = []

    for run_idx, run_entry, group_key in _trainer_info_run_iter(data):
        inits = run_entry.get("inits", run_entry.get("runs", []))  # inits = list of inits for this run
        if not inits:
            continue
        valid = [r for r in inits if r.get("loss") is not None and np.isfinite(r.get("loss", np.nan))]
        best = min(valid, key=lambda r: r["loss"]) if valid else inits[0]
        chosen_list.append(best)
        metrics = run_entry.get("metrics") or {}
        rrmse = metrics.get("RRMSE")
        run_rrmse.append(float(rrmse) if rrmse is not None and np.isfinite(rrmse) else float("inf"))
        for r in inits:
            rec = dict(r)
            rec["_run_idx"] = run_idx
            rec["_group_key"] = group_key  # e.g. run_1 for file naming
            rec["_chosen"] = r is best or (
                r.get("loss") == best.get("loss") and r.get("num_epochs") == best.get("num_epochs")
            )
            rec["_best_rrmse"] = False  # set below
            all_inits.append(rec)

    # Mark the chosen init from the run with best (lowest) RRMSE
    if run_rrmse and chosen_list:
        best_run_idx = int(np.argmin(run_rrmse))
        for r in all_inits:
            if r.get("_chosen") and r.get("_run_idx") == best_run_idx:
                r["_best_rrmse"] = True
                break

    return all_inits, chosen_list


def _get_per_iteration_records(run: dict) -> list[dict] | None:
    """
    Get per-iteration parameter records for a run from iteration_parameters.records
    or lbfgs_parameters. Each record has 'iteration' and 'parameters' (nested dict).
    Returns None if no per-iteration data.
    """
    ip = run.get("iteration_parameters")
    if ip and isinstance(ip, dict):
        recs = ip.get("records")
        if recs and isinstance(recs, list):
            return recs
    lp = run.get("lbfgs_parameters")
    if lp and isinstance(lp, list):
        return lp
    return None


def build_arrays_per_hyperparameter(
    all_runs: list[dict], param_keys: list[str]
) -> dict[str, dict[str, Any]]:
    """
    For each hyperparameter key, build:
    - num_iters (treated as total LBFGS iterations), best_iter, initial_value, final_value,
      is_chosen, is_best_rrmse.
    - iter_series: list of length n_runs; each element is None or (x_array, y_array) for
      iteration indices and param values when run has iteration_parameters or lbfgs_parameters.
    best_iter = LBFGS iteration when best loss occurred (checkpoint we use);
    num_iters = total number of LBFGS iterations when the run ended.
    If no inner-iteration metrics are available, falls back to epoch-based counts.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in param_keys:
        iters, best_iters, init_vals, final_vals, is_chosen, is_best_rrmse = [], [], [], [], [], []
        iter_series_list: list[tuple[np.ndarray, np.ndarray] | None] = []
        for r in all_runs:
            init = r.get("initial_parameters") or {}
            fin = r.get("final_parameters") or {}
            vi = _get_hyperparameter_value(init, key)
            vf = _get_hyperparameter_value(fin, key)
            if vi is None and vf is None:
                continue
            # Per-iteration series when available (iteration_parameters.records or lbfgs_parameters)
            recs = _get_per_iteration_records(r)
            if recs:
                x_vals, y_vals = [], []
                for rec in recs:
                    it = rec.get("iteration")
                    params = rec.get("parameters") or {}
                    val = _get_hyperparameter_value(params, key)
                    if it is not None and val is not None and np.isfinite(val):
                        x_vals.append(float(it))
                        y_vals.append(float(val))
                if x_vals and y_vals:
                    iter_series_list.append((np.array(x_vals), np.array(y_vals)))
                else:
                    iter_series_list.append(None)
            else:
                iter_series_list.append(None)
            # Prefer record's best_iter (from trainer when LBFGS); else from inner metrics or epochs.
            r_best_iter = r.get("best_iter")
            if r_best_iter is not None and isinstance(r_best_iter, (int, float)) and np.isfinite(r_best_iter):
                best_iter = float(r_best_iter)
            else:
                best_iter = np.nan
            inner = r.get("lbfgs_inner_metrics") or []
            if inner:
                # Total iterations = max lbfgs_iter.
                valid_inner = [m for m in inner if m.get("lbfgs_iter") is not None and np.isfinite(m.get("loss", np.nan))]
                if valid_inner:
                    num_iters = float(max(m["lbfgs_iter"] for m in valid_inner))
                    if np.isnan(best_iter):
                        best_entry = min(valid_inner, key=lambda m: m.get("loss", np.inf))
                        best_iter = float(best_entry.get("lbfgs_iter", num_iters))
                else:
                    num_iters = float(len(inner))
            else:
                # Fallback: epochs
                num_iters = float(r.get("num_epochs", 0))
                if np.isnan(best_iter):
                    be = r.get("best_epoch") if r.get("best_epoch") is not None else fin.get("best_epoch")
                    if be is not None and isinstance(be, (int, float)) and np.isfinite(be):
                        best_iter = float(be)

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
            "iter_series": iter_series_list,
        }
    return out


def _param_display_label(key: str, is_raw_keys: bool) -> str:
    """Return axis/title label for a hyperparameter key."""
    if (not is_raw_keys) and key == "noise_log":
        return "Noise (log scale)"
    if is_raw_keys and key.startswith("raw_"):
        return "Raw " + key.replace("raw_", "", 1).replace("_", " ").title()
    return key.replace("_", " ").title()


def plot_hyperparameter_epochs(
    arrays_per_param: dict[str, dict[str, Any]],
    title: str,
    save_path: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 4),
    is_raw_keys: bool = True,
    log_y: bool = False,
    only_keys: list[str] | None = None,
) -> None:
    """
    One subplot per hyperparameter. X = iterations (0 = start, num_iters = end).
    Y = hyperparameter value.
    When iteration_parameters or lbfgs_parameters are present (per-iteration data),
    the actual trajectory is plotted; otherwise 2 points: start and end.
    Same for gray, green (chosen), and red (best RRMSE) runs.
    is_raw_keys: if True, keys are raw_* and displayed as "Raw Noise", etc.; if False, displayed as "Noise", "Power", etc.
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

    base_keys = [k for k in arrays_per_param if np.any(np.isfinite(arrays_per_param[k]["initial_value"]))]
    if only_keys is not None:
        # Respect the explicit ordering in only_keys, but drop keys that
        # are not present or have no finite values.
        param_keys = [k for k in only_keys if k in base_keys]
    else:
        param_keys = base_keys
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
        iter_series_list = d.get("iter_series") or [None] * len(epochs)

        # Lines: use per-iteration series when available, else 2-point (start, end)
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
            series = iter_series_list[i] if i < len(iter_series_list) else None
            if series is not None and len(series[0]) > 0 and len(series[1]) > 0:
                x_pts, y_pts = series[0], series[1]
                mx = np.array([x_pts[0], x_pts[-1]])
                my = np.array([y_pts[0], y_pts[-1]])
            else:
                x_pts, y_pts = _run_line_points(epochs[i], best_epochs[i], init_i, final_i)
                mx, my = _run_marker_points(epochs[i], best_epochs[i], init_i, final_i)
            if best_rrmse[i]:
                ax.plot(x_pts, y_pts, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (line)" if not labeled_best_rrmse_line else None)
                labeled_best_rrmse_line = True
                best_x.append((mx, my))
            elif chosen[i]:
                ax.plot(x_pts, y_pts, c="C2", alpha=0.8, linewidth=2, zorder=2,
                        label="Chosen (line)" if not labeled_chosen_line else None)
                labeled_chosen_line = True
                chosen_x.append((mx, my))
            else:
                ax.plot(x_pts, y_pts, c="gray", alpha=0.4, linewidth=0.8, zorder=1,
                        label="All runs (line)" if not labeled_all_line else None)
                labeled_all_line = True
                all_x.append((mx, my))

        # Scatter: start (circle) and end (star) for each run
        if all_x:
            all_mx = np.concatenate([a[0] for a in all_x])
            all_my = np.concatenate([a[1] for a in all_x])
            ax.scatter(all_mx, all_my, s=20, c="gray", alpha=0.6, label="All runs", zorder=3)
        if chosen_x:
            starts_x = np.concatenate([a[0][0:1] for a in chosen_x])
            starts_y = np.concatenate([a[1][0:1] for a in chosen_x])
            ends_x = np.concatenate([a[0][1:] for a in chosen_x])
            ends_y = np.concatenate([a[1][1:] for a in chosen_x])
            ax.scatter(starts_x, starts_y, s=120, c="C2", alpha=0.9, marker="o",
                       edgecolors="darkgreen", linewidths=1.5, label="Chosen (start)", zorder=4)
            ax.scatter(ends_x, ends_y, s=180, c="C0", alpha=0.9, marker="*",
                       edgecolors="darkblue", linewidths=1, label="Chosen (end)", zorder=5)
        if best_x:
            starts_x = np.concatenate([a[0][0:1] for a in best_x])
            starts_y = np.concatenate([a[1][0:1] for a in best_x])
            ends_x = np.concatenate([a[0][1:] for a in best_x])
            ends_y = np.concatenate([a[1][1:] for a in best_x])
            ax.scatter(starts_x, starts_y, s=150, c="red", alpha=0.95, marker="o",
                       edgecolors="darkred", linewidths=2, label="Best RRMSE (start)", zorder=7)
            ax.scatter(ends_x, ends_y, s=220, c="red", alpha=0.95, marker="*",
                       edgecolors="darkred", linewidths=1.5, label="Best RRMSE (end)", zorder=8)

        label = _param_display_label(key, is_raw_keys)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-0.2)  # small gap before step 0 (max 0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # steps are integers only

        # Y-axis: bounds from chosen inits (initial + final) and from per-iteration series when present
        final_chosen_init = init_v[chosen]
        final_chosen_final = final_v[chosen]
        final_chosen_vals = np.concatenate([final_chosen_init, final_chosen_final])
        for i in range(len(chosen)):
            if chosen[i] and i < len(iter_series_list):
                ser = iter_series_list[i]
                if ser is not None and len(ser[1]) > 0:
                    final_chosen_vals = np.concatenate([final_chosen_vals, ser[1]])
        chosen_finite = final_chosen_vals[np.isfinite(final_chosen_vals)]
        use_log = log_y or ((not is_raw_keys) and key == "noise_log")
        if len(chosen_finite) > 0:
            if use_log:
                # Log-scale: only use positive values and multiplicative padding.
                pos = chosen_finite[chosen_finite > 0]
                if len(pos) > 0:
                    y_min, y_max = np.min(pos), np.max(pos)
                    if y_min > 0 and np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                        ax.set_yscale("log")
                        factor = 0.2
                        ax.set_ylim(y_min * (1 - factor), y_max * (1 + factor))
                    else:
                        # Fall back to default scaling if values are not suitable
                        y_min, y_max = np.min(chosen_finite), np.max(chosen_finite)
                        base_min_pad = 0.01 if (not is_raw_keys and key in ("noise", "noise_log")) else 0.2
                        pad = max((y_max - y_min) * 0.1, base_min_pad)
                        ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    # No positive values -> fall back to linear scaling
                    y_min, y_max = np.min(chosen_finite), np.max(chosen_finite)
                    base_min_pad = 0.01 if (not is_raw_keys and key in ("noise", "noise_log")) else 0.2
                    pad = max((y_max - y_min) * 0.1, base_min_pad)
                    ax.set_ylim(y_min - pad, y_max + pad)
            else:
                y_min, y_max = np.min(chosen_finite), np.max(chosen_finite)
                # Smaller minimum padding for actual (non-raw) noise, which typically lives in [0, 0.2]
                base_min_pad = 0.01 if (not is_raw_keys and key in ("noise", "noise_log")) else 0.2
                pad = max((y_max - y_min) * 0.1, base_min_pad)
                ax.set_ylim(y_min - pad, y_max + pad)
        else:
            # Fallback: use all runs only if no chosen runs have finite values
            all_vals = np.concatenate([init_v, final_v])
            finite = all_vals[np.isfinite(all_vals)]
            if len(finite) > 0:
                if use_log:
                    pos = finite[finite > 0]
                    if len(pos) > 0:
                        y_min, y_max = np.min(pos), np.max(pos)
                        if y_min > 0 and np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                            ax.set_yscale("log")
                            factor = 0.2
                            ax.set_ylim(y_min * (1 - factor), y_max * (1 + factor))
                            continue
                y_min, y_max = np.min(finite), np.max(finite)
                base_min_pad = 0.01 if (not is_raw_keys and key in ("noise", "noise_log")) else 0.2
                pad = max((y_max - y_min) * 0.1, base_min_pad)
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
        init_params = _params_flat_view(r.get("initial_parameters") or {})
        final_params = _params_flat_view(r.get("final_parameters") or {})
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

    param_keys_raw = _infer_hyperparameter_keys(all_runs)
    if param_keys_raw:
        arrays_raw = build_arrays_per_hyperparameter(all_runs, param_keys_raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path_raw = out_dir / f"trainer_hyperparams_raw_{title}.png"
        plot_hyperparameter_epochs(arrays_raw, title=title, save_path=save_path_raw, is_raw_keys=True)
        print(f"Trainer analysis plot (raw) saved to: {save_path_raw}")
    param_keys_actual = _infer_hyperparameter_keys_actual(all_runs)
    if param_keys_actual:
        arrays_actual = build_arrays_per_hyperparameter(all_runs, param_keys_actual)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path_actual = out_dir / f"trainer_hyperparams_{title}.png"
        if "noise" in arrays_actual:
            # Insert a log-scale copy of noise as a second subplot, directly after noise.
            arrays_actual["noise_log"] = arrays_actual["noise"]
            ordered_keys = ["noise", "noise_log"] + [k for k in param_keys_actual if k != "noise"]
            plot_hyperparameter_epochs(
                arrays_actual,
                title=title,
                save_path=save_path_actual,
                is_raw_keys=False,
                only_keys=ordered_keys,
            )
        else:
            plot_hyperparameter_epochs(arrays_actual, title=title, save_path=save_path_actual, is_raw_keys=False)
        print(f"Trainer analysis plot (actual) saved to: {save_path_actual}")

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

    param_keys_raw = _infer_hyperparameter_keys(all_runs)
    if param_keys_raw:
        arrays_raw = build_arrays_per_hyperparameter(all_runs, param_keys_raw)
        save_path_raw = out_dir / f"trainer_hyperparams_raw_{title}.png" if out_dir is not None else None
        plot_hyperparameter_epochs(arrays_raw, title=title, save_path=save_path_raw, is_raw_keys=True)
    param_keys_actual = _infer_hyperparameter_keys_actual(all_runs)
    if param_keys_actual:
        arrays_actual = build_arrays_per_hyperparameter(all_runs, param_keys_actual)
        save_path_actual = out_dir / f"trainer_hyperparams_{title}.png" if out_dir is not None else None
        if "noise" in arrays_actual:
            arrays_actual["noise_log"] = arrays_actual["noise"]
            ordered_keys = ["noise", "noise_log"] + [k for k in param_keys_actual if k != "noise"]
            plot_hyperparameter_epochs(
                arrays_actual,
                title=title,
                save_path=save_path_actual,
                is_raw_keys=False,
                only_keys=ordered_keys,
            )
        else:
            plot_hyperparameter_epochs(arrays_actual, title=title, save_path=save_path_actual, is_raw_keys=False)

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
                            "C:/Users/kianb/Repos/gp-private/experiments_final/results_IDETC/10_runs_logging_full_Gaussian/griewank/",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results/plotsfortyler",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/custom_GPvsPFN/buckling",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/custom_GPvsPFN/buckling/buckling",
                            # "C:/Users/kianb/Repos/gp-private/experiments_final/results_final/1_12/griewank",
                            ]

# If True, DEFAULT_PATHS are treated as roots and we recurse through all subfolders
# to find trainer_analysis/*_GP_Trainer_Analysis.json.
DEFAULT_ALL_SUBFOLDERS: bool = True


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent
    # If no CLI args but DEFAULT_PATHS set, use those (paths relative to script dir)
    if not sys.argv[1:] and DEFAULT_PATHS:
        default_args: list[str] = []
        if DEFAULT_ALL_SUBFOLDERS:
            default_args.append("--all")
        default_args.extend(str(root / p) for p in DEFAULT_PATHS)
        sys.argv[1:] = default_args
    main()
