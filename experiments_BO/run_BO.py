"""
run_BO: equivalent of train_eval for Bayesian Optimization.

- run_BO_gp: run one BO loop with a GP surrogate (data and bounds are passed in).
- run_BO_pfn: placeholder for future PFN-based BO.

BX scripts (B1–B9) load data and bounds using load_experimental_data, set up
experiment runs, then call run_BO_gp (or run_BO_pfn) from this module. All
config (optimizer, kernel, etc.) is passed in from the BX script (which reads
from experiments_BO/defaults.py); this module does not import defaults.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch.distributions import Normal

import gpplus
from gpplus.training.eval2 import evaluate_gp_model
from gpplus.utils import set_seed, train_eval_gp
from gpplus.utils.metrics_functions import compute_nis, compute_metrics
from gpplus.utils.standard_scaler import StandardScaler, UniformScaler
from GITBO.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from GITBO.gitbo_bo_helpers import sample_dominant_subspace, compute_acquisition_values_gitbo

try:
    from tabpfn import TabPFNRegressor as SklearnTabPFNRegressor
except ImportError:
    SklearnTabPFNRegressor = None

# No defaults import: BX scripts (B1, B4, etc.) pass in all config from experiments_BO/defaults.py.

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


AcquisitionType = Literal["EI", "TS"]


@dataclass
class run_BO_Result:
    """Container for per-run start_size results."""

    best_y_history: List[float]
    x_history: torch.Tensor
    y_history: torch.Tensor
    best_y_clean_history: List[float]
    y_clean_history: Optional[torch.Tensor]
    test_metrics_history: List[Dict[str, float]]
    nis_history: List[Dict[str, float]]
    n_iterations: int
    # Per-iteration: chosen point (raw), AF value, train time (s), AF time (s)
    x_chosen_history: List[torch.Tensor] = field(default_factory=list)
    af_value_history: List[float] = field(default_factory=list)
    train_time_history: List[float] = field(default_factory=list)
    af_time_history: List[float] = field(default_factory=list)


# ---------- Scalers (caller passes x_standardize_method) ----------


def _make_X_scaler(
    x_standardize_method: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Any]:
    """Build X scaler (0=Gaussian, 1=Uniform [0,1], 2=Uniform [-1,1]). Caller passes x_standardize_method from defaults."""
    if x_standardize_method == 0:
        return StandardScaler()
    if x_standardize_method == 1:
        return UniformScaler(scale_to_neg_one=False)
    if x_standardize_method == 2:
        return UniformScaler(scale_to_neg_one=True)
    raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")


# ---------- Bounds helpers (match load_experimental_data) ----------


def get_bounds_wing(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Wing 10D bounds as (2, d) tensor. Matches load_experimental_data."""
    l_bound = torch.tensor(
        [150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025],
        dtype=dtype,
        device=device,
    )
    u_bound = torch.tensor(
        [200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08],
        dtype=dtype,
        device=device,
    )
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_ackley(
    dimensions: int,
    x_bounds: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Ackley box bounds (2, d). Matches load_experimental_data."""
    lb, ub = float(x_bounds[0]), float(x_bounds[1])
    l_bound = torch.full((dimensions,), lb, dtype=dtype, device=device)
    u_bound = torch.full((dimensions,), ub, dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_buckling(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Buckling 4D bounds. Matches load_experimental_data."""
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=dtype, device=device)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_borehole(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Borehole 8D bounds. Matches load_experimental_data."""
    l_bound = torch.tensor(
        [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0],
        dtype=dtype,
        device=device,
    )
    u_bound = torch.tensor(
        [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0],
        dtype=dtype,
        device=device,
    )
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_analytic(
    dimensions: int,
    x_bounds: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generic box bounds (2, d) for analytic problems (Rastrigin, Rosenbrock, etc.)."""
    lb, ub = float(x_bounds[0]), float(x_bounds[1])
    l_bound = torch.full((dimensions,), lb, dtype=dtype, device=device)
    u_bound = torch.full((dimensions,), ub, dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


# ---------- Acquisition helpers ----------


def _sample_sobol_in_bounds(
    n: int,
    bounds: torch.Tensor,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    d = bounds.shape[1]
    base = sobol.draw(n).to(dtype=dtype, device=device)
    l, u = bounds[0], bounds[1]
    return l + (u - l) * base


def _compute_ei(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f_signed: float,
) -> torch.Tensor:
    std = torch.clamp(std, min=1e-12)
    best = torch.as_tensor(best_f_signed, dtype=mean.dtype, device=mean.device)
    improvement = mean - best
    z = improvement / std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    pdf = torch.exp(normal.log_prob(z))
    cdf = normal.cdf(z)
    ei = improvement * cdf + std * pdf
    ei = torch.where(std <= 1e-12, torch.zeros_like(ei), ei)
    return ei


def _compute_ts_samples(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.clamp(std, min=0.0)
    return torch.normal(mean, std)


def _select_next_point_sampling(
    model,
    bounds: torch.Tensor,
    n_candidates: int,
    acquisition: AcquisitionType,
    best_f_signed: float,
    sign: float,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    x_scaler: Optional[Any] = None,
) -> Tuple[torch.Tensor, float]:
    X_cand = _sample_sobol_in_bounds(n_candidates, bounds, sobol, dtype, device)
    X_for_model = x_scaler.transform(X_cand) if x_scaler is not None else X_cand
    mean, _, _, std = evaluate_gp_model(model, X_for_model)
    mean_signed = sign * mean.squeeze(-1)
    std = std.squeeze(-1)
    if acquisition.upper() == "EI":
        acq_vals = _compute_ei(mean_signed, std, best_f_signed)
    elif acquisition.upper() == "TS":
        acq_vals = _compute_ts_samples(mean_signed, std)
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition}")
    idx = torch.argmax(acq_vals)
    af_val = float(acq_vals[idx].item())
    return X_cand[idx : idx + 1], af_val


def _optimize_ei_with_scipy(
    model,
    bounds: torch.Tensor,
    n_starts: int,
    best_f_signed: float,
    sign: float,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    x_scaler: Optional[Any] = None,
) -> Tuple[torch.Tensor, float]:
    if minimize is None:
        x_next, af_val = _select_next_point_sampling(
            model=model,
            bounds=bounds,
            n_candidates=n_starts,
            acquisition="EI",
            best_f_signed=best_f_signed,
            sign=sign,
            sobol=sobol,
            dtype=dtype,
            device=device,
            x_scaler=x_scaler,
        )
        return x_next, af_val
    d = bounds.shape[1]
    X_starts = _sample_sobol_in_bounds(n_starts, bounds, sobol, dtype, device).cpu().numpy()
    scipy_bounds = [(float(bounds[0, i].cpu().item()), float(bounds[1, i].cpu().item())) for i in range(d)]

    def neg_ei_np(x_np: np.ndarray) -> float:
        x = torch.as_tensor(x_np, dtype=dtype, device=device).unsqueeze(0)
        x_for_model = x_scaler.transform(x) if x_scaler is not None else x
        mean, _, _, std = evaluate_gp_model(model, x_for_model)
        mean_signed = sign * mean.view(-1)
        std = std.view(-1)
        ei = _compute_ei(mean_signed, std, best_f_signed)
        return -float(ei.item())

    best_x = None
    best_ei = -np.inf
    with torch.no_grad():
        X_starts_t = torch.as_tensor(X_starts, dtype=dtype, device=device)
        X_for_model = x_scaler.transform(X_starts_t) if x_scaler is not None else X_starts_t
        mean_s, _, _, std_s = evaluate_gp_model(model, X_for_model)
        mean_s_signed = sign * mean_s.view(-1)
        std_s = std_s.view(-1)
        ei_starts = _compute_ei(mean_s_signed, std_s, best_f_signed)
    k = min(5, n_starts)
    topk_idx = torch.topk(ei_starts, k).indices.cpu().tolist()
    for idx in topk_idx:
        x0 = X_starts[idx]
        res = minimize(
            neg_ei_np,
            x0,
            method="Powell",
            bounds=scipy_bounds,
            options={"maxiter": 50, "disp": False},
        )
        if not res.success:
            continue
        ei_val = -res.fun
        if ei_val > best_ei:
            best_ei = ei_val
            best_x = res.x
    if best_x is None:
        idx = int(torch.argmax(ei_starts).item())
        best_x = X_starts[idx]
        best_ei = float(ei_starts[idx].item())
    return torch.as_tensor(best_x, dtype=dtype, device=device).unsqueeze(0), best_ei


# ---------- Main run API ----------


def run_BO_gp(
    *,
    X_init: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    bounds: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    objective_fn_clean: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    minimization_problem: bool = False,
    acquisition: AcquisitionType = "EI",
    gp_optimize_af: bool = True,
    n_AF_opt: int = 16,
    n_AF_sample: int = 5000,
    max_iter: int = 30,
    patience_no_improve: int = 10,
    run_seed: int = 42,
    num_epochs: int = 1,
    num_inits: int = 16,
    lr: Optional[float] = None,
    convergence_patience: int = 10,
    min_epochs: int = 0,
    min_loss_change: float = 1e-7,
    optimizer_class: Any = None,
    optimizer_kwargs: Optional[Dict] = None,
    initializer_class: Any = None,
    gp_device: str = "cpu",
    gp_dtype: torch.dtype = torch.float64,
    standardize_X: bool = True,
    standardize_y: bool = True,
    x_standardize_method: int = 2,
    save_path: Optional[str] = None,
    run_log_filename: Optional[str] = None,
    verbose: bool = True,
    kernel_module: Any = None,
    mean_module: Any = None,
    likelihood: Any = None,
    log_lbfgs_inner: bool = False,
    bo_test_metrics: bool = True,
) -> "run_BO_Result":
    """
    Run one BO loop with a GP surrogate (equivalent of train_eval_gp for BO).

    Caller (BX script) is responsible for loading data and bounds via load_experimental_data.
    When standardize_X/standardize_y are True, X and y are scaled for GP training; bounds and
    objective remain in raw space; chosen points and logs are in raw space.
    """
    device = torch.device(gp_device)
    dtype = gp_dtype
    if optimizer_kwargs is None:
        raise ValueError("optimizer_kwargs must be passed in by the BX script (e.g. from defaults.TRAINER_OPTIMIZER_KWARGS)")
    set_seed(run_seed)

    if objective_fn_clean is None:
        objective_fn_clean = objective_fn

    X_train_raw = X_init.to(device=device, dtype=dtype)
    y_train_raw = objective_fn(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    y_train_clean_raw = (
        objective_fn_clean(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    )
    X_test_raw = X_test.to(device=device, dtype=dtype)
    y_test_raw = y_test.to(device=device, dtype=dtype).reshape(-1)

    # Fit scalers on initial data
    x_scaler: Optional[Any] = None
    y_scaler: Optional[Any] = None
    if standardize_X:
        x_scaler = _make_X_scaler(x_standardize_method, device, dtype)
        x_scaler.fit(X_train_raw)
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_raw.unsqueeze(1))
        # avoid div-by-zero
        if (y_scaler.std is not None) and (y_scaler.std.squeeze() == 0).all():
            y_scaler.std = torch.ones_like(y_scaler.std, device=device, dtype=dtype)

    X_train = x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
    y_train = y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1) if y_scaler is not None else y_train_raw
    X_test = x_scaler.transform(X_test_raw) if x_scaler is not None else X_test_raw
    y_test = y_test_raw  # train_eval_gp expects raw y_test for metrics; we pass y_train_mean/std for denorm

    y_train_mean = y_scaler.mean.squeeze() if y_scaler is not None else None
    y_train_std = y_scaler.std.squeeze() if y_scaler is not None else None

    sobol = SobolEngine(X_train_raw.shape[1], scramble=True, seed=run_seed)
    # Minimize objective → maximize (-y) → sign = -1. Maximize objective → sign = 1.
    sign = -1.0 if minimization_problem else 1.0

    best_y_history: List[float] = []
    best_y_clean_history: List[float] = []
    test_metrics_history: List[Dict[str, float]] = []
    nis_history: List[Dict[str, float]] = []
    x_chosen_history: List[torch.Tensor] = []
    af_value_history: List[float] = []
    train_time_history: List[float] = []
    af_time_history: List[float] = []
    no_improve = 0

    for _ in range(max_iter):
        model = gpplus.models.GPR(
            X_train,
            y_train,
            kernel_module=kernel_module,
            mean_module=mean_module,
            likelihood=likelihood,
        )
        callbacks: List = []

        t_train = time.perf_counter()
        gp_metric, y_pred_test, output_std_test, _ = train_eval_gp(
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
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            source_cols=None,
            trainer_info=False,
            callback_save_path=None,
            callbacks=callbacks,
            log_lbfgs_inner=log_lbfgs_inner,
        )
        train_time_history.append(time.perf_counter() - t_train)

        if bo_test_metrics and isinstance(gp_metric, dict):
            test_metrics_history.append(gp_metric)
        if bo_test_metrics and y_pred_test is not None and output_std_test is not None:
            try:
                nis_dict = compute_nis(
                    y_true=y_test,
                    y_hat=y_pred_test,
                    output_std=output_std_test,
                )
                nis_history.append(nis_dict)
            except Exception:
                nis_history.append({})

        # Best so far: raw for logging, scaled for AF (model predicts in scaled space when standardize_y)
        y_train_raw_cur = (
            y_scaler.inverse_transform(y_train.unsqueeze(1)).squeeze(1)
            if y_scaler is not None
            else y_train
        )
        y_train_clean_raw_cur = y_train_clean_raw
        y_signed_raw = sign * y_train_raw_cur
        best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
        best_raw_val = y_train_raw_cur[best_idx].item()
        best_y_history.append(best_raw_val)

        y_signed_clean = sign * y_train_clean_raw_cur
        _, best_clean_idx = torch.max(y_signed_clean, dim=0)
        best_clean_raw_val = y_train_clean_raw_cur[best_clean_idx].item()
        best_y_clean_history.append(best_clean_raw_val)
        best_f_signed_for_af = float((sign * y_train).max().item())  # scaled space for EI/TS

        if acquisition.upper() not in {"EI", "TS"}:
            raise ValueError(f"Unsupported acquisition function: {acquisition}")

        t_af = time.perf_counter()
        if acquisition.upper() == "EI" and gp_optimize_af:
            x_next_raw, af_val = _optimize_ei_with_scipy(
                model=model,
                bounds=bounds,
                n_starts=n_AF_opt,
                best_f_signed=best_f_signed_for_af,
                sign=sign,
                sobol=sobol,
                dtype=dtype,
                device=device,
                x_scaler=x_scaler,
            )
        else:
            n_cand = n_AF_sample if n_AF_sample is not None else 5000
            x_next_raw, af_val = _select_next_point_sampling(
                model=model,
                bounds=bounds,
                n_candidates=n_cand,
                acquisition=acquisition.upper(),
                best_f_signed=best_f_signed_for_af,
                sign=sign,
                sobol=sobol,
                dtype=dtype,
                device=device,
                x_scaler=x_scaler,
            )
        af_time_history.append(time.perf_counter() - t_af)

        x_chosen_history.append(x_next_raw.detach().cpu().clone())
        af_value_history.append(af_val)

        if verbose:
            iter_idx = len(best_y_history)
            print(
                f"  BO iter {iter_idx:02d} | "
                f"best_y={best_y_history[-1]:.4f} | "
                f"af={af_val:.4f} | "
                f"train_s={train_time_history[-1]:.4f} | "
                f"af_s={af_time_history[-1]:.4f}"
            )
            print("  " + "-" * 70)

        y_next = objective_fn(x_next_raw).to(device=device, dtype=dtype).reshape(-1)
        y_next_clean = (
            objective_fn_clean(x_next_raw).to(device=device, dtype=dtype).reshape(-1)
        )
        new_signed = sign * y_next[0]
        if new_signed > best_signed_val + 1e-8:
            no_improve = 0
        else:
            no_improve += 1

        # Append in raw space for objective and history; scaled for GP
        X_train_raw = torch.cat([X_train_raw, x_next_raw], dim=0)
        y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
        y_train_clean_raw = torch.cat(
            [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
        )
        X_train = x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
        y_train = y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1) if y_scaler is not None else y_train_raw

        # Early stopping on lack of improvement only when patience_no_improve is set and > 0.
        # If patience_no_improve is None or 0, this condition is disabled (use max_iter only).
        if (
            patience_no_improve is not None
            and patience_no_improve > 0
            and no_improve >= patience_no_improve
        ):
            break

    result = run_BO_Result(
        best_y_history=best_y_history,
        best_y_clean_history=best_y_clean_history,
        x_history=X_train_raw.detach().cpu(),
        y_history=y_train_raw.detach().cpu(),
        y_clean_history=y_train_clean_raw.detach().cpu(),
        test_metrics_history=test_metrics_history,
        nis_history=nis_history,
        n_iterations=len(best_y_history),
        x_chosen_history=x_chosen_history,
        af_value_history=af_value_history,
        train_time_history=train_time_history,
        af_time_history=af_time_history,
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_data: Dict[str, Any] = {
            "run_seed": run_seed,
            "n_iterations": result.n_iterations,
            "x_chosen": [t.tolist() for t in result.x_chosen_history],
            "af_values": result.af_value_history,
            "train_time_s": result.train_time_history,
            "af_time_s": result.af_time_history,
            "best_y_history": result.best_y_history,
            "best_y_clean_history": result.best_y_clean_history,
            "final_best_y": result.best_y_history[-1]
            if result.best_y_history
            else None,
            "final_best_y_clean": result.best_y_clean_history[-1]
            if result.best_y_clean_history
            else None,
        }
        log_filename = run_log_filename if run_log_filename else "bo_run_log.json"
        if not log_filename.endswith(".json"):
            log_filename += ".json"
        log_file = os.path.join(save_path, log_filename)
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    return result


def _scale_to_unit(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Scale X from bounds to [0,1]."""
    lb = bounds[0].to(X.device, dtype=X.dtype)
    ub = bounds[1].to(X.device, dtype=X.dtype)
    return (X - lb) / (ub - lb + 1e-9)


def _scale_from_unit(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Scale X from [0,1] back to bounds."""
    lb = bounds[0].to(X.device, dtype=X.dtype)
    ub = bounds[1].to(X.device, dtype=X.dtype)
    return lb + X * (ub - lb)


def run_BO_pfn(
    *,
    X_init: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    bounds: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    minimization_problem: bool = False,
    acquisition: AcquisitionType = "EI",
    n_AF_sample: int = 5000,
    max_iter: int = 30,
    patience_no_improve: int = 10,
    run_seed: int = 42,
    gp_device: str = "cpu",
    gp_dtype: torch.dtype = torch.float64,
    save_path: Optional[str] = None,
    run_log_filename: Optional[str] = None,
    verbose: bool = True,
    pfn_device: Optional[str] = None,
    pfn_dtype: torch.dtype = torch.float32,
    bo_test_metrics: bool = True,
    gi_pfn: bool = False,
    objective_fn_clean: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> run_BO_Result:
    """
    Run one BO loop with a PFN surrogate.

    If gi_pfn=True: use gradient-informed (GITBO) method with VanillaDirectTabPFNRegressor.
    If gi_pfn=False: use standard TabPFNRegressor (Sklearn) + 5k Sobol candidates.
    """
    device = torch.device(gp_device)
    dtype = gp_dtype
    if pfn_device is None:
        pfn_device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(run_seed)

    if objective_fn_clean is None:
        objective_fn_clean = objective_fn

    X_train_raw = X_init.to(device=device, dtype=dtype)
    y_train_raw = objective_fn(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    y_train_clean_raw = (
        objective_fn_clean(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    )
    X_test_raw = X_test.to(device=device, dtype=dtype)
    y_test_raw = y_test.to(device=device, dtype=dtype).reshape(-1)

    sobol = SobolEngine(X_train_raw.shape[1], scramble=True, seed=run_seed)
    sign = -1.0 if minimization_problem else 1.0

    best_y_history: List[float] = []
    best_y_clean_history: List[float] = []
    test_metrics_history: List[Dict[str, float]] = []
    nis_history: List[Dict[str, float]] = []
    x_chosen_history: List[torch.Tensor] = []
    af_value_history: List[float] = []
    train_time_history: List[float] = []
    af_time_history: List[float] = []
    no_improve = 0

    bounds_dev = bounds.to(device=device, dtype=dtype)

    if gi_pfn:
        # --- GITBO path: gradient-informed subspace + VanillaDirectTabPFNRegressor ---
        from GITBO.gitbo_bo_helpers import (
            compute_acquisition_values_gitbo,
            sample_dominant_subspace,
        )

        N_PENDING = n_AF_sample if n_AF_sample else 5000
        N_CANDIDATES = 1
        rank_r = 15
        scale = 0.2
        tkwargs = {"device": torch.device(pfn_device), "dtype": pfn_dtype}
        grad_est = None

        for _ in range(max_iter):
            y_signed_raw = sign * y_train_raw
            best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
            best_raw_val = y_train_raw[best_idx].item()
            best_y_history.append(best_raw_val)

            y_signed_clean = sign * y_train_clean_raw
            _, best_clean_idx = torch.max(y_signed_clean, dim=0)
            best_clean_raw_val = y_train_clean_raw[best_clean_idx].item()
            best_y_clean_history.append(best_clean_raw_val)

            if acquisition.upper() not in {"EI", "TS"}:
                raise ValueError(f"Unsupported acquisition for PFN: {acquisition}")

            t_af = time.perf_counter()

            # Scale to [0,1] for GITBO
            trained_X = _scale_to_unit(X_train_raw, bounds_dev).to(**tkwargs)
            trained_Y = (sign * y_train_raw).reshape(-1, 1).to(**tkwargs)

            if grad_est is None:
                total_points = N_CANDIDATES * N_PENDING
                X_pen_draw = sobol.draw(total_points).to(**tkwargs)
                X_pen = X_pen_draw.view(N_PENDING, N_CANDIDATES, X_train_raw.shape[1])
            else:
                X_pen = sample_dominant_subspace(
                    trained_X,
                    trained_Y,
                    X_train_raw.shape[1],
                    grad_est,
                    sobol,
                    rank_r=rank_r,
                    n_samples=N_PENDING,
                    N_CANDIDATES=N_CANDIDATES,
                    scale=scale,
                    GI_SUBSPACE=True,
                    tkwargs=tkwargs,
                )

            ACQ, _, grad_est = compute_acquisition_values_gitbo(
                acquisition=acquisition,
                DIM=X_train_raw.shape[1],
                N_PENDING=N_PENDING,
                N_CANDIDATES=N_CANDIDATES,
                trained_X=trained_X,
                trained_Y=trained_Y,
                X_pen=X_pen,
                gpu_device=pfn_device,
                tkwargs=tkwargs,
            )

            X_pen_perm = X_pen.permute(1, 0, 2)
            ACQ_perm = ACQ.permute(1, 0)
            best_cand_idx = torch.argmax(ACQ_perm, dim=1)
            best_candidate_norm = X_pen_perm[0, best_cand_idx[0], :].unsqueeze(0)
            x_next_raw = _scale_from_unit(best_candidate_norm, bounds_dev)

            af_time_history.append(time.perf_counter() - t_af)
            train_time_history.append(0.0)
            x_chosen_history.append(x_next_raw.detach().cpu().clone())
            af_value_history.append(float(ACQ_perm[0, best_cand_idx[0]].item()))

            if verbose:
                iter_idx = len(best_y_history)
                print(
                    f"  BO iter {iter_idx:02d} | "
                    f"best_y={best_y_history[-1]:.4f} | "
                    f"af={af_value_history[-1]:.4f} | "
                    f"train_s=0.0000 | af_s={af_time_history[-1]:.4f}"
                )
                print("  " + "-" * 70)

            y_next = objective_fn(x_next_raw.to(device=device, dtype=dtype)).to(
                device=device, dtype=dtype
            ).reshape(-1)
            y_next_clean = (
                objective_fn_clean(x_next_raw.to(device=device, dtype=dtype))
                .to(device=device, dtype=dtype)
                .reshape(-1)
            )
            new_signed = sign * y_next[0]
            if new_signed > best_signed_val + 1e-8:
                no_improve = 0
            else:
                no_improve += 1

            X_train_raw = torch.cat(
                [X_train_raw, x_next_raw.to(device=device, dtype=dtype)], dim=0
            )
            y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
            y_train_clean_raw = torch.cat(
                [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
            )

            if (
                patience_no_improve is not None
                and patience_no_improve > 0
                and no_improve >= patience_no_improve
            ):
                break

    else:
        # --- Vanilla PFN path: TabPFNRegressor (Sklearn) + 5k Sobol ---
        if SklearnTabPFNRegressor is None:
            raise ImportError(
                "gi_pfn=False requires tabpfn package. Install with: pip install tabpfn"
            )

        regressor = SklearnTabPFNRegressor(device=pfn_device)
        n_cand = n_AF_sample if n_AF_sample else 5000

        for _ in range(max_iter):
            y_signed_raw = sign * y_train_raw
            best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
            best_raw_val = y_train_raw[best_idx].item()
            best_y_history.append(best_raw_val)

            y_signed_clean = sign * y_train_clean_raw
            _, best_clean_idx = torch.max(y_signed_clean, dim=0)
            best_clean_raw_val = y_train_clean_raw[best_clean_idx].item()
            best_y_clean_history.append(best_clean_raw_val)

            if acquisition.upper() not in {"EI", "TS"}:
                raise ValueError(f"Unsupported acquisition function for PFN: {acquisition}")

            t_af = time.perf_counter()
            X_cand_raw = _sample_sobol_in_bounds(
                n=n_cand,
                bounds=bounds_dev,
                sobol=sobol,
                dtype=dtype,
                device=device,
            )

            # TabPFNRegressor fit/predict: fit on train, predict on candidates
            X_train_np = X_train_raw.cpu().numpy()
            y_train_signed_np = (sign * y_train_raw).cpu().numpy()
            X_cand_np = X_cand_raw.cpu().numpy()

            regressor.fit(X_train_np, y_train_signed_np)
            acq_pred = regressor.predict(X_cand_np)

            idx = int(np.argmax(acq_pred))
            af_val = float(acq_pred[idx])
            x_next_raw = X_cand_raw[idx : idx + 1]
            af_time_history.append(time.perf_counter() - t_af)
            train_time_history.append(0.0)

            x_chosen_history.append(x_next_raw.detach().cpu().clone())
            af_value_history.append(af_val)

            if verbose:
                iter_idx = len(best_y_history)
                print(
                    f"  BO iter {iter_idx:02d} | "
                    f"best_y={best_y_history[-1]:.4f} | "
                    f"af={af_val:.4f} | "
                    f"train_s=0.0000 | af_s={af_time_history[-1]:.4f}"
                )
                print("  " + "-" * 70)

            y_next = objective_fn(x_next_raw.to(device=device, dtype=dtype)).to(
                device=device, dtype=dtype
            ).reshape(-1)
            y_next_clean = (
                objective_fn_clean(x_next_raw.to(device=device, dtype=dtype))
                .to(device=device, dtype=dtype)
                .reshape(-1)
            )
            new_signed = sign * y_next[0]
            if new_signed > best_signed_val + 1e-8:
                no_improve = 0
            else:
                no_improve += 1

            X_train_raw = torch.cat(
                [X_train_raw, x_next_raw.to(device=device, dtype=dtype)], dim=0
            )
            y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
            y_train_clean_raw = torch.cat(
                [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
            )

            if (
                patience_no_improve is not None
                and patience_no_improve > 0
                and no_improve >= patience_no_improve
            ):
                break

    # Final test metrics using PFN surrogate, if enabled.
    if bo_test_metrics:
        try:
            N_train_final = X_train_raw.shape[0]
            N_test = X_test_raw.shape[0]
            t_pred = time.perf_counter()

            if gi_pfn:
                # Use VanillaDirectTabPFNRegressor for test metrics
                regressor_metrics = VanillaDirectTabPFNRegressor(device=pfn_device)
                X_train_pfn = X_train_raw.to(device=pfn_device, dtype=pfn_dtype)
                X_test_pfn = X_test_raw.to(device=pfn_device, dtype=pfn_dtype)
                X_train_seq = X_train_pfn.unsqueeze(1)
                X_test_seq = X_test_pfn.unsqueeze(1)
                X_concat = torch.cat([X_train_seq, X_test_seq], dim=0)
                y_train_signed_pfn = (sign * y_train_raw).to(device=pfn_device, dtype=pfn_dtype)
                Y_train_time = y_train_signed_pfn.view(-1, 1)
                Y_pad = torch.zeros(N_test, 1, device=pfn_device, dtype=pfn_dtype)
                Y_time = torch.cat([Y_train_time, Y_pad], dim=0)
                Y_full = Y_time.unsqueeze(1)
                single_eval_pos = N_train_final
                with torch.no_grad():
                    out = regressor_metrics.forward(X_concat, Y_full, single_eval_pos)
                    logits = out["standard"]
                    mean_all = regressor_metrics.predict_mean(logits)
                    var_all = regressor_metrics.predict_variance(logits)
                    std_all = torch.clamp(var_all, min=1e-8).sqrt()
                mean_test_signed = mean_all[single_eval_pos:, 0]
                std_test = std_all[single_eval_pos:, 0]
            else:
                # Use TabPFNRegressor predict (no uncertainty from sklearn API)
                X_train_np = X_train_raw.cpu().numpy()
                X_test_np = X_test_raw.cpu().numpy()
                y_train_signed_np = (sign * y_train_raw).cpu().numpy()
                regressor.fit(X_train_np, y_train_signed_np)
                mean_test_signed = torch.tensor(
                    regressor.predict(X_test_np), dtype=pfn_dtype, device=pfn_device
                )
                std_test = torch.ones_like(mean_test_signed, device=pfn_device) * 1e-6

            pred_time = time.perf_counter() - t_pred
            y_hat_test = sign * mean_test_signed

            metrics_dict = compute_metrics(
                y_true=y_test_raw,
                y_hat=y_hat_test,
                output_std=std_test,
                training_time=0.0,
                prediction_time=pred_time,
            )
            test_metrics_history.append(metrics_dict)

            nis_dict = compute_nis(
                y_true=y_test_raw,
                y_hat=y_hat_test,
                output_std=std_test,
            )
            nis_history.append(nis_dict)
        except Exception:
            # If anything goes wrong, append empty dicts so downstream code can still index [-1].
            test_metrics_history.append({})
            nis_history.append({})

    result = run_BO_Result(
        best_y_history=best_y_history,
        best_y_clean_history=best_y_clean_history,
        x_history=X_train_raw.detach().cpu(),
        y_history=y_train_raw.detach().cpu(),
        y_clean_history=y_train_clean_raw.detach().cpu(),
        test_metrics_history=test_metrics_history,
        nis_history=nis_history,
        n_iterations=len(best_y_history),
        x_chosen_history=x_chosen_history,
        af_value_history=af_value_history,
        train_time_history=train_time_history,
        af_time_history=af_time_history,
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_data: Dict[str, Any] = {
            "run_seed": run_seed,
            "n_iterations": result.n_iterations,
            "x_chosen": [t.tolist() for t in result.x_chosen_history],
            "af_values": result.af_value_history,
            "train_time_s": result.train_time_history,
            "af_time_s": result.af_time_history,
            "best_y_history": result.best_y_history,
            "best_y_clean_history": result.best_y_clean_history,
            "final_best_y": result.best_y_history[-1]
            if result.best_y_history
            else None,
            "final_best_y_clean": result.best_y_clean_history[-1]
            if result.best_y_clean_history
            else None,
        }
        log_filename = run_log_filename if run_log_filename else "bo_run_log_pfn.json"
        if not log_filename.endswith(".json"):
            log_filename += ".json"
        log_file = os.path.join(save_path, log_filename)
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    return result
