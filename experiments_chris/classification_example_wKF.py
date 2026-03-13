# gp_rbf_powerexp_matern05_tabpfn_multistart_stable_paperfig_kf.py
#
# Benchmark on UCI Electrical Grid Stability Simulated Data (UCI id=471)
# Binary target: stabf (stable=1, unstable=0)
#
# Models:
#   - GP+ GPClassifier with Gaussian / RBF input kernel (ARD)
#   - GP+ GPClassifier with Matérn input kernel (ARD, nu=0.5)
#   - GP+ GPClassifier with Power Exponential input kernel (ARD, learned power)
#   - TabPFN classifier (current default, v2.5 weights)
#   - TabPFN classifier (backward-compatible v2 weights)
#
# Optimization:
#   - ALL GP-based models use gpplus.training.optimizers.LBFGSScipy
#   - GP-based models use:
#         * 10 experimental runs (different data splits / seeds)
#         * 16 hyperparameter initializations (multi-start restarts) per run
#         * best successful restart selected by lowest final training NLL
#   - Failed GP restarts (e.g. Cholesky failure / NaN loss) are skipped
#   - TabPFN uses its own built-in fit/predict pipeline (no GP-style hyperparameter restarts)
#
# Experiment settings:
#   - fixed 500-sample training pool per seed
#   - nested train sizes = [50, 100, 300, 500]
#   - experimental runs = 10 (seeds 0..9)
#   - GP restart initializations per run = 16
#   - test set = ALL remaining samples outside that fixed 500-pool
#
# Outputs:
#   - raw results CSV
#   - summary CSV
#   - averaged training curve PNGs
#   - paper summary figure PNG with 3 subplots:
#         * Train Negative Log Likelihood (GP+ models only)
#         * Test Accuracy (GP+ models + PFN 2.5 + PFN 2.0)
#         * Test Brier Score (GP+ models + PFN 2.5 + PFN 2.0)
#   - GP-only Kernel Flows figure:
#         * Kernel Flows (GP+ models only)

import os
import json
import time
import copy
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from ucimlrepo import fetch_ucirepo

from gpplus.models.gpc import GPClassifier
from gpplus.likelihoods.dirichlet_utils import prepare_dirichlet_targets
from gpplus.training.optimizers import LBFGSScipy
from gpplus.kernels.power_exponential_kernel import (
    PowerExponentialKernel,
    PowerExponentialKernelFixed,
)
from gpplus.kernels.matern_kernel import MaternKernel

# -------------------------
# Optional TabPFN imports
# -------------------------
try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    TABPFN_AVAILABLE = True
    TABPFN_IMPORT_ERROR = None
except Exception as e:
    TABPFN_AVAILABLE = False
    TABPFN_IMPORT_ERROR = repr(e)


# ============================================================
# Repro
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base_dir = os.path.dirname(path) or "."
        base_name = os.path.basename(path)
        stem, ext = os.path.splitext(base_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt1 = os.path.join(base_dir, f"{stem}_{ts}{ext}")
        try:
            df.to_csv(alt1, index=False)
            return alt1
        except Exception:
            alt2 = f"{stem}_{ts}{ext}"
            df.to_csv(alt2, index=False)
            return alt2


def sanitize_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("+", "plus")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace("=", "-")
            .replace("/", "_")
    )


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return float(obj.item())
        return obj.reshape(-1).tolist()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.item())
        return obj.reshape(-1).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, indent=2)


def is_numerical_gp_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    triggers = [
        "cholesky",
        "not positive definite",
        "singular",
        "symeig",
        "linalgerror",
        "nan",
        "inf",
        "not p.d.",
        "matrix not positive definite",
        "psd",
    ]
    return any(t in msg for t in triggers)


# ============================================================
# Dataset
# ============================================================

def ensure_binary_01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    vals = np.unique(y)
    if set(vals.tolist()) == {0, 1}:
        return y.astype(np.int64)
    raise RuntimeError(f"Expected binary labels {{0,1}}, got unique values: {vals[:20]}")


def load_grid_stability_from_uci(
    drop_stab: bool = True,
    drop_p1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grid = fetch_ucirepo(id=471)
    X_df: pd.DataFrame = grid.data.features.copy()
    y_df: pd.DataFrame = grid.data.targets.copy()

    if "stabf" not in y_df.columns:
        raise ValueError("Expected target column 'stabf' from UCI loader.")

    y = y_df["stabf"].map({"stable": 1, "unstable": 0})
    if y.isna().any():
        raise ValueError("Unexpected stabf labels encountered.")

    drop_cols = []
    if drop_stab and "stab" in X_df.columns:
        drop_cols.append("stab")
    if drop_p1 and "p1" in X_df.columns:
        drop_cols.append("p1")

    X_df = X_df.drop(columns=drop_cols, errors="ignore")
    feature_names = list(X_df.columns)

    X = X_df.to_numpy(dtype=np.float32)
    y = ensure_binary_01(y.to_numpy(dtype=np.int64))
    return X, y, feature_names


# ============================================================
# Stratified nested pool selection
# ============================================================

def stratified_nested_train_pools(
    y: np.ndarray,
    train_sizes: List[int],
    max_train_size: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = ensure_binary_01(y)

    train_sizes = sorted(train_sizes)
    if train_sizes[-1] != max_train_size:
        raise ValueError("max_train_size must equal max(train_sizes)")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels; got classes={classes}")

    total_n = len(y)
    idx0 = np.where(y == classes[0])[0]
    idx1 = np.where(y == classes[1])[0]

    max_n0 = int(round(max_train_size * (counts[0] / total_n)))
    max_n1 = max_train_size - max_n0

    pool0 = rng.choice(idx0, size=max_n0, replace=False)
    pool1 = rng.choice(idx1, size=max_n1, replace=False)

    rng.shuffle(pool0)
    rng.shuffle(pool1)

    nested: Dict[int, np.ndarray] = {}
    for n in train_sizes:
        n0 = int(round(n * (counts[0] / total_n)))
        n1 = n - n0
        idx_train = np.concatenate([pool0[:n0], pool1[:n1]])
        rng.shuffle(idx_train)
        nested[n] = idx_train

    return nested


def complement_indices(n_total: int, idx_subset: np.ndarray) -> np.ndarray:
    mask = np.ones(n_total, dtype=bool)
    mask[idx_subset] = False
    return np.where(mask)[0]


# ============================================================
# Scaling
# ============================================================

def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32)


# ============================================================
# Metrics
# ============================================================

def compute_classification_metrics(y_true: np.ndarray, p1: np.ndarray) -> Dict[str, float]:
    p1 = np.clip(np.asarray(p1).reshape(-1), 1e-9, 1 - 1e-9)
    y_pred = (p1 >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, p1)),
    }


def binary_nll_from_probs(y_true: np.ndarray, p1: np.ndarray) -> float:
    p1 = np.clip(np.asarray(p1).reshape(-1), 1e-9, 1 - 1e-9)
    probs = np.stack([1 - p1, p1], axis=1)
    return float(log_loss(y_true, probs))


# ============================================================
# Kernel helpers
# ============================================================

def build_rbf_kernel(
    input_dim: int,
    batch_shape: Optional[torch.Size] = None,
):
    kwargs = {"ard_num_dims": input_dim}
    if batch_shape is not None:
        kwargs["batch_shape"] = batch_shape
    return gpytorch.kernels.RBFKernel(**kwargs)


def build_matern_kernel(
    input_dim: int,
    nu: float = 0.5,
    batch_shape: Optional[torch.Size] = None,
):
    kwargs = {"ard_num_dims": input_dim}
    if batch_shape is not None:
        kwargs["batch_shape"] = batch_shape
    return MaternKernel(nu=nu, **kwargs)


def build_powerexp_kernel(
    input_dim: int,
    learn_power: bool = False,
    fixed_power: float = 1.5,
    batch_shape: Optional[torch.Size] = None,
):
    kwargs = {"ard_num_dims": input_dim}
    if batch_shape is not None:
        kwargs["batch_shape"] = batch_shape

    if learn_power:
        k = PowerExponentialKernel(**kwargs)
        try:
            k.power = fixed_power
        except Exception:
            pass
    else:
        k = PowerExponentialKernelFixed(power=fixed_power, **kwargs)

    return k


def attach_rbf_to_model(
    model: torch.nn.Module,
    input_dim: int,
) -> None:
    if hasattr(model, "covar_module"):
        cm = model.covar_module

        if hasattr(cm, "data_covar_module"):
            batch_shape = getattr(cm.data_covar_module, "batch_shape", torch.Size())
            cm.data_covar_module = build_rbf_kernel(
                input_dim=input_dim,
                batch_shape=batch_shape,
            )
            return

        if hasattr(cm, "base_kernel"):
            batch_shape = getattr(cm.base_kernel, "batch_shape", torch.Size())
            cm.base_kernel = build_rbf_kernel(
                input_dim=input_dim,
                batch_shape=batch_shape,
            )
            return

    if hasattr(model, "data_covar_module"):
        batch_shape = getattr(model.data_covar_module, "batch_shape", torch.Size())
        model.data_covar_module = build_rbf_kernel(
            input_dim=input_dim,
            batch_shape=batch_shape,
        )
        return

    if hasattr(model, "base_kernel"):
        batch_shape = getattr(model.base_kernel, "batch_shape", torch.Size())
        model.base_kernel = build_rbf_kernel(
            input_dim=input_dim,
            batch_shape=batch_shape,
        )
        return

    raise RuntimeError("Could not locate input kernel to replace with RBFKernel.")


def attach_matern_to_model(
    model: torch.nn.Module,
    input_dim: int,
    nu: float = 0.5,
) -> None:
    if hasattr(model, "covar_module"):
        cm = model.covar_module

        if hasattr(cm, "data_covar_module"):
            batch_shape = getattr(cm.data_covar_module, "batch_shape", torch.Size())
            cm.data_covar_module = build_matern_kernel(
                input_dim=input_dim,
                nu=nu,
                batch_shape=batch_shape,
            )
            return

        if hasattr(cm, "base_kernel"):
            batch_shape = getattr(cm.base_kernel, "batch_shape", torch.Size())
            cm.base_kernel = build_matern_kernel(
                input_dim=input_dim,
                nu=nu,
                batch_shape=batch_shape,
            )
            return

    if hasattr(model, "data_covar_module"):
        batch_shape = getattr(model.data_covar_module, "batch_shape", torch.Size())
        model.data_covar_module = build_matern_kernel(
            input_dim=input_dim,
            nu=nu,
            batch_shape=batch_shape,
        )
        return

    if hasattr(model, "base_kernel"):
        batch_shape = getattr(model.base_kernel, "batch_shape", torch.Size())
        model.base_kernel = build_matern_kernel(
            input_dim=input_dim,
            nu=nu,
            batch_shape=batch_shape,
        )
        return

    raise RuntimeError("Could not locate input kernel to replace with MaternKernel.")


def attach_powerexp_to_model(
    model: torch.nn.Module,
    input_dim: int,
    learn_power: bool = False,
    fixed_power: float = 1.5,
) -> None:
    if hasattr(model, "covar_module"):
        cm = model.covar_module

        if hasattr(cm, "data_covar_module"):
            batch_shape = getattr(cm.data_covar_module, "batch_shape", torch.Size())
            cm.data_covar_module = build_powerexp_kernel(
                input_dim=input_dim,
                learn_power=learn_power,
                fixed_power=fixed_power,
                batch_shape=batch_shape,
            )
            return

        if hasattr(cm, "base_kernel"):
            batch_shape = getattr(cm.base_kernel, "batch_shape", torch.Size())
            cm.base_kernel = build_powerexp_kernel(
                input_dim=input_dim,
                learn_power=learn_power,
                fixed_power=fixed_power,
                batch_shape=batch_shape,
            )
            return

    if hasattr(model, "data_covar_module"):
        batch_shape = getattr(model.data_covar_module, "batch_shape", torch.Size())
        model.data_covar_module = build_powerexp_kernel(
            input_dim=input_dim,
            learn_power=learn_power,
            fixed_power=fixed_power,
            batch_shape=batch_shape,
        )
        return

    if hasattr(model, "base_kernel"):
        batch_shape = getattr(model.base_kernel, "batch_shape", torch.Size())
        model.base_kernel = build_powerexp_kernel(
            input_dim=input_dim,
            learn_power=learn_power,
            fixed_power=fixed_power,
            batch_shape=batch_shape,
        )
        return

    raise RuntimeError("Could not locate input kernel to replace with PowerExponentialKernel.")


def get_base_kernel(model: torch.nn.Module):
    if hasattr(model, "covar_module"):
        cm = model.covar_module
        if hasattr(cm, "base_kernel"):
            return cm.base_kernel
        if hasattr(cm, "data_covar_module"):
            return cm.data_covar_module
    if hasattr(model, "base_kernel"):
        return model.base_kernel
    if hasattr(model, "data_covar_module"):
        return model.data_covar_module
    return None


def initialize_rbf_like_fair_start(
    model: torch.nn.Module,
    init_lengthscale: float = 1.0,
    init_outputscale: float = 1.0,
    init_mean_constant: float = 0.0,
) -> None:
    with torch.no_grad():
        bk = get_base_kernel(model)
        if bk is not None and hasattr(bk, "lengthscale"):
            bk.lengthscale = torch.full_like(bk.lengthscale, float(init_lengthscale))

        if hasattr(model, "covar_module") and hasattr(model.covar_module, "outputscale"):
            model.covar_module.outputscale = torch.full_like(
                model.covar_module.outputscale, float(init_outputscale)
            )

        if hasattr(model, "mean_module") and hasattr(model.mean_module, "constant"):
            const = model.mean_module.constant
            const.copy_(torch.full_like(const, float(init_mean_constant)))


def initialize_powerexp_like_working_file(
    model: torch.nn.Module,
    init_power: float = 1.5,
) -> None:
    bk = get_base_kernel(model)
    if bk is not None and hasattr(bk, "power"):
        try:
            with torch.no_grad():
                bk.power = float(init_power)
        except Exception:
            pass


def apply_random_restart_perturbation(
    model: torch.nn.Module,
    restart_seed: int,
    is_powerexp: bool = False,
) -> None:
    rng = np.random.default_rng(restart_seed)

    with torch.no_grad():
        bk = get_base_kernel(model)

        if bk is not None and hasattr(bk, "lengthscale"):
            ls = bk.lengthscale.detach().clone()
            mult = np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=ls.numel()))
            mult_t = torch.tensor(mult, dtype=ls.dtype, device=ls.device).reshape_as(ls)
            bk.lengthscale.copy_(mult_t)

        if hasattr(model, "covar_module") and hasattr(model.covar_module, "outputscale"):
            oscale = model.covar_module.outputscale.detach().clone()
            mult = np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=oscale.numel()))
            mult_t = torch.tensor(mult, dtype=oscale.dtype, device=oscale.device).reshape_as(oscale)
            model.covar_module.outputscale.copy_(mult_t)

        if hasattr(model, "mean_module") and hasattr(model.mean_module, "constant"):
            const = model.mean_module.constant.detach().clone()
            vals = rng.normal(loc=0.0, scale=1.0, size=const.numel())
            vals_t = torch.tensor(vals, dtype=const.dtype, device=const.device).reshape_as(const)
            model.mean_module.constant.copy_(vals_t)

        if is_powerexp and bk is not None and hasattr(bk, "power"):
            try:
                p = float(rng.uniform(0.5, 1.95))
                bk.power = p
            except Exception:
                pass


def extract_trace_row(model: torch.nn.Module, loss_value: float, iteration: int) -> dict:
    row = {"epoch": iteration, "loss": float(loss_value)}
    try:
        if hasattr(model, "covar_module") and hasattr(model.covar_module, "outputscale"):
            outscale = model.covar_module.outputscale.detach().cpu().reshape(-1).numpy().tolist()
            row["outputscale"] = outscale if len(outscale) > 1 else outscale[0]

        bk = get_base_kernel(model)

        if bk is not None and hasattr(bk, "power"):
            p = bk.power.detach().cpu().reshape(-1).numpy().tolist()
            row["power"] = p if len(p) > 1 else p[0]

        if bk is not None and hasattr(bk, "lengthscale"):
            ls = bk.lengthscale.detach().cpu().reshape(-1).numpy().tolist()
            row["lengthscales"] = ls

        if bk is not None and hasattr(bk, "nu"):
            row["nu"] = float(bk.nu)
    except Exception:
        pass
    return row


# ============================================================
# Kernel Flows helper for GPClassifier batched outputs
# ============================================================

def compute_gpclassifier_kf_metric(
    model,
    train_x: torch.Tensor,
    train_y_CxN: torch.Tensor,
    *,
    Nf: int,
    cholesky_jitter: Optional[float] = None,
    seed: Optional[int] = None,
    likelihood=None,
) -> float:
    """
    KF metric adapted for the GPClassifier Dirichlet-transformed batched latent outputs.

    We compute rho separately for each latent/task batch and average:
        rho_c = 1 - (v_c^T K_sub^{-1} v_c) / (u_c^T K_full^{-1} u_c)

    using:
        residual_c = y_c - mean_c

    Args:
        model: trained GPClassifier-like exact GP model
        train_x: (N, D)
        train_y_CxN: transformed training targets, shape (C, N)
        Nf: subset size
        cholesky_jitter: optional jitter
        seed: subset sampling seed
        likelihood: optional likelihood, only used to inspect noise if available

    Returns:
        mean KF across latent tasks, or NaN on failure.
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        N = train_x.size(0)
        if Nf >= N or Nf < 1:
            return float("nan")
        if N > 2000:
            return float("nan")

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            prior = model(train_x)
            mean = prior.mean  # expected shape (C, N) for batch exact GP
            cov = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
            if cov is None:
                return float("nan")

            K_prior_dense = cov.to_dense()

            if mean.ndim != 2:
                return float("nan")

            C, N_mean = mean.shape
            if N_mean != N:
                return float("nan")

            if train_y_CxN.shape != mean.shape:
                return float("nan")

            # Infer a scalar/batch noise if possible, otherwise tiny fallback
            # For KF monitoring, a stable approximation is sufficient.
            noise_batch = torch.full((C,), 1e-6, dtype=mean.dtype, device=mean.device)
            if likelihood is not None and hasattr(likelihood, "noise"):
                try:
                    n = likelihood.noise.detach().to(mean.device, mean.dtype)
                    if n.numel() == 1:
                        noise_batch[:] = max(float(n.item()), 1e-12)
                    else:
                        flat = n.reshape(-1)
                        if flat.numel() >= C:
                            for c in range(C):
                                noise_batch[c] = max(float(flat[c].item()), 1e-12)
                        else:
                            noise_batch[:] = max(float(flat[0].item()), 1e-12)
                except Exception:
                    pass

            gen = torch.Generator(device=train_x.device)
            if seed is not None:
                gen.manual_seed(seed)
            inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]

            rhos = []
            eye_full = torch.eye(N, device=mean.device, dtype=mean.dtype)

            # K_prior_dense may be (C, N, N) for batched latent functions.
            if K_prior_dense.ndim != 3 or K_prior_dense.shape[0] != C:
                return float("nan")

            for c in range(C):
                r = (train_y_CxN[c] - mean[c]).view(-1)
                K_dense = K_prior_dense[c] + noise_batch[c] * eye_full

                K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
                u_sq = (r * K_inv_r).sum().item()
                if not np.isfinite(u_sq) or u_sq <= 0:
                    continue

                r_sub = r[inds]
                K_sub = K_dense[inds][:, inds]
                K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
                v_sq = (r_sub * K_sub_inv_r).sum().item()
                if not np.isfinite(v_sq):
                    continue

                rho = 1.0 - (v_sq / u_sq)
                if np.isfinite(rho):
                    rhos.append(float(rho))

            if len(rhos) == 0:
                return float("nan")
            return float(np.mean(rhos))

    except Exception:
        return float("nan")


# ============================================================
# GP restart helper
# ============================================================

def train_gp_model_with_restarts(
    *,
    model_builder,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    X_test_t: torch.Tensor,
    alpha_epsilon: float,
    verbose: bool,
    num_classes: int,
    base_seed: int,
    n_restarts: int,
    callback_save_path: Optional[str],
    print_model_once: bool,
    print_header: str,
    predict_fn,
    is_powerexp: bool = False,
    extra_trainer_info: Optional[dict] = None,
    jitter_schedule: Optional[List[float]] = None,
    compute_kf: bool = False,
    kf_subset_size: int = 32,
) -> Tuple[np.ndarray, float, float, dict]:
    if extra_trainer_info is None:
        extra_trainer_info = {}
    if jitter_schedule is None:
        jitter_schedule = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    _, transformed_targets, C = prepare_dirichlet_targets(
        y_t,
        alpha_epsilon=alpha_epsilon,
        dtype=y_t.dtype if y_t.is_floating_point() else X_t.dtype,
    )
    if C != num_classes:
        raise RuntimeError(f"prepare_dirichlet_targets num_classes={C}, expected {num_classes}")

    y_CxN = transformed_targets.t().contiguous()

    best = None
    best_state_dict = None
    all_restart_histories = {}
    all_restart_summaries = []

    total_train_time = 0.0
    printed = False
    n_successful = 0
    n_failed = 0

    for restart_idx in range(n_restarts):
        restart_seed = 100000 * base_seed + restart_idx + 1
        set_all_seeds(restart_seed)

        restart_success = False
        last_error = None

        for jitter in jitter_schedule:
            try:
                model, likelihood = model_builder(X_t, y_t)
                model.set_train_data(inputs=X_t, targets=y_CxN, strict=False)

                apply_random_restart_perturbation(
                    model,
                    restart_seed=restart_seed,
                    is_powerexp=is_powerexp,
                )

                if print_model_once and (not printed):
                    print(f"\n[{print_header}]")
                    print(model)
                    printed = True

                model.train()
                likelihood.train()

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                iteration_rows = []

                def iter_cb(iteration, loss, flat_params=None):
                    loss_f = float(loss)
                    iteration_rows.append(extract_trace_row(model, loss_f, iteration))
                    if verbose and (iteration % 25 == 0):
                        print(
                            f"[{print_header}] restart {restart_idx + 1:02d}/{n_restarts} "
                            f"jitter={jitter:.0e} iter {iteration:4d} loss={loss_f:.6f}"
                        )

                opt = LBFGSScipy(
                    model.parameters(),
                    iteration_callback=iter_cb,
                )

                def closure():
                    opt.zero_grad()
                    out = model(X_t)
                    raw_loss = -mll(out, y_CxN)
                    loss = raw_loss.sum() if raw_loss.ndim > 0 else raw_loss

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        raise RuntimeError("Loss became NaN/Inf during optimization.")

                    loss.backward()

                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                raise RuntimeError("Gradient became NaN/Inf during optimization.")
                    return loss

                t0 = time.time()
                with gpytorch.settings.cholesky_jitter(jitter):
                    final_loss = opt.step(closure)
                total_train_time += (time.time() - t0)

                final_train_nll = float(final_loss.item()) if torch.is_tensor(final_loss) else float(final_loss)
                if not np.isfinite(final_train_nll):
                    raise RuntimeError("Final training NLL was not finite.")

                restart_summary = {
                    "restart_idx": restart_idx,
                    "restart_seed": restart_seed,
                    "jitter_used": jitter,
                    "final_train_nll": final_train_nll,
                    "lbfgs_stop_reason": getattr(opt, "_lbfgs_stop_reason", None),
                    "lbfgs_info": getattr(opt, "_lbfgs_info", None),
                    "lbfgs_iterations": getattr(opt, "_n_iter", None),
                    "status": "success",
                }
                all_restart_summaries.append(restart_summary)
                all_restart_histories[str(restart_idx)] = iteration_rows
                n_successful += 1
                restart_success = True

                if (best is None) or (final_train_nll < best["final_train_nll"]):
                    best = restart_summary
                    best_state_dict = copy.deepcopy(model.state_dict())

                break

            except Exception as exc:
                last_error = exc
                if is_numerical_gp_failure(exc):
                    if verbose:
                        print(
                            f"[{print_header}] restart {restart_idx + 1:02d}/{n_restarts} "
                            f"failed at jitter={jitter:.0e}; retrying. Error: {exc}"
                        )
                    continue
                else:
                    if verbose:
                        print(
                            f"[{print_header}] restart {restart_idx + 1:02d}/{n_restarts} "
                            f"non-numerical failure at jitter={jitter:.0e}. Error: {exc}"
                        )
                    break

        if not restart_success:
            n_failed += 1
            fail_summary = {
                "restart_idx": restart_idx,
                "restart_seed": restart_seed,
                "status": "failed",
                "error": None if last_error is None else str(last_error),
            }
            all_restart_summaries.append(fail_summary)
            all_restart_histories[str(restart_idx)] = []

    if best is None or best_state_dict is None:
        raise RuntimeError(
            f"All {n_restarts} restarts failed for model: {print_header}. "
            f"Try increasing jitter or narrowing initialization ranges."
        )

    set_all_seeds(best["restart_seed"])
    best_model, best_likelihood = model_builder(X_t, y_t)
    best_model.set_train_data(inputs=X_t, targets=y_CxN, strict=False)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()
    best_likelihood.eval()

    final_kf = float("nan")
    if compute_kf:
        final_kf = compute_gpclassifier_kf_metric(
            best_model,
            train_x=X_t,
            train_y_CxN=y_CxN,
            Nf=min(kf_subset_size, max(1, X_t.size(0) - 1)),
            cholesky_jitter=best["jitter_used"],
            seed=best["restart_seed"],
            likelihood=best_likelihood,
        )

    t1 = time.time()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(best["jitter_used"]):
        probs = predict_fn(best_model, X_test_t)
    infer_time = time.time() - t1

    p1 = probs[:, 1].detach().cpu().numpy()
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)

    trainer_info = {
        "runs": all_restart_histories,
        "n_restarts": n_restarts,
        "n_successful_restarts": n_successful,
        "n_failed_restarts": n_failed,
        "selected_restart_idx": best["restart_idx"],
        "selected_restart_seed": best["restart_seed"],
        "selected_jitter": best["jitter_used"],
        "lbfgs_stop_reason": best["lbfgs_stop_reason"],
        "lbfgs_info": best["lbfgs_info"],
        "lbfgs_iterations": best["lbfgs_iterations"],
        "final_train_nll": best["final_train_nll"],
        "final_kf": final_kf,
        "all_restart_summaries": all_restart_summaries,
        **extra_trainer_info,
    }

    if callback_save_path is not None:
        ensure_dir(callback_save_path)
        save_json({"runs": all_restart_histories}, os.path.join(callback_save_path, "epoch_parameters.json"))
        save_json(trainer_info, os.path.join(callback_save_path, "trainer_info.json"))

    return p1, total_train_time, infer_time, trainer_info


# ============================================================
# Model interface
# ============================================================

class BinaryModel:
    name: str

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        callback_save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Optional[dict]]:
        raise NotImplementedError


# ============================================================
# GP+ GPClassifier with RBF input kernel + multi-start LBFGS
# ============================================================

class GPPlusGPClassifierRBF(BinaryModel):
    def __init__(
        self,
        alpha_epsilon: float = 0.01,
        verbose: bool = False,
        num_classes: int = 2,
        print_kernel_once: bool = True,
        use_fair_init: bool = True,
        n_restarts: int = 16,
        kf_subset_size: int = 32,
    ):
        self.name = "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)"
        self.alpha_epsilon = alpha_epsilon
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_kernel_once = print_kernel_once
        self.use_fair_init = use_fair_init
        self.n_restarts = int(n_restarts)
        self.kf_subset_size = int(kf_subset_size)
        self.device = torch.device("cpu")

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        callback_save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Optional[dict]]:
        y_train = ensure_binary_01(y_train)

        X_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        def model_builder(train_x, train_y):
            model = GPClassifier(
                train_x=train_x,
                train_y=train_y,
                alpha_epsilon=self.alpha_epsilon,
            ).to(self.device)

            attach_rbf_to_model(model, input_dim=X_train.shape[1])

            if self.use_fair_init:
                initialize_rbf_like_fair_start(model)

            likelihood = model.likelihood
            return model, likelihood

        def predict_fn(model, X_test_t_local):
            out = model.predict(X_test_t_local, return_std=False)
            if isinstance(out, dict) and "probs" in out:
                probs = out["probs"]
            else:
                if not (isinstance(out, dict) and "logits" in out):
                    raise KeyError("GPClassifier.predict missing 'probs' and 'logits'.")
                logits = out["logits"]
                probs = torch.softmax(logits.t(), dim=-1)
            return probs

        return train_gp_model_with_restarts(
            model_builder=model_builder,
            X_t=X_t,
            y_t=y_t,
            X_test_t=X_test_t,
            alpha_epsilon=self.alpha_epsilon,
            verbose=self.verbose,
            num_classes=self.num_classes,
            base_seed=seed,
            n_restarts=self.n_restarts,
            callback_save_path=callback_save_path,
            print_model_once=self.print_kernel_once,
            print_header="GP+ GPClassifier RBF multi-start LBFGS",
            predict_fn=predict_fn,
            is_powerexp=False,
            compute_kf=True,
            kf_subset_size=self.kf_subset_size,
        )


# ============================================================
# GP+ GPClassifier with Matérn(0.5) input kernel + multi-start LBFGS
# ============================================================

class GPPlusGPClassifierMatern(BinaryModel):
    def __init__(
        self,
        alpha_epsilon: float = 0.01,
        verbose: bool = False,
        num_classes: int = 2,
        print_kernel_once: bool = True,
        use_fair_init: bool = True,
        nu: float = 0.5,
        n_restarts: int = 16,
        kf_subset_size: int = 32,
    ):
        self.name = f"GP+ GPClassifier (Matern, nu={nu}, ARD, LBFGS-defaults)"
        self.alpha_epsilon = alpha_epsilon
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_kernel_once = print_kernel_once
        self.use_fair_init = use_fair_init
        self.nu = float(nu)
        self.n_restarts = int(n_restarts)
        self.kf_subset_size = int(kf_subset_size)
        self.device = torch.device("cpu")

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        callback_save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Optional[dict]]:
        y_train = ensure_binary_01(y_train)

        X_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        def model_builder(train_x, train_y):
            model = GPClassifier(
                train_x=train_x,
                train_y=train_y,
                alpha_epsilon=self.alpha_epsilon,
            ).to(self.device)

            attach_matern_to_model(
                model,
                input_dim=X_train.shape[1],
                nu=self.nu,
            )

            if self.use_fair_init:
                initialize_rbf_like_fair_start(model)

            likelihood = model.likelihood
            return model, likelihood

        def predict_fn(model, X_test_t_local):
            out = model.predict(X_test_t_local, return_std=False)
            if isinstance(out, dict) and "probs" in out:
                probs = out["probs"]
            else:
                if not (isinstance(out, dict) and "logits" in out):
                    raise KeyError("GPClassifier.predict missing 'probs' and 'logits'.")
                logits = out["logits"]
                probs = torch.softmax(logits.t(), dim=-1)
            return probs

        return train_gp_model_with_restarts(
            model_builder=model_builder,
            X_t=X_t,
            y_t=y_t,
            X_test_t=X_test_t,
            alpha_epsilon=self.alpha_epsilon,
            verbose=self.verbose,
            num_classes=self.num_classes,
            base_seed=seed,
            n_restarts=self.n_restarts,
            callback_save_path=callback_save_path,
            print_model_once=self.print_kernel_once,
            print_header=f"GP+ GPClassifier Matern nu={self.nu} multi-start LBFGS",
            predict_fn=predict_fn,
            is_powerexp=False,
            extra_trainer_info={"nu": self.nu},
            compute_kf=True,
            kf_subset_size=self.kf_subset_size,
        )


# ============================================================
# GP+ GPClassifier with PowerExp input kernel + multi-start LBFGS
# ============================================================

class GPPlusGPClassifierPowerExp(BinaryModel):
    def __init__(
        self,
        alpha_epsilon: float = 0.01,
        verbose: bool = False,
        num_classes: int = 2,
        print_kernel_once: bool = True,
        learn_power: bool = True,
        fixed_power: float = 1.5,
        use_working_powerexp_init: bool = True,
        debug_prediction_spread: bool = False,
        n_restarts: int = 16,
        kf_subset_size: int = 32,
    ):
        self.name = f"GP+ GPClassifier (PowerExp, p={'learned' if learn_power else fixed_power}, LBFGS-defaults)"
        self.alpha_epsilon = alpha_epsilon
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_kernel_once = print_kernel_once
        self.learn_power = learn_power
        self.fixed_power = fixed_power
        self.use_working_powerexp_init = use_working_powerexp_init
        self.debug_prediction_spread = debug_prediction_spread
        self.n_restarts = int(n_restarts)
        self.kf_subset_size = int(kf_subset_size)
        self.device = torch.device("cpu")

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        callback_save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Optional[dict]]:
        y_train = ensure_binary_01(y_train)

        X_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        def model_builder(train_x, train_y):
            model = GPClassifier(
                train_x=train_x,
                train_y=train_y,
                alpha_epsilon=self.alpha_epsilon,
            ).to(self.device)

            attach_powerexp_to_model(
                model,
                input_dim=X_train.shape[1],
                learn_power=self.learn_power,
                fixed_power=self.fixed_power,
            )

            if self.use_working_powerexp_init:
                initialize_powerexp_like_working_file(
                    model,
                    init_power=self.fixed_power,
                )

            likelihood = model.likelihood
            return model, likelihood

        def predict_fn(model, X_test_t_local):
            out = model.predict(X_test_t_local, return_std=False)
            if isinstance(out, dict) and "probs" in out:
                probs = out["probs"]
            else:
                if not (isinstance(out, dict) and "logits" in out):
                    raise KeyError("GPClassifier.predict missing 'probs' and 'logits'.")
                logits = out["logits"]
                probs = torch.softmax(logits.t(), dim=-1)
            return probs

        p1, train_time, infer_time, trainer_info = train_gp_model_with_restarts(
            model_builder=model_builder,
            X_t=X_t,
            y_t=y_t,
            X_test_t=X_test_t,
            alpha_epsilon=self.alpha_epsilon,
            verbose=self.verbose,
            num_classes=self.num_classes,
            base_seed=seed,
            n_restarts=self.n_restarts,
            callback_save_path=callback_save_path,
            print_model_once=self.print_kernel_once,
            print_header="GP+ GPClassifier PowerExp multi-start LBFGS",
            predict_fn=predict_fn,
            is_powerexp=True,
            compute_kf=True,
            kf_subset_size=self.kf_subset_size,
        )

        if self.debug_prediction_spread:
            print(
                f"[PowerExp debug] seed={seed} "
                f"p1_mean={p1.mean():.6f} p1_std={p1.std():.6f} "
                f"p1_min={p1.min():.6f} p1_max={p1.max():.6f}"
            )
            print(f"[PowerExp debug] selected restart = {trainer_info.get('selected_restart_idx')}")
            print(f"[PowerExp debug] stop_reason = {trainer_info.get('lbfgs_stop_reason')}")

        return p1, train_time, infer_time, trainer_info


# ============================================================
# TabPFN helpers
# ============================================================

def batched_predict_proba_sklearn(estimator, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    X = np.asarray(X)
    n = len(X)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    chunks = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunks.append(np.asarray(estimator.predict_proba(X[start:stop]), dtype=np.float64))
    return np.vstack(chunks)


# ============================================================
# TabPFN binary classifier models
# ============================================================

class TabPFNBinaryClassifier(BinaryModel):
    def __init__(
        self,
        version: str = "v2.5",
        device: Optional[str] = None,
        batch_size_predict: int = 2048,
    ):
        if version not in {"v2.5", "v2"}:
            raise ValueError("version must be one of {'v2.5', 'v2'}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version = version
        self.device = device
        self.batch_size_predict = int(batch_size_predict)
        self.name = (
            "TabPFN Classifier (v2.5 default)"
            if version == "v2.5"
            else "TabPFN Classifier (v2 compatibility)"
        )

    def _make_estimator(self):
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "tabpfn import failed. Original import error:\n"
                f"{TABPFN_IMPORT_ERROR}"
            )

        if self.version == "v2.5":
            return TabPFNClassifier(device=self.device)
        else:
            return TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                device=self.device,
            )

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        callback_save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Optional[dict]]:
        set_all_seeds(seed)
        y_train = ensure_binary_01(y_train)

        clf = self._make_estimator()

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        train_probs = batched_predict_proba_sklearn(
            clf,
            X_train,
            batch_size=self.batch_size_predict,
        )
        train_p1 = np.clip(train_probs[:, 1], 1e-9, 1 - 1e-9)
        train_nll_eval = binary_nll_from_probs(y_train, train_p1)

        t1 = time.time()
        test_probs = batched_predict_proba_sklearn(
            clf,
            X_test,
            batch_size=self.batch_size_predict,
        )
        infer_time = time.time() - t1

        p1 = np.clip(test_probs[:, 1], 1e-9, 1 - 1e-9)

        trainer_info = {
            "tabpfn_version": self.version,
            "tabpfn_device": self.device,
            "final_train_nll": float(train_nll_eval),
            "final_kf": float("nan"),
            "lbfgs_stop_reason": None,
            "lbfgs_info": None,
            "lbfgs_iterations": None,
            "n_restarts": 1,
            "n_successful_restarts": 1,
            "n_failed_restarts": 0,
            "selected_restart_idx": 0,
            "selected_restart_seed": seed,
        }

        if callback_save_path is not None:
            ensure_dir(callback_save_path)
            save_json(trainer_info, os.path.join(callback_save_path, "trainer_info.json"))

        return p1, train_time, infer_time, trainer_info


# ============================================================
# Callback JSON parsing and averaged plots
# ============================================================

def _flatten_numeric(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, list):
        vals = [float(v) for v in x if isinstance(v, (int, float, np.integer, np.floating))]
        return float(np.mean(vals)) if vals else np.nan
    return np.nan


def _extract_epoch_records_from_json(obj: Any) -> List[dict]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]

    if isinstance(obj, dict):
        candidate_keys = [
            "records",
            "history",
            "epochs",
            "epoch_records",
            "parameters",
            "data",
            "stored_params",
            "params",
            "epoch_params",
        ]
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], list):
                return [r for r in obj[k] if isinstance(r, dict)]

        if "runs" in obj and isinstance(obj["runs"], dict):
            recs = []
            for run_id, run_data in obj["runs"].items():
                if isinstance(run_data, list):
                    for r in run_data:
                        if isinstance(r, dict):
                            rr = dict(r)
                            rr["run"] = run_id
                            recs.append(rr)
                elif isinstance(run_data, dict):
                    for _, maybe_val in run_data.items():
                        if isinstance(maybe_val, list):
                            for r in maybe_val:
                                if isinstance(r, dict):
                                    rr = dict(r)
                                    rr["run"] = run_id
                                    recs.append(rr)
            return recs

        if all(isinstance(v, dict) for v in obj.values()):
            recs = []
            for k, v in obj.items():
                rec = dict(v)
                if "epoch" not in rec:
                    try:
                        rec["epoch"] = int(k)
                    except Exception:
                        pass
                recs.append(rec)
            return recs

    return []


def load_epoch_parameter_history(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    recs = _extract_epoch_records_from_json(obj)
    if not recs:
        return pd.DataFrame()

    rows = []
    for r in recs:
        epoch = r.get("epoch", r.get("Epoch", r.get("iteration", r.get("iter", np.nan))))
        loss = r.get("loss", r.get("Loss", r.get("best_loss", np.nan)))
        outputscale = r.get("outputscale", r.get("Outputscale", r.get("output_scale", np.nan)))
        power = r.get("power", r.get("Power", np.nan))
        nu = r.get("nu", r.get("Nu", np.nan))

        ls = (
            r.get("lengthscales", None)
            or r.get("lengthscale", None)
            or r.get("Lengthscales", None)
            or r.get("cont_lengthscales", None)
        )

        if isinstance(ls, dict):
            ls_vals = []
            for v in ls.values():
                if isinstance(v, list):
                    ls_vals.extend(v)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    ls_vals.append(float(v))
            ls = ls_vals

        if isinstance(ls, (int, float, np.integer, np.floating)):
            ls = [float(ls)]

        if isinstance(ls, list) and len(ls) > 0:
            ls_arr = np.asarray(ls, dtype=float).reshape(-1)
            ls_mean = float(np.mean(ls_arr))
            ls_min = float(np.min(ls_arr))
            ls_max = float(np.max(ls_arr))
        else:
            ls_mean = np.nan
            ls_min = np.nan
            ls_max = np.nan

        rows.append({
            "epoch": pd.to_numeric(epoch, errors="coerce"),
            "loss": pd.to_numeric(loss, errors="coerce"),
            "outputscale": _flatten_numeric(outputscale),
            "power": _flatten_numeric(power),
            "nu": _flatten_numeric(nu),
            "lengthscale_mean": ls_mean,
            "lengthscale_min": ls_min,
            "lengthscale_max": ls_max,
        })

    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def aggregate_and_plot_histories(
    callback_root: str,
    out_dir: str,
    train_sizes: List[int],
) -> List[str]:
    ensure_dir(out_dir)
    saved = []

    entries = []
    for root, _, files in os.walk(callback_root):
        if "epoch_parameters.json" not in files:
            continue
        json_path = os.path.join(root, "epoch_parameters.json")
        base = os.path.basename(root)

        seed = None
        train_size = None
        try:
            parts = base.split("_")
            for p in parts:
                if p.startswith("seed"):
                    seed = int(p.replace("seed", ""))
                if p.startswith("train"):
                    train_size = int(p.replace("train", ""))
        except Exception:
            pass

        df = load_epoch_parameter_history(json_path)
        if df.empty:
            continue

        model_name = base
        if "_seed" in base:
            model_name = base.split("_seed")[0]

        df["model_key"] = model_name
        df["seed"] = seed
        df["train_size"] = train_size
        entries.append(df)

    if not entries:
        print("[aggregate_and_plot_histories] No epoch_parameter histories found.")
        return saved

    all_df = pd.concat(entries, ignore_index=True)

    metrics = [
        ("loss", "Loss"),
        ("outputscale", "Outputscale"),
        ("lengthscale_mean", "Mean Lengthscale"),
        ("power", "Learned Power"),
    ]

    for model_key in sorted(all_df["model_key"].dropna().unique()):
        dfm = all_df[all_df["model_key"] == model_key].copy()

        for col, title in metrics:
            if col not in dfm.columns or dfm[col].isna().all():
                continue

            grouped = (
                dfm.groupby(["train_size", "epoch"], as_index=False)[col]
                   .mean()
            )

            plt.figure(figsize=(8, 5))
            plotted = False

            for n_train in sorted(train_sizes):
                dft = grouped[grouped["train_size"] == n_train]
                if dft.empty:
                    continue
                plt.plot(dft["epoch"], dft[col], label=f"train={n_train}")
                plotted = True

            if not plotted:
                plt.close()
                continue

            plt.xlabel("LBFGS iteration")
            plt.ylabel(col)
            plt.title(f"{title} — {model_key}")
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"{model_key}_{col}_avg.png")
            plt.savefig(out_path, dpi=180)
            plt.close()
            saved.append(out_path)

    return saved


# ============================================================
# Paper figure
# ============================================================

def plot_paper_performance_figure(
    summary_df: pd.DataFrame,
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """
    Create a 1x3 paper-style summary figure with:
      - x-axis: numeric train size with proportional spacing
      - left:   final_train_nll_mean (GP+ models only)
      - middle: accuracy_mean (GP+ models + PFN 2.5 + PFN 2.0)
      - right:  brier_mean (GP+ models + PFN 2.5 + PFN 2.0)
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty; cannot create paper figure.")

    ensure_dir(os.path.dirname(out_path) or ".")

    df = summary_df.copy()
    df = df.sort_values(["model", "train_size"]).reset_index(drop=True)

    pretty_name = {
        "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)": "GP+",
        "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)": "GP+ (PE)",
        "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)": "GP+ (Matern)",
        "TabPFN Classifier (v2.5 default)": "PFN 2.5",
        "TabPFN Classifier (v2 compatibility)": "PFN 2.0",
    }

    train_nll_models = [
        "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)",
        "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)",
        "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)",
    ]

    test_metric_models = [
        "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)",
        "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)",
        "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)",
        "TabPFN Classifier (v2.5 default)",
        "TabPFN Classifier (v2 compatibility)",
    ]

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    all_models_for_markers = list(dict.fromkeys(train_nll_models + test_metric_models))
    model_to_marker = {
        model: marker_cycle[i % len(marker_cycle)]
        for i, model in enumerate(all_models_for_markers)
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)
    train_sizes = sorted(df["train_size"].dropna().unique())

    # Left: Train NLL (GP+ only)
    ax = axes[0]
    for model in train_nll_models:
        dfi = df[df["model"] == model].sort_values("train_size")
        if dfi.empty:
            continue
        ax.plot(
            dfi["train_size"],
            dfi["final_train_nll_mean"],
            linestyle=":",
            marker=model_to_marker[model],
            linewidth=2.0,
            markersize=8,
            label=pretty_name.get(model, model),
        )
    ax.set_xlabel("Train Size")
    ax.set_ylabel("Train Negative Log Likelihood")
    ax.set_title("Train Negative Log Likelihood")
    ax.set_xticks(train_sizes)
    ax.set_xlim(min(train_sizes) * 0.9, max(train_sizes) * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    # Middle: Test Accuracy (GP+ + PFNs)
    ax = axes[1]
    acc_handles = []
    acc_labels = []
    for model in test_metric_models:
        dfi = df[df["model"] == model].sort_values("train_size")
        if dfi.empty:
            continue
        line, = ax.plot(
            dfi["train_size"],
            dfi["accuracy_mean"],
            linestyle=":",
            marker=model_to_marker[model],
            linewidth=2.0,
            markersize=8,
            label=pretty_name.get(model, model),
        )
        acc_handles.append(line)
        acc_labels.append(pretty_name.get(model, model))
    ax.set_xlabel("Train Size")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy")
    ax.set_xticks(train_sizes)
    ax.set_xlim(min(train_sizes) * 0.9, max(train_sizes) * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(
        acc_handles,
        acc_labels,
        loc="lower right",
        frameon=True,
    )

    # Right: Test Brier (GP+ + PFNs)
    ax = axes[2]
    for model in test_metric_models:
        dfi = df[df["model"] == model].sort_values("train_size")
        if dfi.empty:
            continue
        ax.plot(
            dfi["train_size"],
            dfi["brier_mean"],
            linestyle=":",
            marker=model_to_marker[model],
            linewidth=2.0,
            markersize=8,
            label=pretty_name.get(model, model),
        )
    ax.set_xlabel("Train Size")
    ax.set_ylabel("Test Brier Score")
    ax.set_title("Test Brier Score")
    ax.set_xticks(train_sizes)
    ax.set_xlim(min(train_sizes) * 0.9, max(train_sizes) * 1.05)
    ax.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title, y=1.04, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gp_kf_figure(
    summary_df: pd.DataFrame,
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """
    Create a GP-only Kernel Flows figure with:
      - x-axis: numeric train size with proportional spacing
      - y-axis: mean KF
      - models: GP+, GP+ (PE), GP+ (Matern)
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty; cannot create KF figure.")

    ensure_dir(os.path.dirname(out_path) or ".")

    df = summary_df.copy()
    df = df.sort_values(["model", "train_size"]).reset_index(drop=True)

    pretty_name = {
        "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)": "GP+",
        "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)": "GP+ (PE)",
        "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)": "GP+ (Matern)",
    }

    gp_models = [
        "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)",
        "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)",
        "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)",
    ]

    marker_cycle = ["o", "s", "^"]
    model_to_marker = {m: marker_cycle[i] for i, m in enumerate(gp_models)}

    train_sizes = sorted(df["train_size"].dropna().unique())

    plt.figure(figsize=(6.8, 4.8))
    plotted_any = False

    for model in gp_models:
        dfi = df[df["model"] == model].sort_values("train_size")
        if dfi.empty or "final_kf_mean" not in dfi.columns:
            continue

        plt.plot(
            dfi["train_size"],
            dfi["final_kf_mean"],
            linestyle=":",
            marker=model_to_marker[model],
            linewidth=2.0,
            markersize=8,
            label=pretty_name.get(model, model),
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No GP KF data available to plot.")

    plt.xlabel("Train Size")
    plt.ylabel("Kernel Flows")
    plt.title("Kernel Flows")
    plt.xticks(train_sizes)
    plt.xlim(min(train_sizes) * 0.9, max(train_sizes) * 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)

    if title:
        plt.suptitle(title, y=1.02, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================
# Experiment runner
# ============================================================

def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    models: List[BinaryModel],
    train_sizes: List[int],
    seeds: List[int],
    max_train_size: int = 500,
    callback_root: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    n_total = len(y)
    train_sizes_sorted = sorted(train_sizes)

    if max(train_sizes) != max_train_size:
        raise ValueError("max_train_size must equal max(train_sizes)")

    for seed in seeds:
        nested_train = stratified_nested_train_pools(
            y=y,
            train_sizes=train_sizes_sorted,
            max_train_size=max_train_size,
            seed=seed,
        )

        idx_pool = nested_train[max_train_size]
        idx_test = complement_indices(n_total, idx_pool)

        X_test_raw, y_test = X[idx_test], y[idx_test]
        y_test = ensure_binary_01(y_test)

        for n_train in train_sizes_sorted:
            idx_train = nested_train[n_train]

            X_train_raw, y_train = X[idx_train], y[idx_train]
            y_train = ensure_binary_01(y_train)

            X_train_s, X_test_s = standardize_train_test(X_train_raw, X_test_raw)

            for model in models:
                cb_dir = None
                if callback_root is not None:
                    cb_dir = os.path.join(
                        callback_root,
                        f"{sanitize_filename(model.name)}_seed{seed}_train{n_train}"
                    )
                    ensure_dir(cb_dir)

                if isinstance(model, TabPFNBinaryClassifier):
                    fit_X_train = X_train_raw
                    fit_X_test = X_test_raw
                else:
                    fit_X_train = X_train_s
                    fit_X_test = X_test_s

                try:
                    p1, train_time, infer_time, trainer_info = model.fit_predict(
                        X_train=fit_X_train,
                        y_train=y_train,
                        X_test=fit_X_test,
                        y_test=y_test,
                        seed=seed,
                        callback_save_path=cb_dir,
                    )

                    m = compute_classification_metrics(y_test, p1)

                    row = {
                        "model": model.name,
                        "train_size": n_train,
                        "train_pool_size": max_train_size,
                        "test_size": len(y_test),
                        "seed": seed,
                        "train_time_s": train_time,
                        "infer_time_s": infer_time,
                        "n_restarts": None if trainer_info is None else trainer_info.get("n_restarts"),
                        "n_successful_restarts": None if trainer_info is None else trainer_info.get("n_successful_restarts"),
                        "n_failed_restarts": None if trainer_info is None else trainer_info.get("n_failed_restarts"),
                        "selected_restart_idx": None if trainer_info is None else trainer_info.get("selected_restart_idx"),
                        "lbfgs_stop_reason": None if trainer_info is None else trainer_info.get("lbfgs_stop_reason"),
                        "lbfgs_iterations": None if trainer_info is None else trainer_info.get("lbfgs_iterations"),
                        "final_train_nll": None if trainer_info is None else trainer_info.get("final_train_nll"),
                        "final_kf": None if trainer_info is None else trainer_info.get("final_kf"),
                        **m,
                    }
                    rows.append(row)
                    print(row)

                except Exception as exc:
                    fail_row = {
                        "model": model.name,
                        "train_size": n_train,
                        "train_pool_size": max_train_size,
                        "test_size": len(y_test),
                        "seed": seed,
                        "train_time_s": np.nan,
                        "infer_time_s": np.nan,
                        "n_restarts": getattr(model, "n_restarts", np.nan),
                        "n_successful_restarts": 0,
                        "n_failed_restarts": getattr(model, "n_restarts", np.nan),
                        "selected_restart_idx": np.nan,
                        "lbfgs_stop_reason": "model_failed",
                        "lbfgs_iterations": np.nan,
                        "final_train_nll": np.nan,
                        "final_kf": np.nan,
                        "accuracy": np.nan,
                        "brier": np.nan,
                        "error": str(exc),
                    }
                    rows.append(fail_row)
                    print(f"[MODEL FAILURE] {fail_row}")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "train_size"], as_index=False).agg(
        runs=("seed", "count"),
        train_pool_size=("train_pool_size", "median"),
        test_size=("test_size", "median"),
        n_restarts=("n_restarts", "median"),
        n_successful_restarts_mean=("n_successful_restarts", "mean"),
        n_failed_restarts_mean=("n_failed_restarts", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        final_train_nll_mean=("final_train_nll", "mean"),
        final_train_nll_std=("final_train_nll", "std"),
        final_kf_mean=("final_kf", "mean"),
        final_kf_std=("final_kf", "std"),
        train_time_med=("train_time_s", "median"),
        infer_time_med=("infer_time_s", "median"),
        lbfgs_iterations_med=("lbfgs_iterations", "median"),
    )


def main():
    train_sizes = [50, 100, 300, 500]
    max_train_size = 500

    seeds = list(range(10))
    n_restarts = 16
    kf_subset_size = 32

    X, y, feats = load_grid_stability_from_uci(drop_stab=True, drop_p1=True)
    print(f"Loaded UCI 471: X={X.shape}, y={y.shape}, num_features={len(feats)}")
    print("Features:", feats)
    print(f"Experimental runs: {len(seeds)}")
    print(f"GP restarts per run: {n_restarts}")
    print(f"KF subset size: {kf_subset_size}")

    models: List[BinaryModel] = [
        GPPlusGPClassifierRBF(
            alpha_epsilon=0.01,
            verbose=False,
            num_classes=2,
            print_kernel_once=True,
            use_fair_init=True,
            n_restarts=n_restarts,
            kf_subset_size=kf_subset_size,
        ),
        GPPlusGPClassifierMatern(
            alpha_epsilon=0.01,
            verbose=False,
            num_classes=2,
            print_kernel_once=True,
            use_fair_init=True,
            nu=0.5,
            n_restarts=n_restarts,
            kf_subset_size=kf_subset_size,
        ),
        GPPlusGPClassifierPowerExp(
            alpha_epsilon=0.01,
            verbose=False,
            num_classes=2,
            print_kernel_once=True,
            learn_power=True,
            fixed_power=1.5,
            use_working_powerexp_init=True,
            debug_prediction_spread=False,
            n_restarts=n_restarts,
            kf_subset_size=kf_subset_size,
        ),
    ]

    if TABPFN_AVAILABLE:
        models.extend([
            TabPFNBinaryClassifier(
                version="v2.5",
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size_predict=2048,
            ),
            TabPFNBinaryClassifier(
                version="v2",
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size_predict=2048,
            ),
        ])
    else:
        print("\n[WARNING] TabPFN is not available, so TabPFN baselines were skipped.")
        print(f"TabPFN import error: {TABPFN_IMPORT_ERROR}\n")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", run_id)
    ensure_dir(out_dir)

    callback_root = os.path.join(out_dir, "callback_runs")
    ensure_dir(callback_root)

    df = run_benchmark(
        X=X,
        y=y,
        models=models,
        train_sizes=train_sizes,
        seeds=seeds,
        max_train_size=max_train_size,
        callback_root=callback_root,
    )

    raw_path = os.path.join(out_dir, "grid_rbf_matern05_powerexp_tabpfn_multistart_stable_kf_raw.csv")
    summ_path = os.path.join(out_dir, "grid_rbf_matern05_powerexp_tabpfn_multistart_stable_kf_summary.csv")

    raw_saved = safe_to_csv(df, raw_path)
    print(f"\nSaved raw results to {raw_saved}")

    summ = summarize(df)
    summ_saved = safe_to_csv(summ, summ_path)
    print(f"Saved summary to {summ_saved}\n")
    print(summ)

    paper_fig_path = os.path.join(out_dir, "paper_performance_figure.png")
    saved_paper_fig = plot_paper_performance_figure(
        summary_df=summ,
        out_path=paper_fig_path,
        title="Electrical Grid Stability Benchmark",
    )
    print(f"Saved paper figure to {saved_paper_fig}")

    kf_fig_path = os.path.join(out_dir, "gp_kf_figure.png")
    saved_kf_fig = plot_gp_kf_figure(
        summary_df=summ,
        out_path=kf_fig_path,
        title="Electrical Grid Stability Benchmark",
    )
    print(f"Saved GP KF figure to {saved_kf_fig}")

    avg_plot_dir = os.path.join(out_dir, "averaged_training_curves")
    ensure_dir(avg_plot_dir)
    saved_plots = aggregate_and_plot_histories(
        callback_root=callback_root,
        out_dir=avg_plot_dir,
        train_sizes=train_sizes,
    )
    print(f"Saved {len(saved_plots)} averaged plots to {avg_plot_dir}")


if __name__ == "__main__":
    main()