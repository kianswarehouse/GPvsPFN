"""
Metrics computed on training data for monitoring (e.g. NIS, LOO, Kernel Flows).
Used when log_nis / log_loo / log_kf / log_residual_mse are enabled in the trainer.
"""
import warnings
from typing import Dict, Optional

import gpytorch
import torch

from ..utils.metrics_functions import compute_nis


def compute_kf_metric(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    Nf: int,
    cholesky_jitter: Optional[float] = None,
    seed: Optional[int] = None,
    likelihood=None,
) -> float:
    """
    Kernel Flows (KF) metric ρ_t = 1 - (||v||_K^2 / ||u||_K^2) per Algorithm 1 (line 15).

    - ||u||_K^2 = r^T K_θ^{-1} r  (full residuals, full kernel + noise)
    - ||v||_K^2 = r_sub^T K_sub^{-1} r_sub  (subset residuals, subset kernel + noise)
    - r = y - m(X; β), and K = k(X,X) + σ_n^2 I.

    Used to monitor geometric stability and detect overfitting. Higher ρ can indicate
    the subset approximates the full fit well.

    Args:
        model: GP model (ExactGP or similar) with train data already set.
        train_x: Training inputs (N, D).
        train_y: Training targets (N,) or (N, 1).
        Nf: Subset size for KF (batch size for KF).
        cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.
        seed: Optional seed for subset sampling (for reproducibility).
        likelihood: Optional; if provided, use likelihood(prior) so K includes noise (recommended).

    Returns:
        ρ_t in [0, 1] or float('nan') on failure.
    """
    try:
        import linear_operator

        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        N = train_x.size(0)
        if Nf >= N or Nf < 1:
            return float("nan")
        if N > 2000:
            return float("nan")

        # Noise variance for K = prior_cov + noise_var*I
        noise_var = 1e-6
        if likelihood is not None and hasattr(likelihood, "noise"):
            try:
                n = likelihood.noise
                noise_var = n.item() if n.numel() == 1 else n.flatten()[0].item()
                noise_var = max(noise_var, 1e-12)
            except Exception:
                pass

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            try:
                linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
            except Exception:
                pass

            # Single forward on full training data (ExactGP only allows this)
            prior = model(train_x)
            r = (train_y - prior.mean).view(-1)
            K_prior = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
            if K_prior is None:
                return float("nan")

            # Dense path: build K = prior + noise*I once, then full solve and submatrix solve
            K_prior_dense = K_prior.to_dense()
            K_dense = K_prior_dense + noise_var * torch.eye(
                N, device=K_prior_dense.device, dtype=K_prior_dense.dtype
            )
            K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
            u_sq = (r * K_inv_r).sum().item()
            if u_sq <= 0 or not torch.isfinite(torch.tensor(u_sq)):
                return float("nan")

            # Subset: use principal submatrix of K_dense (no second model call)
            gen = torch.Generator(device=train_x.device)
            if seed is not None:
                gen.manual_seed(seed)
            inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]
            r_sub = r[inds]
            K_sub = K_dense[inds][:, inds]
            K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
            v_sq = (r_sub * K_sub_inv_r).sum().item()
            if not torch.isfinite(torch.tensor(v_sq)):
                return float("nan")

            rho = 1.0 - (v_sq / u_sq)
            return float(rho)
    except Exception:
        return float("nan")


def compute_kf_metric_tensor(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    Nf: int,
    cholesky_jitter: Optional[float] = None,
    seed: Optional[int] = None,
    likelihood=None,
) -> torch.Tensor:
    """
    Differentiable Kernel Flows metric ρ_t = 1 - (||v||_K^2 / ||u||_K^2) as a 0-dim tensor.
    Same formula as compute_kf_metric but keeps the computation in the graph for use as a loss.
    Use loss = -rho to maximize KF, or loss = nll + rho / nll - rho for composites.

    Returns:
        0-dim tensor ρ_t on the same device as train_x. Returns zeros (no gradient) if N>2000 or Nf invalid.
    """
    jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
    N = train_x.size(0)
    if Nf >= N or Nf < 1 or N > 2000:
        return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
    try:
        with gpytorch.settings.cholesky_jitter(jitter):
            prior = model(train_x)
            r = (train_y - prior.mean).view(-1)
            K_prior = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
            if K_prior is None:
                return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
            noise_var = torch.tensor(1e-6, device=train_x.device, dtype=train_x.dtype)
            if likelihood is not None and hasattr(likelihood, "noise"):
                n = likelihood.noise
                noise_var = n.clamp(min=1e-12) if n.numel() == 1 else n.flatten()[0].clamp(min=1e-12)
            K_prior_dense = K_prior.to_dense()
            K_dense = K_prior_dense + noise_var * torch.eye(
                N, device=K_prior_dense.device, dtype=K_prior_dense.dtype
            )
            K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
            u_sq = (r * K_inv_r).sum()
            u_sq_safe = u_sq.clamp(min=1e-12)
            gen = torch.Generator(device=train_x.device)
            if seed is not None:
                gen.manual_seed(seed)
            inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]
            r_sub = r[inds]
            K_sub = K_dense[inds][:, inds]
            K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
            v_sq = (r_sub * K_sub_inv_r).sum()
            rho = 1.0 - (v_sq / u_sq_safe)
            return rho
    except Exception:
        return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)


def compute_rrmse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Relative root mean squared error (RRMSE) of training residuals:
    sqrt(MSE) / std(train_y), where MSE is computed on (train_y - model_mean).

    When train_y is standardized (mean 0, std 1), std(train_y)=1 so this equals
    RMSE in standardized space, which is the same as RRMSE on the original scale
    (no need for original y_mean/y_std). Lower is better.

    Args:
        model: GP model with train data already set.
        train_x: Training inputs.
        train_y: Training targets.
        cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.

    Returns:
        RRMSE (scalar). Returns float('nan') on failure.
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        r = (y_flat - mean)
        mse = (r**2).mean()
        # Normalize by std of training targets
        y_std = y_flat.std()
        eps = torch.tensor(1e-12, device=y_std.device, dtype=y_std.dtype)
        denom = torch.clamp(y_std, min=eps)
        rrmse = torch.sqrt(mse) / denom
        model.train()
        return float(rrmse.item()) if torch.isfinite(rrmse) else float("nan")
    except Exception:
        return float("nan")


def compute_mse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Mean squared error of training residuals: mean((train_y - model_mean)^2).
    Lower is better.
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        r = y_flat - mean
        mse = (r ** 2).mean()
        model.train()
        return float(mse.item()) if torch.isfinite(mse) else float("nan")
    except Exception:
        return float("nan")


def compute_r2(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Coefficient of determination R^2 on training data.

    R^2 = 1 - SSE / SST, where:
      - SSE = sum((y - y_hat)^2)
      - SST = sum((y - mean(y))^2)
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        resid = y_flat - mean
        sse = (resid * resid).sum()
        y_mean = y_flat.mean()
        sst = ((y_flat - y_mean) ** 2).sum()
        eps = torch.tensor(1e-12, device=sst.device, dtype=sst.dtype)
        sst = torch.clamp(sst, min=eps)
        r2 = 1.0 - (sse / sst)
        model.train()
        return float(r2.item()) if torch.isfinite(r2) else float("nan")
    except Exception:
        return float("nan")


def compute_residual_mse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """Backwards-compatible alias for compute_mse (residual MSE)."""
    return compute_mse(model, train_x, train_y, cholesky_jitter=cholesky_jitter)


def compute_training_nis(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Compute Normalized Interval Score (NIS) on training data.

    Uses the model's posterior at train_x to get mean and variance, then
    computes NIS (interval score normalized by std of y). Lower is better.

    Args:
        model: GP model (ExactGP or similar) with train data already set.
        train_x: Training inputs.
        train_y: Training targets.
        alpha: Nominal miscoverage for interval (default 0.05).

    Returns:
        NIS scalar (float). Returns float('nan') on failure.
    """
    try:
        with torch.no_grad():
            model.eval()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model.likelihood(model(train_x))
            mean = out.mean
            var = out.variance
            std = var.sqrt()
            model.train()

        # compute_nis expects numpy or tensor; it converts to numpy
        result = compute_nis(
            train_y,
            y_hat=mean,
            output_std=std,
            alpha=alpha,
            normalize_by_y_std=True,
        )

        return result["NIS"]
    except Exception:
        return float("nan")


def compute_loo_pll(
    model,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Compute leave-one-out pseudo log likelihood on training data.

    Uses GPyTorch's LeaveOneOutPseudoLikelihood. Higher is better.

    Args:
        model: GP model (ExactGP or similar).
        likelihood: Model's likelihood (e.g. model.likelihood).
        train_x: Training inputs.
        train_y: Training targets.
        cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.

    Returns:
        LOO pseudo log likelihood (scalar). Returns float('nan') on failure.
    """
    try:
        import linear_operator

        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        # Stay in train mode to avoid GPInputWarning (eval + same input as train data)
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            try:
                linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
            except Exception:
                pass
            loo_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
            out = model(train_x)
            pll = loo_mll(out, train_y)
            if pll.dim() == 0:
                return pll.item()
            return pll.sum().item()
    except Exception:
        return float("nan")


def compute_training_metrics_batch(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
    log_nis: bool = True,
    log_loo: bool = True,
    log_kf: bool = True,
    log_mse: bool = True,
    log_r2: bool = True,
    kf_Nf: Optional[int] = None,
    kf_seed: Optional[int] = None,
    nis_alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute NIS, LOO_NLL, KF, MSE, R2 in one go: single model.eval(), one forward
    (prior = model(train_x)), one likelihood(prior) for NIS, then model.train().
    Returns only requested metrics to avoid redundant work.
    """
    out: Dict[str, float] = {}
    jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
    train_x = train_x.to(dtype=getattr(model, "dtype", torch.float64))
    train_y = train_y.to(dtype=getattr(model, "dtype", torch.float64))
    N = train_x.size(0)

    try:
        import linear_operator
    except Exception:
        return out

    
    try:
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            try:
                linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
            except Exception:
                pass
            with warnings.catch_warnings():
                model.eval()
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                prior = model(train_x)
                pred = model.likelihood(prior)
            model.train()
            mean = pred.mean.detach().view(-1)
            y_flat = train_y.view(-1)

            if log_mse:
                r = y_flat - mean
                mse = (r ** 2).mean()
                out["MSE"] = float(mse.item()) if torch.isfinite(mse) else float("nan")
            if log_r2:
                resid = y_flat - mean
                sse = (resid * resid).sum()
                sst = ((y_flat - y_flat.mean()) ** 2).sum().clamp(min=1e-12)
                r2 = 1.0 - (sse / sst)
                out["R2"] = float(r2.item()) if torch.isfinite(r2) else float("nan")

            if log_nis:
                try:
                    var = pred.variance
                    std = var.sqrt()
                    nis_result = compute_nis(
                        train_y,
                        y_hat=mean,
                        output_std=std,
                        alpha=nis_alpha,
                        normalize_by_y_std=True,
                    )
                    out["NIS"] = nis_result["NIS"]
                except Exception:
                    out["NIS"] = float("nan")

            if log_loo:
                try:
                    loo_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
                    pll = loo_mll(prior, train_y)
                    val = pll.sum().item() if pll.dim() > 0 else pll.item()
                    out["LOO_NLL"] = float(-val) if torch.isfinite(torch.tensor(val)) else float("nan")
                except Exception:
                    out["LOO_NLL"] = float("nan")

            if log_kf:
                Nf = kf_Nf if kf_Nf is not None else min(64, max(2, N // 2))
                if Nf < N and Nf >= 1 and N <= 2000:
                    try:
                        r = (y_flat - mean).view(-1)
                        # Use posterior covariance (already K_prior + noise*I from likelihood)
                        K_lazy = getattr(pred, "lazy_covariance_matrix", None) or getattr(pred, "lazy_covar", None)
                        if K_lazy is not None:
                            K_dense = K_lazy.to_dense()
                            K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
                            u_sq = (r * K_inv_r).sum().item()
                            if u_sq > 0 and torch.isfinite(torch.tensor(u_sq)):
                                gen = torch.Generator(device=train_x.device)
                                if kf_seed is not None:
                                    gen.manual_seed(kf_seed)
                                inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]
                                r_sub = r[inds]
                                K_sub = K_dense[inds][:, inds]
                                K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
                                v_sq = (r_sub * K_sub_inv_r).sum().item()
                                if torch.isfinite(torch.tensor(v_sq)):
                                    out["KF"] = float(1.0 - (v_sq / u_sq))
                                else:
                                    out["KF"] = float("nan")
                            else:
                                out["KF"] = float("nan")
                        else:
                            out["KF"] = float("nan")
                    except Exception:
                        out["KF"] = float("nan")
                else:
                    out["KF"] = float("nan")
    finally:
        model.train()

    return out
