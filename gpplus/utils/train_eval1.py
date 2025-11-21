import time

import torch

from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from gpplus.training import GPTrainer
from gpplus.training.callbacks import FinalParameterStorageCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import compute_metrics


def train_eval_gp(
    model,
    X_test: torch.Tensor,
    y_test,
    *,
    num_epochs: int,
    seed: int,
    num_runs: int,
    lr: float,
    convergence_patience: int,
    optimizer_class=torch.optim.Adam,
    initializer_class=None,
    device: str,
    dtype: torch.dtype = torch.float64,
    y_train_mean: torch.Tensor | None = None,
    y_train_std: torch.Tensor | None = None,
):
    """
    Train a GP model and evaluate metrics on the provided test set.

    If y_train_mean/std are provided, predictions and uncertainty are denormalized
    before metrics are computed.

    Returns:
        gp_metric: dict of computed metrics (includes timing via start_time)
        y_pred: numpy array of predictions (denormalized if mean/std provided)
        output_std: numpy array of predictive std (denormalized if mean/std provided)
    """
    trainer = GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs,
        optimizer_kwargs={"lr": lr},
        convergence_patience=convergence_patience,
        optimizer_class=optimizer_class,
        callbacks=[FinalParameterStorageCallback(save_file="gp_parameters.json", verbose=True)],
        device=device,
        initializer_class=initializer_class,
    )

    t_start = time.time()
    _ = trainer.train()

    y_pred, _, _, output_std = evaluate_gp_model(model, X_test)

    if y_train_mean is not None and y_train_std is not None:
        y_pred = (y_pred * y_train_std) + y_train_mean
        output_std = output_std * y_train_std

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)

    gp_metric = compute_metrics(y_test, y_pred_np, output_std_np, start_time=t_start)

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
):
    """
    Train/evaluate TabPFN on provided split and return metrics, preds, std.

    Expects regressor with forward/predict_mean/predict_variance API.
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
    y_pred_test = y_pred[-len(y_test) :]
    output_std_test = output_std[-len(y_test) :]

    metrics = compute_metrics(y_test, y_pred_test, output_std_test, start_time=t_start)

    return metrics, y_pred_test, output_std_test
