"""
Tutorial: Using callbacks with GPTrainer — no train_eval.

Shows how to:
- Set up data and model and pass them directly to GPTrainer (no train_eval_gp).
- Use callbacks (FinalParameterStorageCallback, LBFGSInnerMetricsCallback) to capture parameters and metrics.
- When save_path is set, the trainer builds and saves the unified JSON automatically
  (logic in gpplus.training.trainer_analysis).

Run from repo root or experiments_final:
  python experiments_final/tutorial_callbacks_one_run.py
  python experiments_final/tutorial_callbacks_one_run.py --save_path results_v5.0
"""
import sys
from pathlib import Path

# Allow importing from experiments_final and gpplus
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import gpplus
from gpplus.utils import set_seed
from gpplus.training import GPTrainerV3 as GPTrainer
from gpplus.training.callbacks import (
    FinalParameterStorageCallback,
    IterationParameterCallback,
    LBFGSInnerMetricsCallbackV3 as LBFGSInnerMetricsCallback,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.training.stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
)

# Optional: use wing data (same as A1_wing_SF_GPvsPFN)
try:
    from load_experimental_data import generate_mf_wing_data
    from defaults import (
        TRAINER_OPTIMIZER_KWARGS,
        TRAINER_INITIALIZER_CLASS,
        TRAINER_GP_DEVICE,
        TRAINER_CHOLESKY_JITTER,
        TRAINER_CONVERGENCE_PATIENCE,
        TRAINER_MIN_LOSS_CHANGE,
        SEED,
        DTYPE_GP,
    )
    SF_kernel = getattr(__import__("defaults", fromlist=["SF_kernel"]), "SF_kernel", None)
    SF_mean = getattr(__import__("defaults", fromlist=["SF_mean"]), "SF_mean", None)
    SF_likelihood = getattr(__import__("defaults", fromlist=["SF_likelihood"]), "SF_likelihood", None)
    HAS_WING = True
except Exception:
    HAS_WING = False


def get_wing_data_one_run(train_size=10, test_size=100, seed=42):
    """Generate one train/test split of wing data (single run)."""
    set_seed(seed)
    train_n = train_size * 10
    X_train_all, y_train_all, X_test_all, y_test_all = generate_mf_wing_data(
        train_samples_per_source=[train_n, 0, 0, 0],
        test_samples_per_source=[test_size, 0, 0, 0],
        train_noise=0.005,
        test_noise=0.005,
        seed=seed,
    )
    if X_train_all.shape[1] == 11:
        X_train_all = X_train_all[:, :10]
    if X_test_all.shape[1] == 11:
        X_test_all = X_test_all[:, :10]
    X_train = X_train_all.to(dtype=torch.float64)
    y_train = y_train_all.to(dtype=torch.float64)
    X_test = X_test_all.to(dtype=torch.float64)
    y_test = y_test_all.to(dtype=torch.float64)
    return X_train, y_train, X_test, y_test


def get_callbacks(extra_callbacks=None, iteration_params_only=False, save_path=None):
    """
    Build the callback list for one run (LBFGS).

    - FinalParameterStorageCallback: stores initial/final parameters (and jitter) per init.
    - IterationParameterCallback: initial + final (same format) + optional records every N iters.
    - LBFGSInnerMetricsCallbackV3: per-iteration NLL, NIS, LOO_NLL, KF, MSE, R2.

    If iteration_params_only=True, only LBFGSInnerMetricsCallback is used (no FinalParameterStorageCallback);
    parameters are built in via log_parameters_every_n_iters and reported as lbfgs_parameters.

    If save_path is set (e.g. "results_v5.0"), callbacks get save_file under
    save_path/tutorial/trainer_analysis/ so the trainer can auto-save after train().
    """
    if save_path:
        out_dir = Path(save_path).resolve() / "tutorial" / "trainer_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        callback_save_file = str(out_dir / "GP_Trainer_Analysis_tutorial.json")
    else:
        callback_save_file = None

    callbacks = [
        LBFGSInnerMetricsCallback(
            log_record_every_n_iters=1,
            log_metrics_every_n_iters=5,
            log_nll=True,
            log_nis=True,
            log_loo=True,
            log_kf=True,
            log_residual_mse=True,
            log_parameters_every_n_iters=10,
            save_file=callback_save_file,
        ),
    ]
    if not iteration_params_only:
        callbacks.insert(0, FinalParameterStorageCallback(save_file=callback_save_file, verbose=False))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    return callbacks


def main(save_path="results_v5.0", iteration_params_only=False, use_callback_save_only=False):
    """
    Run one training run with callbacks. No train_eval.

    save_path: Directory to save trainer analysis JSON (e.g. results_v5.0).
              File is written to save_path/tutorial/trainer_analysis/GP_Trainer_Analysis_tutorial.json.
              Use None to skip saving.
    """
    print("Tutorial: GPTrainer + callbacks (one run, no train_eval)")
    if save_path:
        out_dir = Path(save_path).resolve() / "tutorial" / "trainer_analysis"
        out_file = out_dir / "GP_Trainer_Analysis_tutorial.json"
        print(f"Saving to: {out_file}\n")
    else:
        print("(Saving disabled; all results in-memory.)\n")

    if not HAS_WING:
        print("Wing data not available. Using tiny synthetic data.")
        set_seed(42)
        n_train, n_test, d = 30, 50, 5
        X_train = torch.randn(n_train, d, dtype=torch.float64) * 0.5
        y_train = torch.randn(n_train, dtype=torch.float64) * 0.3
        X_test = torch.randn(n_test, d, dtype=torch.float64) * 0.5
        y_test = torch.randn(n_test, dtype=torch.float64) * 0.3
    else:
        X_train, y_train, X_test, y_test = get_wing_data_one_run(
            train_size=10, test_size=200, seed=SEED
        )
        print(f"Wing data: train {X_train.shape}, test {X_test.shape}")

    model = gpplus.models.GPR(
        X_train,
        y_train,
        kernel_module=SF_kernel if HAS_WING else None,
        mean_module=SF_mean if HAS_WING else None,
        likelihood=SF_likelihood if HAS_WING else None,
    )
    model.dtype = DTYPE_GP if HAS_WING else torch.float64

    callbacks = get_callbacks(iteration_params_only=iteration_params_only, save_path=save_path)
    num_inits = 2
    optimizer_kwargs = TRAINER_OPTIMIZER_KWARGS if HAS_WING else {"max_iter": 500, "history_size": 10}
    device = TRAINER_GP_DEVICE if HAS_WING else "cpu"

    # When save_path is set and not use_callback_save_only, trainer builds and saves the unified JSON automatically
    trainer_analysis_save_path = None
    if save_path and not use_callback_save_only:
        out_dir = Path(save_path).resolve() / "tutorial" / "trainer_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer_analysis_save_path = str(out_dir / "GP_Trainer_Analysis_tutorial.json")

    trainer = GPTrainer(
        model=model,
        num_epochs=1,
        seed=SEED if HAS_WING else 42,
        num_inits=num_inits,
        stop_conditions=[
            ConvergencePatienceStopCondition(
                patience=TRAINER_CONVERGENCE_PATIENCE if HAS_WING else 10
            ),
            MinLossChangeStopCondition(
                min_loss_change=TRAINER_MIN_LOSS_CHANGE if HAS_WING else 1e-7
            ),
        ],
        min_epochs=0,
        callbacks=callbacks,
        optimizer_class=LBFGSScipy,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        initializer_class=TRAINER_INITIALIZER_CLASS if HAS_WING else None,
        initializer_kwargs=None,
        cholesky_jitter=TRAINER_CHOLESKY_JITTER if HAS_WING else 1e-6,
        log_lbfgs_inner=True,
        trainer_analysis_save_path=trainer_analysis_save_path,
    )

    print(f"Training (one run, {num_inits} inits)...")
    train_results = trainer.train()

    print(f"\nCompleted. Got {len(train_results)} result(s) (one per init).")
    best = min((r for r in train_results if r.get("loss") is not None), key=lambda r: r["loss"])
    print(f"Best loss: {best['loss']:.6f} (init_index {best.get('init_index')})")

    for cb in callbacks:
        if hasattr(cb, "get_stored_parameters"):
            stored = cb.get_stored_parameters()
            if not stored:
                continue
            name = cb.__class__.__name__
            if isinstance(stored, list):
                print(f"  {name}: {len(stored)} record(s)")
            else:
                print(f"  {name}: {list(stored.keys())}")

    if train_results and train_results[0].get("lbfgs_inner_metrics"):
        print(f"  LBFGS inner metrics: {len(train_results[0]['lbfgs_inner_metrics'])} iterations (first init)")

    if save_path and trainer_analysis_save_path:
        print(f"\nTrainer analysis saved to: {trainer_analysis_save_path}")
    elif save_path and use_callback_save_only:
        print(f"\nSaved via callbacks to {Path(save_path).resolve() / 'tutorial' / 'trainer_analysis' / 'GP_Trainer_Analysis_tutorial.json'}")

    print("\nDone. Use get_callbacks(extra_callbacks=[...]) to add custom callbacks.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tutorial: GPTrainer + callbacks (one run, no train_eval)")
    p.add_argument("--save_path", default="results_v5.0", help="Directory to save JSON (default: results_v5.0). Callbacks get save_file under this path so trainer auto-saves.")
    p.add_argument("--iteration_params_only", action="store_true", help="Use only IterationParameterCallback (no FinalParameterStorageCallback).")
    p.add_argument("--use_callback_save_only", action="store_true", help="Rely only on callback/trainer save (no manual aggregation). JSON format is from IterationParameterCallback aggregation.")
    args = p.parse_args()
    save_path = None if args.save_path.lower() in ("", "none") else args.save_path
    main(save_path=save_path, iteration_params_only=args.iteration_params_only, use_callback_save_only=args.use_callback_save_only)
