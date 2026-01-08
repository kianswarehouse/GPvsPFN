import json
import time
from pathlib import Path

import gpplus
import gpytorch
import numpy as np
import torch
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from gpplus.utils import set_seed, train_eval_PFN
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings

# Allow running as a script from this folder OR importing as a module.
try:
    import defaults_gpytorch as defaults
    from gpytorch_train_eval import train_eval_gp_gpytorch_default
    from load_experimental_data import generate_rastrigin_data
except ModuleNotFoundError:  # pragma: no cover
    from experiments.Problems import defaults_gpytorch as defaults
    from experiments.Problems.gpytorch_train_eval import train_eval_gp_gpytorch_default
    from experiments.Problems.load_experimental_data import generate_rastrigin_data


class ExactGPModel(gpytorch.models.ExactGP):
    """Pure gpytorch ExactGP model for regression."""

    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def rastrigin_GPvsPFN(
    num_folds: int = defaults.NUM_FOLDS,
    num_test: int = 5000,
    train_size: int = 10,  # total training size is train_size * number of X input dimensions
    dimensions: int = 5,
    x_bounds: tuple[float, float] | list[float] = (-10.0, 10.0),
    num_runs: int = defaults.TRAINER_NUM_RUNS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float | None = defaults.TRAINER_LR,
    convergence_patience: int | None = defaults.TRAINER_CONVERGENCE_PATIENCE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,  # kept for API parity; not used in gpytorch flow
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    save_path: str | None = "./results/rastrigin",
    title: str | None = None,
    standardize_X: bool = True,
    standardize_y: bool = True,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    shift: torch.Tensor | float | None = None,
    gp_dtype: torch.dtype = getattr(defaults, "DTYPE_GP", torch.float64),
    pfn_dtype: torch.dtype = getattr(defaults, "DTYPE_PFN", torch.float32),
):
    """
    GP (gpytorch defaults: GaussianLikelihood + ConstantMean + ScaleKernel(RBFKernel(ARD)))
    vs TabPFN on the Rastrigin function.
    """

    _ = initializer_class  # not used, but accepted to match other scripts
    shift_str = f"_shifted{shift}" if shift is not None else ""
    x_bounds_str = f"[{float(x_bounds[0])},{float(x_bounds[1])}]"

    if title is None:
        title = (
            f"Rastrigin{shift_str}_gpytorch_{dimensions}xdim_{train_size}D_"
            f"{num_epochs}epochs_{num_runs}runs_{lr}_{x_bounds_str}bounds_"
            f"noiseTest{noise_test}_noiseTrain{noise_train}"
        )
    else:
        title = f"Rastrigin{shift_str}_gpytorch_{dimensions}xdim_{train_size}D_{title}"

    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)

    plot_save_path = f"{save_path}/plots" if save_path is not None else None

    # Generate data (unique Sobol points)
    set_seed(seed)

    train_per_fold = train_size * dimensions
    total_train = num_folds * train_per_fold
    total_samples = num_test + total_train

    print(
        f"Generating {total_samples} unique Sobol samples for {dimensions}D Rastrigin function"
        f"\n\tTest samples: {num_test} / Train samples: {total_train}"
    )
    if shift is not None:
        print(f"  Using shifted Rastrigin with shift: {shift}")

    X_train_all, y_train_all, X_test_all, y_test_all = generate_rastrigin_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
        shift=shift,
        x_bounds=x_bounds,
    )

    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("=" * 10)
    print(f"{title}: TabPFN vs GP Comparison (GPyTorch Defaults)")
    print("=" * 10)

    # Encode/column-type discovery (all continuous here, but kept consistent)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    TabPFN_metrics: list[dict] = []
    GPPlus_metrics: list[dict] = []

    # Per-fold run seeds (same pattern as other gpytorch scripts)
    set_seed(0)
    seeds = np.random.RandomState(0).choice(10**6, size=num_folds, replace=False).tolist()

    # Fold splits: make deterministic given `seed`
    torch.manual_seed(seed)
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)

    total_start_time = time.time()
    for i in range(num_folds):
        print(f"\n{'=' * 20} {title} FOLD {i + 1}/{num_folds} {'=' * 20}")

        fold_train_indices = train_indices_2d[i]
        X_train = X_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section (gpytorch)
        # =============================================================================
        print(f"\n--- {title} GP Training (GPyTorch Defaults) ---")

        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)

        if standardize_X:
            Xscaler = gpplus.utils.StandardScaler()
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])

        # Normalize the GP data (fit on train)
        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        # gpytorch defaults
        input_dim = X_train.shape[-1]
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=gp_dtype)
        mean_module = gpytorch.means.ConstantMean().to(dtype=gp_dtype)
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        ).to(dtype=gp_dtype)

        model = ExactGPModel(
            X_train,
            y_train_normal if standardize_y else y_train,
            likelihood,
            mean_module,
            covar_module,
        )

        if (i == 0) or (i == num_folds - 1):
            print(f"X_train: {X_train.shape}")
            print(f"X_test: {X_test.shape}")
            print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
            print(model)

        fold_seed = seeds[i]
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp_gpytorch_default(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            num_runs=num_runs,
            seed=fold_seed,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            convergence_patience=convergence_patience,
            optimizer_class=optimizer_class,
            lr=lr,
        )
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Fold {i + 1}/{num_folds})")
        for k, v in gp_metric.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"  {k}: NaN" if np.isnan(v) else f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")

        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_train,
            X_test,
            y_train_normal if standardize_y else y_train,
            y_test,
            amp_device=amp_device,
            amp_dtype=pfn_dtype,
            regressor=regressor,
            source_cols=source_cols,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
        )
        TabPFN_metrics.append(tabpfn_metric)

        print(f"\nTabPFN Results (Fold {i + 1}/{num_folds})")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")

        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item()),
            }
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": int(X_train.shape[1]),
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(num_test),
                "y_train_mean": float(y_train_mean.item()),
                "y_train_std": float(y_train_std.item()),
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": int(num_epochs),
                "num_runs": int(num_runs),
                "lr": lr if lr is not None else (0.1 if optimizer_class == torch.optim.LBFGS else 1e-3),
                "optimizer": optimizer_class.__name__ if optimizer_class is not None else "LBFGS",
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                "shift": str(shift) if shift is not None else None,
                "x_bounds": x_bounds_str,
                **y_test_stats,
                "num_folds": int(num_folds),
                "seed": int(seed),
                "seed_trainer": seed_trainer,
            }
            tabpfn_model_info = {
                "model_path": regressor.model_path,
                "fit_mode": regressor.fit_mode,
                "device": str(regressor.device_),
                "inference_precision": regressor.inference_precision,
                "random_state": regressor.random_state,
                "use_autocast": regressor.use_autocast_,
                "forced_inference_dtype": str(regressor.forced_inference_dtype_)
                if regressor.forced_inference_dtype_
                else None,
            }

    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title)

    if save_path is not None:
        plot_metrics(
            TabPFN_metrics,
            GPPlus_metrics,
            labels=["TabPFN", "GP"],
            title=title,
            save_path=plot_save_path,
        )

        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            combined_data = {
                "gp_data": {"summary": GPPlus_summary, "metrics": GPPlus_metrics, "gp_model_info": gp_model_info},
                "tabpfn_data": {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info,
                },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass

    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("=" * 60)
    optimizer_name = optimizer_class.__name__ if optimizer_class is not None else "LBFGS"
    lr_display = lr if lr is not None else (0.1 if optimizer_class == torch.optim.LBFGS else 1e-3)
    print(
        "Trainer details:"
        f"\n\tnumber of epochs: {num_epochs}"
        f"\n\tnumber of runs: {num_runs}"
        f"\n\tlearning rate: {lr_display}"
        f"\n\toptimizer: {optimizer_name}"
        f"\n\tconvergence patience: {convergence_patience}"
        f"\n\tdevice: {gp_device}"
        f"\n\tcont_cols: {cont_cols}"
        f"\n\tcat_cols: {cat_cols}"
        f"\n\tsource_cols: {source_cols}"
        f"\n\tqual_dict: {qual_dict}"
        f"\n\tX_standardize: {standardize_X}"
        f"\n\ty_standardize: {standardize_y}"
        f"\n\tshift: {shift}"
        f"\n\tx_bounds: {x_bounds_str}"
    )
    print(
        "Experiment details:"
        f"\n\t{len(X_test_all)} test samples, {train_per_fold} train samples per fold"
        f"\n\tfolds: {num_folds}"
    )

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Quick smoke runs (adjust as needed)
    rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=20, num_runs=4, save_path="./results/rastrigin/temp")
    # rastrigin_GPvsPFN(num_folds=1, train_size=10, dimensions=40, num_runs=4, save_path="./results/rastrigin/temp")
    # rastrigin_GPvsPFN(
    #     num_folds=1, train_size=10, dimensions=20, num_runs=4, save_path="./results/rastrigin/temp", shift=2.0
    # )
