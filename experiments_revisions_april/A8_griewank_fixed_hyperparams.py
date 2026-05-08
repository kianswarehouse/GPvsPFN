import json
import time
from pathlib import Path

import torch

import defaults
import gpplus
from gpplus.utils import set_seed
from gpplus.utils.metrics_functions import analyze_metrics, compute_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from gpplus.training.eval2 import evaluate_gp_model
from load_experimental_data import generate_griewank_data


def _extract_final_parameters(
    trainer_json_path: str | Path,
    source_run: int = 1,
    source_init: int | None = None,
) -> dict:
    """
    Load final hyperparameters from trainer_analysis JSON.

    If source_init is None, the best init (lowest loss) from source_run is used.
    """
    path = Path(trainer_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    trainer_info = data.get("trainer_info", {})
    run_key = f"run_{source_run}"
    run_data = trainer_info.get(run_key)
    if run_data is None:
        raise ValueError(f"Run '{run_key}' not found in {path}")

    inits = run_data.get("inits", [])
    if not inits:
        raise ValueError(f"No init entries found in {path} for {run_key}")

    if source_init is None:
        best_entry = min(
            (entry for entry in inits if entry.get("loss") is not None),
            key=lambda entry: entry["loss"],
            default=None,
        )
        if best_entry is None:
            best_entry = inits[0]
    else:
        best_entry = next((entry for entry in inits if int(entry.get("init", -1)) == source_init), None)
        if best_entry is None:
            raise ValueError(f"Init {source_init} not found under {run_key} in {path}")

    final_parameters = best_entry.get("final_parameters")
    if not isinstance(final_parameters, dict):
        raise ValueError(f"final_parameters missing/invalid in {path} for {run_key}")
    return final_parameters


def _collect_raw_parameters(node: dict, prefix: str = "") -> dict[str, object]:
    """
    Recursively collect raw_* fields into dotted paths.
    Example: covar_module.base_kernel.raw_lengthscale -> [...]
    """
    out: dict[str, object] = {}
    for key, value in node.items():
        if key in {"type", "jitter"}:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_collect_raw_parameters(value, prefix=full_key))
        elif key.startswith("raw_"):
            out[full_key] = value
    return out


def _apply_raw_parameters(model: torch.nn.Module, final_parameters: dict) -> None:
    raw_map = _collect_raw_parameters(final_parameters)
    if not raw_map:
        raise ValueError("No raw_* parameters were found in final_parameters")

    missing_paths: list[str] = []
    with torch.no_grad():
        for dotted_name, value in raw_map.items():
            module = model
            name_parts = dotted_name.split(".")
            for part in name_parts[:-1]:
                if not hasattr(module, part):
                    missing_paths.append(dotted_name)
                    module = None
                    break
                module = getattr(module, part)
            if module is None:
                continue

            leaf = name_parts[-1]
            if not hasattr(module, leaf):
                # Backward-compatible alias: older trainer JSONs can store mean raw_constant
                # while the current mean module may expose only `constant`.
                if leaf == "raw_constant" and hasattr(module, "constant"):
                    leaf = "constant"
                else:
                    missing_paths.append(dotted_name)
                    continue

            target = getattr(module, leaf)
            if not torch.is_tensor(target):
                missing_paths.append(dotted_name)
                continue

            src = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if target.shape != src.shape:
                if target.numel() == src.numel():
                    src = src.reshape(target.shape)
                elif target.ndim == 2 and target.shape[0] == 1 and src.ndim == 1 and target.shape[1] == src.shape[0]:
                    src = src.unsqueeze(0)
                elif target.numel() == 1 and src.numel() == 1:
                    src = src.reshape(target.shape)
                else:
                    raise ValueError(
                        f"Shape mismatch for '{dotted_name}': model {tuple(target.shape)} vs source {tuple(src.shape)}"
                    )
            target.copy_(src)

    if missing_paths:
        missing_unique = sorted(set(missing_paths))
        print(
            "Warning: skipped missing/non-tensor fixed hyperparameters: "
            + ", ".join(missing_unique)
        )


def griewank_fixed_hyperparams_eval(
    trainer_json_path: str | Path,
    source_run: int = 1,
    source_init: int | None = None,
    num_runs: int = defaults.NUM_RUNS,
    num_test: int = 5000,
    train_size: int = 10,  # train_per_run = train_size * dimensions
    dimensions: int = 40,
    x_bounds: list[float] = [-600, 600],
    standardize_X: bool = defaults.STANDARDIZE_X,
    standardize_y: bool = defaults.STANDARDIZE_Y,
    x_standardize_method: int = defaults.X_STANDARDIZE_METHOD,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = defaults.NOISE_TYPE,
    seed: int = defaults.SEED,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    save_path: str | None = "./results/griewank_fixed_hyperparams",
    title: str | None = None,
    single_dataset: bool = False,
):
    """
    Evaluate Griewank GP using fixed pre-learned hyperparameters (no optimization).
    """
    fixed_final_params = _extract_final_parameters(
        trainer_json_path=trainer_json_path,
        source_run=source_run,
        source_init=source_init,
    )
    fixed_raw_map = _collect_raw_parameters(fixed_final_params)

    if title is None:
        init_txt = "best" if source_init is None else f"init{source_init}"
        title = (
            f"Griewank_fixedHP_fromRun{source_run}_{init_txt}_"
            f"{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_"
            f"noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
        )

    print("=" * 80)
    print("Fixed-hyperparameter Griewank GP evaluation (no optimization)")
    print("=" * 80)
    print(f"Trainer JSON source: {trainer_json_path}")
    print(f"Source run/init: run_{source_run} / {'best' if source_init is None else source_init}")
    print(f"Collected raw params: {list(sorted(fixed_raw_map.keys()))}")

    set_seed(seed)
    train_per_run = train_size * dimensions
    if single_dataset:
        n_train_generate = train_per_run
    else:
        num_runs_gen = max(num_runs, 20)
        n_train_generate = num_runs_gen * train_per_run

    X_train_all, y_train_all, X_test_all, y_test_all = generate_griewank_data(
        n_train=n_train_generate,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)
    qual_dict = learn_encodings(X)
    _, cont_cols, _, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)

    if not single_dataset:
        torch.manual_seed(seed)
        all_indices = torch.randperm(n_train_generate)
        train_indices_2d = all_indices.reshape(max(num_runs, 20), train_per_run)

    metrics = []
    total_start = time.time()
    for i in range(num_runs):
        print(f"\n{'=' * 25} {title} RUN {i + 1}/{num_runs} {'=' * 25}")
        if single_dataset:
            X_train = X_train_all
            y_train = y_train_all
        else:
            idx = train_indices_2d[i]
            X_train = X_train_all[idx]
            y_train = y_train_all[idx]

        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)

        if standardize_X:
            if x_standardize_method == 0:
                x_scaler = gpplus.utils.StandardScaler()
            elif x_standardize_method == 1:
                x_scaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
            elif x_standardize_method == 2:
                x_scaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
            else:
                raise ValueError(f"x_standardize_method must be 0/1/2, got {x_standardize_method}")
            x_scaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = x_scaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = x_scaler.transform(X_test[:, cont_cols])

        y_scaler = gpplus.utils.StandardScaler()
        y_scaler.fit(y_train)
        y_train_mean = y_scaler.mean
        y_train_std = y_scaler.std
        y_train_normal = y_scaler.transform(y_train)

        model = gpplus.models.GPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            kernel_module=defaults.SF_kernel,
            mean_module=defaults.SF_mean,
            likelihood=defaults.SF_likelihood,
        )
        _apply_raw_parameters(model, fixed_final_params)

        t_pred = time.time()
        y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
        prediction_time = time.time() - t_pred

        if standardize_y:
            y_pred = (y_pred * y_train_std) + y_train_mean
            output_std = output_std * y_train_std

        y_test_np = y_test.detach().cpu().numpy().reshape(-1)
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        output_std_np = output_std.detach().cpu().numpy().reshape(-1)
        metric = compute_metrics(
            y_test_np,
            y_pred_np,
            output_std=output_std_np,
            training_time=0.0,
            prediction_time=prediction_time,
        )
        metric["fixed_hyperparams"] = True
        metric["source_run"] = source_run
        metric["source_init"] = "best" if source_init is None else source_init
        metric["run"] = i + 1
        metrics.append(metric)

        print(f"RRMSE: {metric.get('RRMSE', float('nan')):.4f}")
        print(f"NIS:   {metric.get('NIS', float('nan')):.4f}")
        print(f"Pred time: {prediction_time:.4f}s")

    summary = analyze_metrics(metrics, print_summary=True, label="GP_fixedHP", title=title)

    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "source_trainer_json": str(trainer_json_path),
            "source_run": source_run,
            "source_init": "best" if source_init is None else source_init,
            "fixed_raw_parameters": fixed_raw_map,
            "metrics": metrics,
            "summary": summary,
            "experiment": {
                "num_runs": num_runs,
                "num_test": num_test,
                "train_size": train_size,
                "dimensions": dimensions,
                "x_bounds": x_bounds,
                "noise_train": noise_train,
                "noise_test": noise_test,
                "noise_type": noise_type,
                "single_dataset": single_dataset,
                "seed": seed,
            },
        }
        out_file = out_dir / f"gp_fixedHP_{title}.json"
        out_file.write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {out_file}")

    print(f"\nTotal fixed-hyperparameter experiment time: {time.time() - total_start:.2f}s")
    return metrics, summary


if __name__ == "__main__":
    # Update trainer_json_path to your Dx=40, train_size=10 Griewank trainer-analysis JSON.
    trainer_json_path = (
        "./results_April28/20_runs_Gaussian/"
        "griewank/trainer_analysis/"
        "gp_Griewank_40Dx_10Dn_[-600,600]_16inits_noiseTest0.005_noiseTrain0.005_x20_GP_Trainer_Analysis.json"
    )
    griewank_fixed_hyperparams_eval(
        trainer_json_path=trainer_json_path,
        source_run=1,
        source_init=None,  # None => best init from source_run
        num_runs=20,
        train_size=40,
        dimensions=40,
        noise_train=0.005,
        noise_test=0.005,
        single_dataset=False,
        save_path="./results_April29/fixed_hyperparams/griewank",
    )
