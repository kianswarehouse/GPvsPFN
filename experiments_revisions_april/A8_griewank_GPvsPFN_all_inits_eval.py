import json
import time
from pathlib import Path

import torch

import defaults
import gpplus
from gpplus.training.eval2 import evaluate_gp_model
from gpplus.utils import set_seed, train_eval_gp
from gpplus.utils.metrics_functions import analyze_metrics, compute_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from load_experimental_data import generate_griewank_data


def _collect_raw_parameters(node: dict, prefix: str = "") -> dict[str, object]:
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


def _apply_raw_parameters_relaxed(model: torch.nn.Module, final_parameters: dict) -> None:
    raw_map = _collect_raw_parameters(final_parameters)
    with torch.no_grad():
        for dotted_name, value in raw_map.items():
            module = model
            parts = dotted_name.split(".")
            for part in parts[:-1]:
                if not hasattr(module, part):
                    module = None
                    break
                module = getattr(module, part)
            if module is None:
                continue

            leaf = parts[-1]
            if not hasattr(module, leaf):
                if leaf == "raw_constant" and hasattr(module, "constant"):
                    leaf = "constant"
                else:
                    continue

            target = getattr(module, leaf)
            if not torch.is_tensor(target):
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
                    continue
            target.copy_(src)


def griewank_GP_all_inits_eval(
    num_runs=1,
    num_test=5000,
    train_size=40,
    dimensions=40,
    x_bounds=[-600, 600],
    num_inits=defaults.TRAINER_NUM_INITS,
    num_epochs=defaults.TRAINER_NUM_EPOCHS,
    lr=defaults.TRAINER_LR,
    convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs=defaults.TRAINER_MIN_EPOCHS,
    min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
    cholesky_jitter=defaults.TRAINER_CHOLESKY_JITTER,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device=defaults.TRAINER_GP_DEVICE,
    save_path="./results/griewank/all_inits_eval",
    title=None,
    standardize_X=defaults.STANDARDIZE_X,
    standardize_y=defaults.STANDARDIZE_Y,
    x_standardize_method=defaults.X_STANDARDIZE_METHOD,
    noise_train=0.0,
    noise_test=0.0,
    noise_type=defaults.NOISE_TYPE,
    seed=defaults.SEED,
    seed_trainer=defaults.SEED_TRAINER,
    gp_dtype=defaults.DTYPE_GP,
    trainer_info=True,
    log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
    single_dataset=False,
    n_jobs=defaults.TRAINER_N_JOBS,
    inner_max_num_threads=defaults.TRAINER_INNER_MAX_NUM_THREADS,
):
    if title is None:
        title = (
            f"Griewank_allInits_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_"
            f"{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
        )

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

    gp_metrics = []
    all_inits_per_run = {}
    trainer_payload = {}
    total_start = time.time()

    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
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
            else:
                x_scaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
            x_scaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = x_scaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = x_scaler.transform(X_test[:, cont_cols])
        else:
            x_scaler = None

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

        gp_metric, _, _, gp_trainer_info = train_eval_gp(
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
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,
            trainer_info=trainer_info,
            callbacks=defaults.get_default_gp_callbacks(
                optimizer_class,
                callback_save_path=f"{save_path}/trainer_analysis/plots" if save_path else None,
                log_lbfgs_inner=log_lbfgs_inner,
            ),
            callback_save_path=f"{save_path}/trainer_analysis/plots" if save_path else None,
            cholesky_jitter=cholesky_jitter,
            optimizer_kwargs=optimizer_kwargs,
            log_lbfgs_inner=log_lbfgs_inner,
            n_jobs=n_jobs,
            inner_max_num_threads=inner_max_num_threads,
        )
        gp_metric["run"] = i + 1
        gp_metrics.append(gp_metric)

        run_init_metrics = []
        if gp_trainer_info and isinstance(gp_trainer_info.get("inits"), list):
            try:
                from threadpoolctl import threadpool_info as _eval_tp_info

                _eval_info = _eval_tp_info()
                _eval_summary = ", ".join(
                    f"{e.get('user_api', e.get('prefix', '?'))}={e.get('num_threads', '?')}"
                    for e in _eval_info
                )
                print(
                    f"[A8_all_inits_eval] threadpool_info[eval_main run={i + 1}]: {_eval_summary} | "
                    f"deterministic={torch.are_deterministic_algorithms_enabled()}"
                )
            except Exception as _exc:  # noqa: BLE001
                print(f"[A8_all_inits_eval] threadpool_info() unavailable: {_exc}")

            for init_rec in gp_trainer_info["inits"]:
                init_id = init_rec.get("init")
                final_params = init_rec.get("final_parameters")
                if not isinstance(final_params, dict):
                    continue

                model_init = gpplus.models.GPR(
                    X_train,
                    y_train_normal if standardize_y else y_train,
                    kernel_module=defaults.SF_kernel,
                    mean_module=defaults.SF_mean,
                    likelihood=defaults.SF_likelihood,
                )
                _apply_raw_parameters_relaxed(model_init, final_params)

                pred_start = time.time()
                y_pred, _, _, output_std = evaluate_gp_model(model_init, X_test)
                pred_time = time.time() - pred_start

                if standardize_y:
                    y_pred = (y_pred * y_train_std) + y_train_mean
                    output_std = output_std * y_train_std

                init_metric = compute_metrics(
                    y_test.detach().cpu().numpy().reshape(-1),
                    y_pred.detach().cpu().numpy().reshape(-1),
                    output_std=output_std.detach().cpu().numpy().reshape(-1),
                    training_time=0.0,
                    prediction_time=pred_time,
                )
                init_metric["run"] = i + 1
                init_metric["init"] = init_id
                init_metric["train_loss"] = init_rec.get("loss")
                init_metric["best_iter"] = init_rec.get("best_iter")
                run_init_metrics.append(init_metric)

        all_inits_per_run[f"run_{i + 1}"] = run_init_metrics
        trainer_payload[f"run_{i + 1}"] = gp_trainer_info

    summary_best = analyze_metrics(gp_metrics, print_summary=True, label="GP_bestInit", title=title)
    flat_all_inits = [m for per_run in all_inits_per_run.values() for m in per_run]
    summary_all = analyze_metrics(flat_all_inits, print_summary=True, label="GP_allInits", title=title) if flat_all_inits else None

    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "gp_data": {
            "summary_best_init": summary_best,
            "metrics_best_init": gp_metrics,
            "summary_all_inits_eval": summary_all,
            "all_inits_eval_metrics": all_inits_per_run,
            "config": {
                "num_runs": num_runs,
                "num_inits": num_inits,
                "dimensions": dimensions,
                "train_size": train_size,
                "noise_train": noise_train,
                "noise_test": noise_test,
                "single_dataset": single_dataset,
            },
        }
    }
    combined_file = out_dir / f"gp_{title}.json"
    combined_file.write_text(json.dumps(combined, indent=2))

    trainer_dir = out_dir / "trainer_analysis"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    trainer_eval_file = trainer_dir / f"gp_{title}_AllInits_TestEval.json"
    trainer_eval_file.write_text(
        json.dumps(
            {
                "title": title,
                "num_runs": num_runs,
                "num_inits_per_run": num_inits,
                "trainer_info": trainer_payload,
                "all_inits_eval_metrics": all_inits_per_run,
                "summary_all_inits_eval": summary_all,
                "summary_best_init": summary_best,
            },
            indent=2,
        )
    )

    # Compact text summary: per-run best by train loss vs best by test RRMSE
    summary_lines = []
    summary_lines.append(f"Title: {title}")
    summary_lines.append(f"num_runs={num_runs}, num_inits={num_inits}")
    summary_lines.append("")
    summary_lines.append("Per-run best init comparison")
    summary_lines.append("-" * 92)
    summary_lines.append(
        "run | best_train_loss_init (loss, RRMSE) | best_test_rrmse_init (RRMSE, loss)"
    )
    summary_lines.append("-" * 92)

    for run_key in sorted(all_inits_per_run.keys(), key=lambda k: int(k.replace("run_", ""))):
        per_init = all_inits_per_run.get(run_key, [])
        if not per_init:
            summary_lines.append(f"{run_key:>5} | no init metrics")
            continue

        valid_loss = [m for m in per_init if isinstance(m.get("train_loss"), (int, float))]
        valid_rrmse = [m for m in per_init if isinstance(m.get("RRMSE"), (int, float))]

        best_by_loss = min(valid_loss, key=lambda m: m["train_loss"]) if valid_loss else None
        best_by_rrmse = min(valid_rrmse, key=lambda m: m["RRMSE"]) if valid_rrmse else None

        if best_by_loss is None:
            left = "n/a"
        else:
            left = (
                f"init {best_by_loss.get('init')} "
                f"(loss={best_by_loss.get('train_loss'):.6g}, RRMSE={best_by_loss.get('RRMSE', float('nan')):.6g})"
            )

        if best_by_rrmse is None:
            right = "n/a"
        else:
            right = (
                f"init {best_by_rrmse.get('init')} "
                f"(RRMSE={best_by_rrmse.get('RRMSE'):.6g}, loss={best_by_rrmse.get('train_loss', float('nan')):.6g})"
            )

        summary_lines.append(f"{run_key:>5} | {left} | {right}")

    summary_txt_file = trainer_dir / f"gp_{title}_AllInits_CompactSummary.txt"
    summary_txt_file.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nSaved: {combined_file}")
    print(f"Saved: {trainer_eval_file}")
    print(f"Saved: {summary_txt_file}")
    print(f"Total experiment time: {time.time() - total_start:.2f}s")
    return gp_metrics, all_inits_per_run


if __name__ == "__main__":
    griewank_GP_all_inits_eval(
        single_dataset=False,
        num_runs=1,
        train_size=40,
        dimensions=40,
        num_inits=16,
        noise_train=0.005,
        noise_test=0.005,
        save_path="./results_May07/griewank/series_test_1run_16init_allInitsEval",
    )
