"""
IDETC figures for Overleaf: create individual per-problem BO plots
and a LaTeX snippet that arranges them as subfigures (3 per row).

Run: python experiments_BO/plot_BO_IDETC_overleaf.py

Output
------
experiments_BO/results_BO/gpVpfn_BO_plots/
  - Individual PNGs for each (problem, noise) pair using clean y-values
  - gpVpfn_BO_figures.tex with two figure* environments (one per noise level)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_BO import (
    RunTuple,
    _average_best_y_at_each_iteration,
    _darker_color,
    _get_palette,
    collect_runs,
)
from plot_BO_IDETC import (
    COLOR_FACTOR,
    DEFAULT_PATHS,
    IDETC_PROBLEMS,
    MAXIMIZATION_PROBLEMS,
    NOISE_HIGH,
    NOISE_LOW,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results_BO" / "gpVpfn_BO_plots"


@dataclass(frozen=True)
class ProblemTexMeta:
    folder: str
    dim_key: str
    noise_key: str  # NOISE_LOW / NOISE_HIGH
    letter: str
    name: str  # e.g. "Buckling"
    dim_caption: str  # e.g. "D_x=4"
    slug: str  # used for filenames, e.g. "buckling_D4"


PROBLEM_DIM_INFO = {
    ("B2_buckling", ""): "D_x=4",
    ("B3_borehole", ""): "D_x=8",
    ("B1_wing", ""): "D_x=10",
    ("B4_ackley", "20Dx"): "D_x=20",
    ("B8_griewank", "20Dx"): "D_x=20",
    ("B7_zakharov", "20Dx"): "D_x=20",
    ("B4_ackley", "40Dx"): "D_x=40",
    ("B9_dixon_price", "40Dx"): "D_x=40",
}

PROBLEM_SLUGS = {
    ("B2_buckling", ""): "buckling_D4",
    ("B3_borehole", ""): "borehole_D8",
    ("B1_wing", ""): "wing_D10",
    ("B4_ackley", "20Dx"): "ackley_D20",
    ("B8_griewank", "20Dx"): "griewank_D20",
    ("B7_zakharov", "20Dx"): "zakharov_D20",
    ("B4_ackley", "40Dx"): "ackley_D40",
    ("B9_dixon_price", "40Dx"): "dixon_price_D40",
}

PROBLEM_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]


def _build_problem_tex_meta(noise_key: str) -> list[ProblemTexMeta]:
    """Create ordered metadata matching IDETC_PROBLEMS and the lettering in the paper."""
    meta_list: list[ProblemTexMeta] = []
    for idx, (folder, dim_key, subtitle) in enumerate(IDETC_PROBLEMS):
        letter = PROBLEM_LETTERS[idx]
        dim_caption = PROBLEM_DIM_INFO[(folder, dim_key)]
        slug = PROBLEM_SLUGS[(folder, dim_key)]
        meta_list.append(
            ProblemTexMeta(
                folder=folder,
                dim_key=dim_key,
                noise_key=noise_key,
                letter=letter,
                name=subtitle,
                dim_caption=dim_caption,
                slug=slug,
            )
        )
    return meta_list


def _draw_single_subplot(
    ax: plt.Axes,
    models: list[tuple[str, list[RunTuple]]],
    title: str,
    ylabel: str,
) -> None:
    """Copy of plot_BO_IDETC._draw_subplot but kept local to avoid circular imports."""
    palette = _get_palette(len(models))
    all_vals: list[float] = []
    for _, runs in models:
        _, mean_vals = _average_best_y_at_each_iteration(runs)
        all_vals.extend(v for v in mean_vals if not np.isnan(v))
        for _, hist, _ in runs:
            all_vals.extend(hist)
    y_range = max(all_vals) - min(all_vals) if all_vals else 1.0
    tick_h = y_range * 0.11

    # Individual runs
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        for run_id, hist, _ in runs:
            iters = list(range(1, len(hist) + 1))
            ax.plot(iters, hist, color=color, alpha=0.75, linestyle="-", linewidth=0.4, zorder=1)

    # Average lines
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        avg_color = _darker_color(color, factor=COLOR_FACTOR)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            ax.plot(
                iters,
                mean_vals,
                color=avg_color,
                linestyle="--",
                linewidth=2.5,
                label=label,
                alpha=0.85,
                zorder=2,
            )

    # Tick marks at average iteration
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        avg_color = _darker_color(color, factor=COLOR_FACTOR)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            avg_n_iters = np.mean([len(r[1]) for r in runs])
            idx_at_avg = int(round(avg_n_iters)) - 1
            if 0 <= idx_at_avg < len(mean_vals):
                x_tick = iters[idx_at_avg]
                y_tick = mean_vals[idx_at_avg]
                ax.plot(
                    [x_tick, x_tick],
                    [y_tick - tick_h / 2, y_tick + tick_h / 2],
                    color=avg_color,
                    linewidth=2,
                    linestyle=":",
                    zorder=3,
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def create_overleaf_plots_and_tex() -> None:
    """Generate per-problem PNGs (clean y) and a LaTeX snippet with 3 subfigures per row."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only use uncommented DEFAULT_PATHS entries from plot_BO_IDETC
    models = [(SCRIPT_DIR / p, label) for p, label in DEFAULT_PATHS]

    # Preload all runs for clean y once per model
    all_data: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    for model_dir, _ in models:
        data, _ = collect_runs(model_dir, use_clean_y=True)
        all_data.append(data)

    y_label = "Best y (clean)"
    clean_suffix = "_clean"

    # We create metadata lists separately for low and high noise
    noise_configs = [
        (NOISE_LOW, "0.002", "low"),
        (NOISE_HIGH, "0.08", "high"),
    ]

    tex_lines: list[str] = []

    for noise_key, noise_label, noise_name in noise_configs:
        metas = _build_problem_tex_meta(noise_key)

        # 1. Make individual PNGs for this noise level
        for meta in metas:
            key = (meta.folder, meta.dim_key, meta.noise_key)
            model_runs: list[tuple[str, list[RunTuple]]] = []
            for j, (_, label) in enumerate(models):
                runs = all_data[j].get(key, [])
                model_runs.append((label, runs))

            is_max = (meta.folder, meta.dim_key) in MAXIMIZATION_PROBLEMS
            if is_max:
                plot_runs = [
                    (label, [(run_id, [-v for v in hist], t) for run_id, hist, t in runs])
                    for label, runs in model_runs
                ]
                subplot_ylabel = "Best (-y) (clean)"
            else:
                plot_runs = model_runs
                subplot_ylabel = y_label

            fig, ax = plt.subplots(figsize=(4, 3))
            _draw_single_subplot(ax, plot_runs, meta.name, ylabel=subplot_ylabel)
            fig.tight_layout()

            out_file = OUT_DIR / f"{meta.slug}_noise{noise_label}{clean_suffix}.png"
            fig.savefig(out_file, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_file}")

        # 2. Append LaTeX figure* environment for this noise level (3 subfigures per row)
        tex_lines.append(r"\begin{figure*}[hptb]")
        tex_lines.append(r"    \centering")
        tex_lines.append(r"    \captionsetup{font=footnotesize}")

        for idx, meta in enumerate(metas):
            # 3 subfigures per row: after every 3rd, avoid trailing \hfill to help spacing
            tex_lines.append(r"    \begin{minipage}[b]{0.32\textwidth}")
            tex_lines.append(r"        \centering")
            tex_lines.append(
                rf"        \includegraphics[width=\textwidth]{{results_BO/gpVpfn_BO_plots/{meta.slug}_noise{noise_label}{clean_suffix}.png}}"
            )
            tex_lines.append(
                rf"        \subcaption{{{meta.letter} {meta.name}: {meta.dim_caption}}}"
            )
            tex_lines.append(
                rf"        \label{{fig:bo_idetc_{meta.slug}_noise{noise_name}}}"
            )
            tex_lines.append(r"    \end{minipage}")
            if (idx + 1) % 3 != 0 and idx != len(metas) - 1:
                tex_lines.append(r"    \hfill")

        tex_lines.append(
            rf"    \caption{{Bayesian optimization performance across benchmark problems (Noise $={noise_label} \times y_{{\text{{std}}}}$).}}"
        )
        tex_lines.append(
            rf"    \label{{fig:bo_idetc_gpVpfn_noise{noise_name}}}"
        )
        tex_lines.append(r"\end{figure*}")
        tex_lines.append("")

    tex_path = OUT_DIR / "gpVpfn_BO_figures.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote LaTeX snippet to: {tex_path}")


def main() -> None:
    create_overleaf_plots_and_tex()


if __name__ == "__main__":
    main()

