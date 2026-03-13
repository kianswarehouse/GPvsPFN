"""
Create violin plots comparing GP+ Defaults vs GP+ Custom vs other models.

Compares GP+ Defaults (baseline) against GP+ Custom (experimental) results alongside
other models (TabPFN v2.5, TabPFN v2, GPyTorch). Uses same SP_ISS filtering as
plot_violin_metrics_SP_ISS.py.

Inputs:
- GP+ Defaults directory: Path to default GP+ configuration results
- GP+ Custom directory: Path to custom/experimental GP+ results
- Optional directories for TabPFN v2.5, TabPFN v2, GPyTorch

Features:
- Same SP_ISS filters: Dx ∈ {10, 20, 40}, N ∈ {10Dx, 40Dx}, noise ∈ {0.005, 0.05}
- Enhanced legend showing both GP+ variants
- 4 y-axis ticks per plot
- 2x2 legend in top right corner
- Large readable fonts (22 for labels, 24 for y-axis)
- Legend only appears on Wing Weight plots

Output:
- Violin plots per problem showing RRMSE and NIS metrics
- GP+ Defaults shown in blue, GP+ Custom in orange for easy comparison
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def iter_json_files(base_dir: Path, problems: Optional[Set[str]]) -> Iterable[Path]:
    """Iterate over JSON files in base_dir, optionally filtering by problem names."""
    for path in base_dir.rglob("*.json"):
        rel = path.relative_to(base_dir)
        if problems:
            if not rel.parts:
                continue
            # Case-insensitive matching
            problem_lower = rel.parts[0].lower()
            problems_lower = {p.lower() for p in problems}
            if problem_lower not in problems_lower:
                continue
        yield path


def parse_meta_from_filename(path: Path, base_dir: Path) -> Dict[str, str]:
    """Parse metadata from filename structure."""
    rel_parts = path.relative_to(base_dir).parts
    problem_raw = rel_parts[0] if rel_parts else ""

    date = rel_parts[1] if len(rel_parts) > 1 else ""
    stem = path.stem

    # Prefer detecting problem name from the filename (stem) so that layout / directory
    # structure does not affect problem identification.
    stem_lower = stem.lower()
    known_problems = [
        "rosenbrock",
        "dixonprice",
        "ackley",
        "rastrigin",
        "griewank",
        "zakharov",
        "wing",
        "buckling",
        "borehole",
        "m2ax",
    ]

    problem = ""
    for pname in known_problems:
        if pname in stem_lower:
            problem = pname
            break

    # Fallback: infer problem from first path component if filename did not contain any
    # known problem token (this preserves compatibility with older directory layouts).
    if not problem:
        problem_clean = problem_raw.lower()
        problem_clean = problem_clean.replace("gpvpfn_", "")
        problem_clean = problem_clean.replace("gpytorch_", "")
        problem_clean = problem_clean.replace("seek_", "")
        problem = problem_clean.split("_")[0]

    # Try standard noise pattern first
    noise_match = re.search(r"noiseTest(?P<noiseTest>[0-9.]+)_noiseTrain(?P<noiseTrain>[0-9.]+)", stem)
    dim_match = re.search(r"([0-9]+)Dn", stem)
    if not dim_match:
        dim_match = re.search(r"_([0-9]+D)_", stem)
    xdim_match = re.search(r"([0-9]+)Dx", stem)
    if not xdim_match:
        xdim_match = re.search(r"([0-9]+)xdim", stem)

    # For M2AX and similar problems, try to extract test_size as noise label
    test_size_match = re.search(r"([0-9.]+)tsize", stem)

    noise_test = noise_match.group("noiseTest") if noise_match else ""
    noise_train = noise_match.group("noiseTrain") if noise_match else ""

    # If no noise pattern found but test_size found, use test_size as noise label
    if not noise_test and not noise_train and test_size_match:
        noise_test = test_size_match.group(1)
        noise_train = test_size_match.group(1)

    # Extract xdim from filename, or use problem-specific defaults
    xdim = xdim_match.group(1) if xdim_match else ""

    # If xdim not found in filename, use problem-specific defaults
    if not xdim:
        if problem == "wing":
            xdim = "10"
        elif problem == "buckling":
            xdim = "4"
        elif problem == "m2ax":
            xdim = "3"
        elif problem == "borehole":
            xdim = "8"

    # Extract dimension
    dim = ""
    if dim_match:
        dim_value = dim_match.group(1)
        if "Dn" in stem[dim_match.start() : dim_match.end()]:
            dim = f"{dim_value}D"
        else:
            dim = dim_match.group(1)

    return {
        "problem": problem,
        "date": date,
        "dim": dim,
        "xdim": xdim,
        "noiseTest": noise_test,
        "noiseTrain": noise_train,
    }


def extract_rows_from_section(
    data: Dict,
    section: str,
    model_family: str,
    meta: Dict[str, str],
) -> List[Dict]:
    """Extract rows from a data section."""
    rows: List[Dict] = []
    section_data = data.get(section, {})
    metrics_list = section_data.get("metrics")
    if not isinstance(metrics_list, list):
        return rows
    for m in metrics_list:
        if not isinstance(m, dict):
            continue
        # Fallbacks for TabPFN JSONs that store inference time under "Time"
        pred_time = m.get("Prediction_Time")
        if pred_time is None and model_family.startswith("tabpfn"):
            pred_time = m.get("Time")

        # GP+ (train_eval3) may write Train_Time instead of Training_Time
        train_time = m.get("Training_Time")
        if train_time is None and section == "gp_data":
            train_time = m.get("Train_Time")

        row = {
            "model_family": model_family,
            "problem": meta["problem"],
            "date": meta["date"],
            "dim": meta["dim"],
            "xdim": meta.get("xdim", ""),
            "noiseTest": meta["noiseTest"],
            "noiseTrain": meta["noiseTrain"],
            "RRMSE": m.get("RRMSE"),
            "RMSE": m.get("RMSE"),
            "NIS": m.get("NIS"),
            "Training_Time": train_time,
            "Total_Time": m.get("Total_Time"),
            "Prediction_Time": pred_time,
            "source_file": str(meta.get("source_file", "")),
        }
        rows.append(row)
    return rows


def collect_per_run_rows_enhanced(
    gpplus_default_dir: Optional[Path],
    gpplus_custom_dir: Optional[Path],
    gpytorch_dir: Optional[Path],
    tabpfn_v25_dir: Optional[Path],
    tabpfn_v2_dir: Optional[Path],
    extra_dirs: Optional[List[Tuple[Path, str]]] = None,
    problems: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Collect data from multiple GP+ sources and other models.
    
    Args:
        extra_dirs: List of (directory_path, model_name) tuples for additional directories.
    """
    rows: List[Dict] = []

    # Process GP+ Defaults
    if gpplus_default_dir is not None and gpplus_default_dir.exists():
        for path in iter_json_files(gpplus_default_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, gpplus_default_dir)
            meta["source_file"] = str(path)

            # Extract GP+ Defaults data
            if "gp_data" in data and "gpytorch" not in path.name.lower():
                rows.extend(extract_rows_from_section(data, "gp_data", "gpplus_default", meta))

    # Process GP+ Custom
    if gpplus_custom_dir is not None and gpplus_custom_dir.exists():
        for path in iter_json_files(gpplus_custom_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, gpplus_custom_dir)
            meta["source_file"] = str(path)

            # Extract GP+ Custom data
            if "gp_data" in data and "gpytorch" not in path.name.lower():
                rows.extend(extract_rows_from_section(data, "gp_data", "gpplus_custom", meta))

    # Process GPyTorch files (any file in gpytorch_dir with gp_data is treated as GPyTorch;
    # wing/borehole/buckling use gpVpfn_*_SF_* naming without "gpytorch" in the filename)
    if gpytorch_dir is not None and gpytorch_dir.exists():
        for path in iter_json_files(gpytorch_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, gpytorch_dir)
            meta["source_file"] = str(path)

            if "gp_data" in data:
                rows.extend(extract_rows_from_section(data, "gp_data", "gpytorch", meta))

    # Process TabPFN v2.5 files
    if tabpfn_v25_dir and tabpfn_v25_dir.exists():
        for path in iter_json_files(tabpfn_v25_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, tabpfn_v25_dir)
            meta["source_file"] = str(path)

            if "tabpfn_data" in data:
                rows.extend(extract_rows_from_section(data, "tabpfn_data", "tabpfn_v2.5", meta))

    # Process TabPFN v2 files
    if tabpfn_v2_dir and tabpfn_v2_dir.exists():
        for path in iter_json_files(tabpfn_v2_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, tabpfn_v2_dir)
            meta["source_file"] = str(path)
            rows.extend(extract_rows_from_section(data, "tabpfn_data", "tabpfn_v2", meta))

    # Process extra directories
    if extra_dirs:
        for extra_dir, model_name in extra_dirs:
            if extra_dir is not None and extra_dir.exists():
                for path in iter_json_files(extra_dir, problems):
                    with path.open("r") as f:
                        data = json.load(f)
                    meta = parse_meta_from_filename(path, extra_dir)
                    meta["source_file"] = str(path)

                    # Try gp_data first (for GP+ variants), then tabpfn_data
                    if "gp_data" in data and "gpytorch" not in path.name.lower():
                        rows.extend(extract_rows_from_section(data, "gp_data", model_name, meta))
                    elif "tabpfn_data" in data:
                        rows.extend(extract_rows_from_section(data, "tabpfn_data", model_name, meta))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize noise label for plotting
    noise_col = df["noiseTrain"].fillna("")
    fallback = df["noiseTest"].fillna("")
    df["noise_label"] = noise_col.where(noise_col != "", other=fallback)
    df["dim"] = df["dim"].replace("", "all")
    df["xdim"] = df["xdim"].replace("", "all")
    df["noise_label"] = df["noise_label"].replace("", "unknown")

    # SP_ISS-style filter: keep only a reasonable set of N values for non-special problems,
    # but allow both N = 5D_x and larger multiples so that e.g. Ackley 5D and 20D are kept.
    special_problems = ["wing", "buckling", "borehole", "m2ax"]
    non_special_df = df[~df["problem"].isin(special_problems)]
    special_df = df[df["problem"].isin(special_problems)]

    # Keep common N values (in units of D_x) and drop any odd extras
    allowed_dims = {"5D", "10D", "20D", "40D", "80D"}
    non_special_df = non_special_df[non_special_df["dim"].isin(allowed_dims)]

    # Combine back together
    df = pd.concat([non_special_df, special_df])

    return df


def get_model_order_enhanced(
    gpplus_default_dir: Optional[Path] = None,
    gpplus_custom_dir: Optional[Path] = None,
    gpytorch_dir: Optional[Path] = None,
    tabpfn_v25_dir: Optional[Path] = None,
    tabpfn_v2_dir: Optional[Path] = None,
    extra_dirs: Optional[List[Tuple[Path, str]]] = None,
) -> List[str]:
    """Get model order including GP+ Defaults, GP+ Custom, and extra directories."""
    models = []

    # Add GP+ variants (always in this order)
    if gpplus_default_dir is not None:
        models.append("gpplus_default")
    if gpplus_custom_dir is not None:
        models.append("gpplus_custom")

    # Add extra directories (in order provided)
    if extra_dirs:
        for _, model_name in extra_dirs:
            models.append(model_name)

    # Add other models (optional)
    if tabpfn_v25_dir is not None:
        models.append("tabpfn_v2.5")
    if tabpfn_v2_dir is not None:
        models.append("tabpfn_v2")
    if gpytorch_dir is not None:
        models.append("gpytorch")

    return models


def get_display_names(extra_dirs: Optional[List[Tuple[Path, str]]] = None) -> Dict[str, str]:
    """Get mapping from internal model names to display names.
    
    Args:
        extra_dirs: List of (directory_path, model_name) tuples. The model_name
                    will be used as both the internal key and display name unless
                    it's a known model.
    """
    display_names = {
        "gpplus_default": "GP+",
        "gpplus_custom": "GP+ (PE)",
        "tabpfn_v2.5": "PFN 2.5",
        "tabpfn_v2": "PFN 2.0",
        "gpytorch": "GPyTorch",
    }
    
    # Add extra directories - use model_name as display name
    if extra_dirs:
        for _, model_name in extra_dirs:
            # Only add if not already in the dict (allows override)
            if model_name not in display_names:
                display_names[model_name] = model_name
    
    return display_names


def format_label_for_plot(label: str) -> str:
    """Format label for plotting, splitting on first space only for multi-word labels."""
    # For "GP+ PE Kernel", split only the first space to get "GP+\nPE Kernel"
    # For other labels, split all spaces as before
    if label == "GP+ PE Kernel":
        return "GP+\nPE Kernel"
    # For other multi-word labels, split all spaces
    # return label.replace(" ", "\n")
    return label


def format_problem_title(problem: str) -> str:
    """Format problem name for display."""
    problem_lower = problem.lower()
    if problem_lower == "wing":
        return "Wing Weight"
    elif problem_lower in ["dixonprice", "dixon"]:
        return "Dixon Price"
    elif problem_lower == "ackley":
        return "Ackley"
    elif problem_lower == "rosenbrock":
        return "Rosenbrock"
    elif problem_lower == "rastrigin":
        return "Rastrigin"
    elif problem_lower == "griewank":
        return "Griewank"
    elif problem_lower == "zakharov":
        return "Zakharov"
    return problem.title()


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Remove outliers using IQR method."""
    if len(data) == 0:
        return data

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    if iqr == 0:
        return data

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask]


def make_violin_plots(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: List[str],
    remove_outliers: bool = False,
    extra_dirs: Optional[List[Tuple[Path, str]]] = None,
) -> None:
    """Create violin plots with GP+ Defaults vs GP+ Custom comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    # Filter model_order to only include models that exist in the data
    available_models = set(df["model_family"].unique())
    model_order = [m for m in model_order if m in available_models]

    # Get display name mapping
    display_names = get_display_names(extra_dirs=extra_dirs)

    # Color map - GP+ Defaults in blue, GP+ Custom in orange
    model_color_map = {
        "gpplus_default": "#1f77b4",  # Blue
        "gpplus_custom": "#ff7f0e",  # Orange
        "tabpfn_v2.5": "#2ca02c",  # Green
        "tabpfn_v2": "#9467bd",  # Purple
        "gpytorch": "#8c564b",  # Brown
    }

    # Noise color map
    noise_color_map = {
        "0.0": "#1f77b4",
        "0.002": "#ff7f0e",  # Red
        "0.08": "#2ca02c",  # Green
    }
    extra_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # Handle non-M2AX problems
    non_m2ax_df = df[df["problem"] != "m2ax"]

    plot_count = 0
    for (problem, xdim), prob_xdim_df in non_m2ax_df.groupby(["problem", "xdim"]):
        if xdim == "":
            continue

        # Get dims and noise levels
        # For all problems, use whatever dimensions are present in the data (after the
        # earlier SP_ISS-style filtering), but order them so that smaller N (e.g. 5D)
        # appears above larger N (e.g. 20D) in the figure.
        raw_dims = [d for d in prob_xdim_df["dim"].unique() if d != "" and d != "all"]

        def _dim_sort_key(d: str) -> int:
            try:
                return int(d.rstrip("D"))
            except ValueError:
                return 0

        dims = sorted(raw_dims, key=_dim_sort_key)

        noise_levels = sorted([n for n in prob_xdim_df["noise_label"].unique() if n != "unknown"])

        if not dims or not noise_levels:
            continue

        # Create subplots - separate figures for RRMSE and NIS
        n_rows = len(dims)
        fig_width = 12
        fig_height = 3 * n_rows

        fig_rrmse, axes_rrmse = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
        fig_nis, axes_nis = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)

        # Handle single row case so that indexing is consistent
        if n_rows == 1:
            axes_rrmse = np.array([axes_rrmse])
            axes_nis = np.array([axes_nis])

        # Create plots for each dim
        for row_idx, dim in enumerate(dims):
            dim_df = prob_xdim_df[prob_xdim_df["dim"] == dim]

            # Filter models to only those present
            models = [m for m in model_order if m in dim_df["model_family"].unique()]
            if not models:
                continue

            # Get colors for each model
            model_colors: Dict[str, str] = {}
            for model in models:
                if model in model_color_map:
                    model_colors[model] = model_color_map[model]
                else:
                    model_colors[model] = "#9467bd"

            # Layout parameters
            width = 0.8
            inner_width = width / max(len(noise_levels), 1) if len(noise_levels) > 1 else width * 0.7

            # Plot RRMSE
            metric = "RRMSE"
            data_groups = []
            positions = []
            colors = []
            ax_rrmse = axes_rrmse[row_idx]

            for mi, model in enumerate(models):
                center = mi + 1
                start = center - width / 2 + inner_width / 2
                for ni, noise in enumerate(noise_levels):
                    arr = (
                        dim_df.loc[
                            (dim_df["model_family"] == model) & (dim_df["noise_label"] == noise),
                            metric,
                        ]
                        .dropna()
                        .astype(float)
                        .values
                    )
                    if arr.size == 0:
                        continue
                    if remove_outliers:
                        arr = remove_outliers_iqr(arr)
                    if arr.size == 0:
                        continue
                    xpos = start + ni * inner_width
                    data_groups.append(arr)
                    positions.append(xpos)
                    color = noise_color_map.get(noise, extra_colors[ni % len(extra_colors)])
                    colors.append(color)

            # Plot violins
            if data_groups:
                parts = ax_rrmse.violinplot(
                    data_groups,
                    positions=positions,
                    widths=inner_width * 0.9,
                    showmeans=False,
                    showmedians=False,
                    showextrema=True,
                )
                for pc, color in zip(parts["bodies"], colors):
                    pc.set_facecolor(color)
                    pc.set_edgecolor("black")
                    pc.set_alpha(0.7)

                # Add mean and median lines
                for i, arr in enumerate(data_groups):
                    if len(arr) == 0:
                        continue
                    mean_v = float(np.mean(arr))
                    med_v = float(np.median(arr))
                    xpos = positions[i]
                    ax_rrmse.hlines(
                        mean_v, xpos - inner_width * 0.5, xpos + inner_width * 0.5, colors="#ff69b4", linewidth=2
                    )
                    ax_rrmse.hlines(
                        med_v,
                        xpos - inner_width * 0.5,
                        xpos + inner_width * 0.5,
                        colors="black",
                        linewidth=2,
                        linestyles="--",
                    )

            # Configure RRMSE subplot
            ax_rrmse.set_xticks([mi + 1 for mi in range(len(models))])
            ax_rrmse.set_xticklabels([format_label_for_plot(display_names.get(m, m)) for m in models], fontsize=22)
            # ax_rrmse.set_ylabel(metric, fontsize=24)
            ax_rrmse.tick_params(axis="both", which="major", labelsize=22)
            ax_rrmse.locator_params(axis="y", nbins=4)
            ax_rrmse.grid(False)
            ax_rrmse.grid(axis="y", linestyle=":", alpha=0.4, which="major")
            for boundary in [mi + 0.5 for mi in range(len(models) + 1)]:
                ax_rrmse.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=10)

            # Add dim label
            dim_number = dim.rstrip("D") if dim.endswith("D") else dim
            ax_rrmse.text(
                0.02,
                0.98,
                f"N = {dim_number}D$_x$",
                transform=ax_rrmse.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Plot NIS
            metric_nis = "NIS"
            data_groups_nis = []
            positions_nis = []
            colors_nis = []
            ax_nis = axes_nis[row_idx]

            for mi, model in enumerate(models):
                center = mi + 1
                start = center - width / 2 + inner_width / 2
                for ni, noise in enumerate(noise_levels):
                    arr = (
                        dim_df.loc[
                            (dim_df["model_family"] == model) & (dim_df["noise_label"] == noise),
                            metric_nis,
                        ]
                        .dropna()
                        .astype(float)
                        .values
                    )
                    if arr.size == 0:
                        continue
                    if remove_outliers:
                        arr = remove_outliers_iqr(arr)
                    if arr.size == 0:
                        continue
                    xpos = start + ni * inner_width
                    data_groups_nis.append(arr)
                    positions_nis.append(xpos)
                    color = noise_color_map.get(noise, extra_colors[ni % len(extra_colors)])
                    colors_nis.append(color)

            # Plot NIS violins
            if data_groups_nis:
                parts_nis = ax_nis.violinplot(
                    data_groups_nis,
                    positions=positions_nis,
                    widths=inner_width * 0.9,
                    showmeans=False,
                    showmedians=False,
                    showextrema=True,
                )
                for pc, color in zip(parts_nis["bodies"], colors_nis):
                    pc.set_facecolor(color)
                    pc.set_edgecolor("black")
                    pc.set_alpha(0.7)

                # Add mean and median lines for NIS
                for i, arr in enumerate(data_groups_nis):
                    if len(arr) == 0:
                        continue
                    mean_v = float(np.mean(arr))
                    med_v = float(np.median(arr))
                    xpos = positions_nis[i]
                    ax_nis.hlines(
                        mean_v, xpos - inner_width * 0.5, xpos + inner_width * 0.5, colors="#ff69b4", linewidth=2
                    )
                    ax_nis.hlines(
                        med_v,
                        xpos - inner_width * 0.5,
                        xpos + inner_width * 0.5,
                        colors="black",
                        linewidth=2,
                        linestyles="--",
                    )

            # Configure NIS subplot
            ax_nis.set_xticks([mi + 1 for mi in range(len(models))])
            ax_nis.set_xticklabels([format_label_for_plot(display_names.get(m, m)) for m in models], fontsize=22)
            # ax_nis.set_ylabel(metric_nis, fontsize=24)
            ax_nis.tick_params(axis="both", which="major", labelsize=22)
            ax_nis.locator_params(axis="y", nbins=4)
            ax_nis.grid(False)
            ax_nis.grid(axis="y", linestyle=":", alpha=0.4, which="major")
            for boundary in [mi + 0.5 for mi in range(len(models) + 1)]:
                ax_nis.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=10)

            # Add dim label
            dim_number = dim.rstrip("D") if dim.endswith("D") else dim
            ax_nis.text(
                0.02,
                0.98,
                f"N = {dim_number}D$_x$",
                transform=ax_nis.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            
            plot_count += 1

        # Add legend (only for wing weight) - attach to both figures for consistency
        if noise_levels and "buckling" in problem.lower():
            legend_elements = []
            # Add noise level colors
            for i, noise in enumerate(noise_levels):
                color = noise_color_map.get(noise, extra_colors[i % len(extra_colors)])
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f"Noise {noise}"))
            # Add mean and median line styles
            legend_elements.append(plt.Line2D([0, 1], [0, 1], color="#ff69b4", linewidth=2, label="Mean"))
            legend_elements.append(
                plt.Line2D([0, 1], [0, 1], color="black", linewidth=2, linestyle="--", label="Median")
            )
            fig_rrmse.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.0, 0.45), ncol=2, fontsize=16)
            fig_nis.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.0, 0.45), ncol=2, fontsize=16)

        # Add overall title
        # title = f"{format_problem_title(str(problem))} - GP+ Comparison ($D_x$ = {xdim})"
        # fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

        # Adjust layout and save - separate files for RRMSE and NIS
        fig_rrmse.tight_layout(rect=[0, 0, 1, 0.96])
        fig_nis.tight_layout(rect=[0, 0, 1, 0.96])
        suffix = "_no_outliers" if remove_outliers else ""
        outfile_rrmse = out_dir / f"{problem}_xdim{xdim}_gpplus_comparison_RRMSE{suffix}.pdf"
        outfile_nis = out_dir / f"{problem}_xdim{xdim}_gpplus_comparison_NIS{suffix}.pdf"
        fig_rrmse.savefig(outfile_rrmse, dpi=200, bbox_inches="tight", facecolor="white")
        fig_nis.savefig(outfile_nis, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_rrmse)
        plt.close(fig_nis)
        print(f"Saved {outfile_rrmse}")
        print(f"Saved {outfile_nis}")

    if plot_count == 0:
        print("No plots were generated. Check your input directories and data.")


def main():
    parser = argparse.ArgumentParser(
        description="Create violin plots comparing GP+ Defaults vs GP+ Custom with other models."
    )
    parser.add_argument(
        "--gpplus-default-dir",
        type=Path,
        required=False,
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_final/2_3_Compiled"),
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/10_runs_logging_full_Gaussian"),
        help="Directory containing GP+ Defaults (baseline) results.",
    ) 
    parser.add_argument(
        "--gpplus-custom-dir",
        type=Path,
        required=False,
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results/rosenbrock/2_11"),
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/10_folds_logging_full_PE"),
        help="Directory containing GP+ Custom (experimental) results.",
    )
    parser.add_argument(
        "--extra-dir",
        type=str,
        action="append",
        default=None,
        # default=["C:/Users/forty/tyler_gpplus/gp-private/results/kernel_test/dixon_price/gaussian",
        # "C:/Users/forty/tyler_gpplus/gp-private/results/kernel_test/dixon_price/power_exponential"],
        help="Extra directory to include. Format: 'DIR:NAME' where DIR is the path and NAME is the model name "
             "(e.g., 'GP+ v3'). Can be specified multiple times. Example: --extra-dir '/path/to/dir:GP+ v3'",
    )
    parser.add_argument(
        "--gpytorch-dir",
        type=Path,
        # default=None,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/10_folds_gpytorch"),
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results/kernel_test/dixon_price/power_exponential_softclamp"),
        help="Optional: Directory containing GPyTorch results.",
    )
    parser.add_argument(
        "--tabpfn-v25-dir",
        type=Path,
        # default=None,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/10_runs_PFN_V2.5"),
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_final/2_3_Compiled"),
        help="Optional: Directory containing TabPFN v2.5 results.",
    )
    parser.add_argument(
        "--tabpfn-v2-dir",
        type=Path,
        # default=None,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/10_runs_PFN_V2.0"),
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_final/pfnv2_0_IQR"),
        help="Optional: Directory containing TabPFN v2 results.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=None,
        help="Optional: Filter to specific problems (e.g., wing dixonprice ackley).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_March2/plots_gpplus_comparison"),
        help="Directory to write violin plot PDFs.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outliers using IQR method before plotting.",
    )


    args = parser.parse_args()

    # Validate required directories
    if not args.gpplus_default_dir.exists():
        raise ValueError(f"GP+ Defaults directory does not exist: {args.gpplus_default_dir}")
    if not args.gpplus_custom_dir.exists():
        raise ValueError(f"GP+ Custom directory does not exist: {args.gpplus_custom_dir}")

    # Parse extra directories
    extra_dirs: List[Tuple[Path, str]] = []
    if args.extra_dir:
        for extra_arg in args.extra_dir:
            # Convert to string if it's a Path object
            extra_arg_str = str(extra_arg) if isinstance(extra_arg, Path) else extra_arg

            # On Windows paths like "C:/foo/bar" contain a ":" for the drive
            # letter. We only treat ":" as a DIR:NAME separator if there is a
            # second ":" beyond the drive, or if the string does not look like
            # a simple drive path.
            has_drive = (
                len(extra_arg_str) >= 3
                and extra_arg_str[1] == ":"
                and extra_arg_str[0].isalpha()
            )
            colon_count = extra_arg_str.count(":")

            if colon_count >= 2 or (colon_count == 1 and not has_drive):
                # Treat the last ":" as the separator between DIR and NAME
                dir_path_str, model_name = extra_arg_str.rsplit(":", 1)
                dir_path = Path(dir_path_str)
                if not dir_path.exists():
                    raise ValueError(f"Extra directory does not exist: {dir_path}")
                extra_dirs.append((dir_path, model_name.strip()))
            else:
                # If no DIR:NAME separator, use directory name as model name
                dir_path = Path(extra_arg_str)
                if not dir_path.exists():
                    raise ValueError(f"Extra directory does not exist: {dir_path}")
                model_name = dir_path.name
                extra_dirs.append((dir_path, model_name))

    # Get problems filter
    problems: Optional[Set[str]] = None
    if args.problems:
        problems = set(p.lower() for p in args.problems)

    # Get model order based on provided directories
    model_order = get_model_order_enhanced(
        gpplus_default_dir=args.gpplus_default_dir,
        gpplus_custom_dir=args.gpplus_custom_dir,
        gpytorch_dir=args.gpytorch_dir,
        tabpfn_v25_dir=args.tabpfn_v25_dir,
        tabpfn_v2_dir=args.tabpfn_v2_dir,
        extra_dirs=extra_dirs if extra_dirs else None,
    )

    print(f"Collecting data from:")
    print(f"  GP+ Defaults: {args.gpplus_default_dir}")
    print(f"  GP+ Custom: {args.gpplus_custom_dir}")
    if args.gpytorch_dir:
        print(f"  GPyTorch: {args.gpytorch_dir}")
    if args.tabpfn_v25_dir:
        print(f"  TabPFN v2.5: {args.tabpfn_v25_dir}")
    if args.tabpfn_v2_dir:
        print(f"  TabPFN v2: {args.tabpfn_v2_dir}")
    for extra_dir, model_name in extra_dirs:
        print(f"  {model_name}: {extra_dir}")
    print(f"Models to compare: {[get_display_names(extra_dirs=extra_dirs if extra_dirs else None).get(m, m) for m in model_order]}")

    # Collect data
    df = collect_per_run_rows_enhanced(
        gpplus_default_dir=args.gpplus_default_dir,
        gpplus_custom_dir=args.gpplus_custom_dir,
        gpytorch_dir=args.gpytorch_dir,
        tabpfn_v25_dir=args.tabpfn_v25_dir,
        tabpfn_v2_dir=args.tabpfn_v2_dir,
        extra_dirs=extra_dirs if extra_dirs else None,
        problems=problems,
    )

    if df.empty:
        print("No data found. Check your input directories.")
        return

    print(f"\nCollected {len(df)} rows from {df['model_family'].nunique()} models")
    print(f"Problems: {sorted(df['problem'].unique())}")
    print(f"Dimensions: {sorted(df['xdim'].unique())}")

    # Create plots
    make_violin_plots(
        df,
        args.output_dir,
        model_order,
        remove_outliers=args.remove_outliers,
        extra_dirs=extra_dirs if extra_dirs else None,
    )

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
