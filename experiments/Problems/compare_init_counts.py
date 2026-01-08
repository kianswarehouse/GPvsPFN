"""
Compare results across different initialization counts (1init, 2init, 4init, 8init).
Creates plots with RRMSE on top and NIS on bottom in separate subplots.

Usage:
    This script expects results folders named:
    - prescreening_test_results_1init/
    - prescreening_test_results_2init/
    - prescreening_test_results_4init/
    - prescreening_test_results_8init/
    
    Each folder should contain subdirectories for each problem (wing/, borehole/, rosenbrock/)
    with subdirectories for conditions (with_prescreening/, without_prescreening/).
    
    The script will:
    1. Load metrics from JSON files in each folder
    2. Create comparison plots showing RRMSE (top) and NIS (bottom) for all init counts
    3. Save plots to ./init_comparison_plots/
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_metrics_from_json(json_path: Path, data_type: str = "gp") -> Optional[List[Dict[str, Any]]]:
    """Load metrics from a JSON file.
    
    Args:
        json_path: Path to JSON file
        data_type: "gp" for GP metrics, "tabpfn" for TabPFN metrics
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data_type == "gp" and 'gp_data' in data and 'metrics' in data['gp_data']:
            return data['gp_data']['metrics']
        elif data_type == "tabpfn" and 'tabpfn_data' in data and 'metrics' in data['tabpfn_data']:
            return data['tabpfn_data']['metrics']
        return None
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def find_json_files(base_folder: str, problem: str, condition: str, train_size: int) -> Optional[Path]:
    """Find the JSON file for a specific problem, condition, and train_size."""
    folder_path = Path(base_folder) / problem / condition
    if not folder_path.exists():
        print(f"  Folder does not exist: {folder_path}")
        return None
    
    # Look for JSON files matching the pattern
    # Files are typically named like: gpVpfn_borehole_SF_3D_100epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json
    # The title format is: borehole_SF_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}
    # Try multiple patterns
    patterns = [
        f"gpVpfn_*{train_size}D*.json",  # Pattern with train_sizeD (e.g., 3D)
        f"gpVpfn_*{train_size}_*.json",  # Pattern with train_size_ (e.g., train_size_3)
        f"gpVpfn_*.json"  # Fallback: any gpVpfn file
    ]
    
    for pattern in patterns:
        json_files = list(folder_path.glob(pattern))
        if json_files:
            # If multiple files match, try to find the one with train_size in the name
            if len(json_files) > 1:
                for f in json_files:
                    if f"{train_size}D" in f.stem or f"train_size_{train_size}" in f.stem:
                        return f
            return json_files[0]
    
    print(f"  No JSON file found in {folder_path} matching train_size={train_size}")
    return None

def extract_metric_values(metrics_list: List[Dict[str, Any]], metric_name: str) -> np.ndarray:
    """Extract values for a specific metric from a metrics list."""
    if not metrics_list:
        return np.array([], dtype=float)
    values = [d[metric_name] for d in metrics_list if isinstance(d, dict) and metric_name in d and d[metric_name] is not None]
    return np.array(values, dtype=float) if values else np.array([], dtype=float)

def create_violin_plot(ax, data: List[np.ndarray], labels: List[str], metric: str, n_seeds: int, 
                      violin_colors: Optional[List[str]] = None, init_counts: Optional[List[int]] = None) -> Optional[plt.Axes]:
    """
    Create a violin plot on the given axis with grouped backgrounds and borders.
    Returns an inset axes if 1init needs separate scaling, None otherwise.
    """
    if not data or all(len(arr) == 0 for arr in data):
        ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
        return None
    
    # Filter out empty arrays
    filtered_data = [arr for arr in data if len(arr) > 0]
    filtered_labels = [labels[i] for i, arr in enumerate(data) if len(arr) > 0]
    filtered_colors = None
    if violin_colors:
        filtered_colors = [violin_colors[i] for i, arr in enumerate(data) if len(arr) > 0]
    
    if not filtered_data:
        ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
        return None
    
    # Check if we need an inset for 1init
    # Group data by init count (assuming groups of 3)
    num_groups = len(filtered_data) // 3
    if num_groups > 0 and init_counts:
        # Get range for 1init (first group) and other inits
        one_init_data = filtered_data[:3]  # First 3 violins (1init group)
        other_init_data = filtered_data[3:]  # Rest
        
        one_init_values = np.concatenate([arr for arr in one_init_data if len(arr) > 0])
        other_init_values = np.concatenate([arr for arr in other_init_data if len(arr) > 0])
        
        inset_needed = False
        if len(one_init_values) > 0 and len(other_init_values) > 0:
            one_init_range = np.max(one_init_values) - np.min(one_init_values)
            other_init_range = np.max(other_init_values) - np.min(other_init_values)
            
            # If 1init range is more than 2x the other inits range, use inset
            if one_init_range > 0 and other_init_range > 0:
                if one_init_range / other_init_range > 2.0:
                    inset_needed = True
        
        if inset_needed:
            # Calculate y-axis limits for main axis (excluding 1init)
            if len(other_init_values) > 0:
                y_min = np.min(other_init_values)
                y_max = np.max(other_init_values)
                y_range = y_max - y_min
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
            else:
                y_min, y_max = 0, 1
            
            # Calculate y-axis limits for secondary axis (1init only)
            if len(one_init_values) > 0:
                y_min_secondary = np.min(one_init_values)
                y_max_secondary = np.max(one_init_values)
                y_range_secondary = y_max_secondary - y_min_secondary
                y_min_secondary -= y_range_secondary * 0.1
                y_max_secondary += y_range_secondary * 0.1
            else:
                y_min_secondary, y_max_secondary = 0, 1
            
            # Set main axis limits (for other inits)
            ax.set_ylim(y_min, y_max)
            
            # Create secondary y-axis for 1init
            ax2 = ax.twinx()
            ax2.set_ylim(y_min_secondary, y_max_secondary)
            ax2.set_ylabel(f"{metric} (1init)", fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=9)
            
            # Create background rectangles for all groups
            num_all_groups = len(filtered_data) // 3
            if len(filtered_data) % 3 != 0:
                num_all_groups += 1
            
            bg_colors = ['#e8e8e8', '#ffffff']
            for group_idx in range(num_all_groups):
                start_pos = group_idx * 3 + 0.5
                end_pos = min((group_idx + 1) * 3, len(filtered_data)) + 0.5
                color = bg_colors[group_idx % 2]
                
                rect = plt.Rectangle(
                    (start_pos, y_min),
                    end_pos - start_pos,
                    y_max - y_min,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=0
                )
                ax.add_patch(rect)
            
            # Plot 1init on secondary axis (positions 1, 2, 3) - same x positions as normal
            one_init_filtered = [arr for arr in one_init_data if len(arr) > 0]
            one_init_colors_filtered = None
            if filtered_colors:
                one_init_colors_filtered = [filtered_colors[i] for i in range(min(3, len(filtered_colors))) 
                                           if i < len(filtered_data) and len(filtered_data[i]) > 0]
            
            if one_init_filtered:
                parts_secondary = ax2.violinplot(one_init_filtered, positions=np.arange(1, len(one_init_filtered) + 1),
                                                 showmeans=False, showmedians=False, showextrema=True)
                for i, pc in enumerate(parts_secondary["bodies"]):
                    if one_init_colors_filtered and i < len(one_init_colors_filtered):
                        pc.set_facecolor(one_init_colors_filtered[i])
                    else:
                        pc.set_facecolor("#888888")
                    pc.set_edgecolor("gray")
                    pc.set_alpha(0.6)  # Slightly more transparent for secondary axis
                    pc.set_zorder(2)  # Above background, below main violins
                
                # Mean and median for secondary axis (lighter/thinner)
                for i, arr in enumerate(one_init_filtered, start=1):
                    if arr.size > 0:
                        mean_v = float(np.mean(arr))
                        med_v = float(np.median(arr))
                        ax2.hlines(mean_v, i - 0.25, i + 0.25, colors="blue", linewidth=1.5, alpha=0.7, zorder=3)
                        ax2.hlines(med_v, i - 0.25, i + 0.25, colors="red", linewidth=1.5, alpha=0.7, zorder=3)
            
            # Plot other inits on main axis (positions 4, 5, 6, 7, 8, 9, etc.)
            other_init_filtered = [arr for arr in other_init_data if len(arr) > 0]
            other_init_colors_filtered = None
            if filtered_colors:
                other_init_colors_filtered = [filtered_colors[i] for i in range(3, len(filtered_colors)) 
                                             if i < len(filtered_data) and len(filtered_data[i]) > 0]
            
            if other_init_filtered:
                # Plot violins for other inits (positions 4, 5, 6, 7, 8, 9, etc.)
                other_positions = np.arange(4, 4 + len(other_init_filtered))
                parts = ax.violinplot(other_init_filtered, positions=other_positions,
                                      showmeans=False, showmedians=False, showextrema=True)
                for i, pc in enumerate(parts["bodies"]):
                    if other_init_colors_filtered and i < len(other_init_colors_filtered):
                        pc.set_facecolor(other_init_colors_filtered[i])
                    else:
                        pc.set_facecolor("#888888")
                    pc.set_edgecolor("black")
                    pc.set_alpha(0.7)
                    pc.set_zorder(4)  # Above secondary axis violins
                
                # Mean and median for main plot
                for i, arr in enumerate(other_init_filtered, start=0):
                    if arr.size > 0:
                        mean_v = float(np.mean(arr))
                        med_v = float(np.median(arr))
                        pos = other_positions[i]
                        ax.hlines(mean_v, pos - 0.25, pos + 0.25, colors="blue", linewidth=2, zorder=5)
                        ax.hlines(med_v, pos - 0.25, pos + 0.25, colors="red", linewidth=2, zorder=5)
            
            # Set x-axis ticks: all init counts in their proper positions
            if init_counts:
                group_centers = []
                group_labels = []
                # First group center (1init) - position 2
                if len(one_init_filtered) > 0:
                    group_centers.append(2)
                    group_labels.append("1init")
                # Other group centers
                if other_init_filtered:
                    other_init_counts = init_counts[1:] if len(init_counts) > 1 else []
                    num_other_groups = len(other_init_filtered) // 3
                    for group_idx in range(num_other_groups):
                        center_pos = 4 + group_idx * 3 + 2  # Start at position 7 (4+3), then 10, etc.
                        if center_pos <= 4 + len(other_init_filtered):
                            group_centers.append(center_pos)
                            if group_idx < len(other_init_counts):
                                group_labels.append(f"{other_init_counts[group_idx]}init")
                            else:
                                group_labels.append(f"{(group_idx+2)*2}init")  # Fallback
                
                ax.set_xticks(group_centers)
                ax.set_xticklabels(group_labels)
            else:
                # Fallback: show all positions
                all_positions = list(range(1, len(filtered_data) + 1))
                ax.set_xticks(all_positions)
                ax.set_xticklabels(filtered_labels)
            
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{metric} distribution (n={n_seeds})")
            ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)
            
            return ax2
    
    # Normal plotting (no inset needed)
    # Calculate y-axis limits from data
    all_values = np.concatenate([arr for arr in filtered_data if len(arr) > 0])
    if len(all_values) > 0:
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
    else:
        y_min, y_max = 0, 1
    
    # Set y-axis limits first
    ax.set_ylim(y_min, y_max)
    
    # Add background rectangles for each group of 3 (one per init count)
    # Groups are: positions 1-3, 4-6, 7-9, 10-12 (for 1init, 2init, 4init, 8init)
    num_groups = len(filtered_data) // 3
    if len(filtered_data) % 3 != 0:
        num_groups += 1
    
    bg_colors = ['#e8e8e8', '#ffffff']  # Light grey and white
    for group_idx in range(num_groups):
        start_pos = group_idx * 3 + 0.5
        end_pos = min((group_idx + 1) * 3, len(filtered_data)) + 0.5
        color = bg_colors[group_idx % 2]
        
        # Add background rectangle with border
        rect = plt.Rectangle(
            (start_pos, y_min),
            end_pos - start_pos,
            y_max - y_min,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(rect)
    
    # Plot violins with colors
    parts = ax.violinplot(filtered_data, showmeans=False, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        if filtered_colors and i < len(filtered_colors):
            pc.set_facecolor(filtered_colors[i])
        else:
            pc.set_facecolor("#888888")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_zorder(2)  # Make sure violins are above background
    
    # Overlay mean (blue) and median (red) lines
    for i, arr in enumerate(filtered_data, start=1):
        if arr.size == 0:
            continue
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        ax.hlines(mean_v, i - 0.25, i + 0.25, colors="blue", linewidth=2, zorder=3)
        ax.hlines(med_v, i - 0.25, i + 0.25, colors="red", linewidth=2, zorder=3)
    
    # Legend: blue = mean, red = median
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="blue", lw=2, label="Mean"),
        Line2D([0], [0], color="red", lw=2, label="Median"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    
    # Set x-axis ticks: one per init count group (at the center of each group of 3)
    num_groups = len(filtered_data) // 3
    if len(filtered_data) % 3 != 0:
        num_groups += 1
    
    # X-axis positions: center of each group (at position 2, 5, 8, 11 for groups of 3)
    group_centers = [group_idx * 3 + 2 for group_idx in range(num_groups)]
    
    # Get init count labels
    if init_counts and len(init_counts) >= num_groups:
        init_labels = [f"{ic}init" for ic in init_counts[:num_groups]]
    else:
        # Fallback: assume 1, 2, 4, 8 pattern
        init_labels = [f"{2**i}init" for i in range(num_groups)]
    
    ax.set_xticks(group_centers)
    ax.set_xticklabels(init_labels)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution (n={n_seeds})")
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)
    
    return None

def plot_init_comparison(
    metrics_by_train_size: Dict[int, Dict[int, Dict[str, List[Dict[str, Any]]]]],  # {train_size: {init_count: {"with": [...], "without": [...], "tabpfn": [...]}}}
    problem_name: str,
    input_dim: int,  # Dx - input dimension
    save_path: Optional[str] = None
):
    """
    Create comparison plots with all train sizes on the same figure.
    Layout: RRMSE and NIS side-by-side, with increasing N going down.
    Shows all init counts (1, 2, 4, 8) on each subplot, with each init count showing
    three methods side-by-side: With Prescreening, Without Prescreening, TabPFN.
    
    Args:
        metrics_by_train_size: Dictionary mapping train_size to metrics_by_init dict
        problem_name: Name of the problem
        input_dim: Input dimension (Dx) for the problem
        save_path: Optional path to save the plot
    """
    # Get sorted train sizes
    train_sizes = sorted(metrics_by_train_size.keys())
    num_rows = len(train_sizes)
    
    # Create figure with subplots: num_rows × 2 (RRMSE left, NIS right)
    fig, axes = plt.subplots(num_rows, 2, figsize=(16, 4 * num_rows))
    
    # Handle case where there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Process each train_size
    for row_idx, train_size in enumerate(train_sizes):
        metrics_by_init = metrics_by_train_size[train_size]
        
        # Organize data by method, then by init count
        # Structure: {method: {init_count: values}}
        rrmse_by_method = {"Prescreening": [], "No Prescreening": [], "TabPFN": []}
        nis_by_method = {"Prescreening": [], "No Prescreening": [], "TabPFN": []}
        
        # Color scheme for different methods
        method_colors = {
            "Prescreening": "#1f77b4",      # Blue
            "No Prescreening": "#ff7f0e",    # Orange
            "TabPFN": "#2ca02c"             # Green
        }
        
        # Get sorted init counts
        sorted_init_counts = sorted(metrics_by_init.keys())
        
        # Process each init count in order (1, 2, 4, 8)
        for init_count in sorted_init_counts:
            init_data = metrics_by_init[init_count]
            
            # Add with_prescreening
            if "with" in init_data and init_data["with"]:
                rrmse_by_method["Prescreening"].append(extract_metric_values(init_data["with"], "RRMSE"))
                nis_by_method["Prescreening"].append(extract_metric_values(init_data["with"], "NIS"))
            else:
                rrmse_by_method["Prescreening"].append(np.array([], dtype=float))
                nis_by_method["Prescreening"].append(np.array([], dtype=float))
            
            # Add without_prescreening
            if "without" in init_data and init_data["without"]:
                rrmse_by_method["No Prescreening"].append(extract_metric_values(init_data["without"], "RRMSE"))
                nis_by_method["No Prescreening"].append(extract_metric_values(init_data["without"], "NIS"))
            else:
                rrmse_by_method["No Prescreening"].append(np.array([], dtype=float))
                nis_by_method["No Prescreening"].append(np.array([], dtype=float))
            
            # Add TabPFN (only once per init count)
            if "tabpfn" in init_data and init_data["tabpfn"]:
                rrmse_by_method["TabPFN"].append(extract_metric_values(init_data["tabpfn"], "RRMSE"))
                nis_by_method["TabPFN"].append(extract_metric_values(init_data["tabpfn"], "NIS"))
            else:
                rrmse_by_method["TabPFN"].append(np.array([], dtype=float))
                nis_by_method["TabPFN"].append(np.array([], dtype=float))
        
        # Prepare data for grouped violin plot
        # We need to create groups: for each init count, show all three methods
        rrmse_data = []
        nis_data = []
        rrmse_colors = []
        nis_colors = []
        
        # For each init count, add all three methods
        for init_count in sorted_init_counts:
            idx = sorted_init_counts.index(init_count)
            for method in ["Prescreening", "No Prescreening", "TabPFN"]:
                rrmse_data.append(rrmse_by_method[method][idx])
                nis_data.append(nis_by_method[method][idx])
                rrmse_colors.append(method_colors[method])
                nis_colors.append(method_colors[method])
        
        # Labels: repeat method names for each init count
        labels = []
        for init_count in sorted_init_counts:
            labels.extend(["Prescreening", "No Prescreening", "TabPFN"])
        
        # Determine n_seeds (min across all)
        seed_counts = []
        for init_data in metrics_by_init.values():
            if "with" in init_data and init_data["with"]:
                seed_counts.append(len(init_data["with"]))
            if "without" in init_data and init_data["without"]:
                seed_counts.append(len(init_data["without"]))
            if "tabpfn" in init_data and init_data["tabpfn"]:
                seed_counts.append(len(init_data["tabpfn"]))
        n_seeds = min(seed_counts) if seed_counts else 0
        
        # Get init count labels for x-axis
        sorted_init_counts = sorted(metrics_by_init.keys())
        
        # Plot RRMSE on left
        ax_rrmse = axes[row_idx, 0]
        create_violin_plot(ax_rrmse, rrmse_data, labels, "RRMSE", n_seeds, violin_colors=rrmse_colors, 
                          init_counts=sorted_init_counts)
        ax_rrmse.text(0.02, 0.98, f"$N = {train_size}D_x$", transform=ax_rrmse.transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot NIS on right
        ax_nis = axes[row_idx, 1]
        create_violin_plot(ax_nis, nis_data, labels, "NIS", n_seeds, violin_colors=nis_colors,
                          init_counts=sorted_init_counts)
        ax_nis.text(0.02, 0.98, f"$N = {train_size}D_x$", transform=ax_nis.transAxes,
                   fontsize=12, fontweight="bold", va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set overall title: "{Problem} - $D_x$ = {input_dim}"
    title = f"{problem_name.title()} - $D_x$ = {input_dim}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    
    # Add legend for methods (use colors from method_colors)
    from matplotlib.lines import Line2D
    method_colors = {
        "Prescreening": "#1f77b4",      # Blue
        "No Prescreening": "#ff7f0e",    # Orange
        "TabPFN": "#2ca02c"             # Green
    }
    
    # Create legend handles for methods
    method_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markersize=10, label=method, markeredgecolor="black", markeredgewidth=0.5)
        for method, color in method_colors.items()
    ]
    
    # Add mean/median handles
    mean_med_handles = [
        Line2D([0], [0], color="blue", lw=2, label="Mean"),
        Line2D([0], [0], color="red", lw=2, label="Median"),
    ]
    
    # Place legend on the right side of the figure
    fig.legend(
        handles=method_handles + mean_med_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        title="Methods",
        frameon=False,
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    
    # Save figure with format: {problem}_xdim{input_dim}_grouped_stacked_violin.png
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        filename = f"{problem_name.lower()}_xdim{input_dim}_grouped_stacked_violin.png"
        full_path = save_path_obj / filename
        fig.savefig(str(full_path), dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved plot to: {full_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    """Main function to compare results across different initialization counts."""
    
    # Configuration
    base_folders = {
        1: "prescreening_test_results_1init",
        2: "prescreening_test_results_2init",
        4: "prescreening_test_results_4init",
        8: "prescreening_test_results_8init"
    }
    
    problems = {
        "wing": "wing",
        "borehole": "borehole",
        "rosenbrock": "rosenbrock"
    }
    
    # Input dimensions (Dx) for each problem
    input_dims = {
        "wing": 10,
        "borehole": 8,
        "rosenbrock": 40
    }
    
    conditions = ["with_prescreening", "without_prescreening"]
    
    # Train sizes for each problem
    train_sizes = {
        "wing": [3, 5, 10, 20, 30],
        "borehole": [3, 5, 10, 20, 30],
        "rosenbrock": [10, 20]
    }
    
    save_path = "./init_comparison_plots"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # For each problem - create one plot with all train_sizes
    for problem_name, problem_folder in problems.items():
        print(f"\n{'='*80}")
        print(f"Processing: {problem_name}")
        print(f"{'='*80}")
        
        # Collect metrics for all train_sizes and init counts
        metrics_by_train_size = {}
        
        for train_size in train_sizes[problem_name]:
            print(f"\n  Processing train_size={train_size}...")
            metrics_by_init = {}
            
            for init_count in sorted(base_folders.keys()):
                print(f"    Processing {init_count}init...")
                
                init_data = {}
                
                # Load with_prescreening metrics
                with_base_folder = base_folders[init_count]
                with_json_path = find_json_files(with_base_folder, problem_folder, "with_prescreening", train_size)
                if with_json_path and with_json_path.exists():
                    with_metrics = load_metrics_from_json(with_json_path, data_type="gp")
                    if with_metrics:
                        init_data["with"] = with_metrics
                        print(f"      ✓ Loaded {len(with_metrics)} metrics (with prescreening)")
                
                # Load without_prescreening metrics
                without_json_path = find_json_files(with_base_folder, problem_folder, "without_prescreening", train_size)
                if without_json_path and without_json_path.exists():
                    without_metrics = load_metrics_from_json(without_json_path, data_type="gp")
                    if without_metrics:
                        init_data["without"] = without_metrics
                        print(f"      ✓ Loaded {len(without_metrics)} metrics (without prescreening)")
                
                # Load TabPFN metrics (same for both conditions, so just load once)
                if with_json_path and with_json_path.exists():
                    tabpfn_metrics = load_metrics_from_json(with_json_path, data_type="tabpfn")
                    if tabpfn_metrics:
                        init_data["tabpfn"] = tabpfn_metrics
                        print(f"      ✓ Loaded {len(tabpfn_metrics)} TabPFN metrics")
                
                if init_data:
                    metrics_by_init[init_count] = init_data
            
            if metrics_by_init:
                metrics_by_train_size[train_size] = metrics_by_init
        
        # Create one plot with all train_sizes if we have data
        if metrics_by_train_size:
            input_dim = input_dims[problem_name]
            plot_init_comparison(
                metrics_by_train_size,
                problem_name=problem_name,
                input_dim=input_dim,
                save_path=save_path
            )
            print(f"\n  ✓ Created plot for {problem_name}")
        else:
            print(f"\n  ✗ Skipping - no data found for any train_size")

if __name__ == "__main__":
    main()

