"""
Variable Analysis Script for Analytic Problems

This script generates data for analytic problems and creates histogram plots
for each feature and target variable, comparing train and test distributions.
"""

import defaults
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from load_experimental_data import (
    generate_ackley_data,
    generate_rastrigin_data,
    generate_rosenbrock_data,
    generate_zakharov_data,
    generate_griewank_data,
    generate_dixon_price_data,
    generate_styblinski_tang_data,
    generate_mf_wing_data,
    generate_mf_buckling_data,
    generate_mf_borehole_data
)

# Configuration
N_TEST = 5000  # Standard test samples
SEED = defaults.SEED  # For reproducibility
SAVE_DIR = "./results/variable_analysis"

# Problem configurations: (name, generate_func, dimensions, train_size_multiplier, is_sf)
# train_size_multiplier: for n-dimensional problems, train_size = multiplier * dimensions
# is_sf: True if single-fidelity (uses MF function but only s0), False otherwise
PROBLEMS = [
    # A1-A3: Single-fidelity problems
    ("wing_SF", generate_mf_wing_data, 10, 20, True),  # 20D train (20 * 10 = 200)
    ("buckling_SF", generate_mf_buckling_data, 4, 20, True),  # 20D train (20 * 4 = 80)
    ("borehole_SF", generate_mf_borehole_data, 8, 20, True),  # 20D train (20 * 8 = 160)
    # A4-A7: N-dimensional problems
    ("ackley", generate_ackley_data, 20, 20, False),  # 20xdim for train
    ("rastrigin", generate_rastrigin_data, 20, 20, False),  # 20xdim for train
    ("rosenbrock", generate_rosenbrock_data, 20, 20, False),  # 20xdim for train
    ("zakharov", generate_zakharov_data, 20, 20, False),  # 20xdim for train
    ("griewank", generate_griewank_data, 20, 20, False),  # 20xdim for train
    ("dixon_price", generate_dixon_price_data, 20, 20, False),  # 20xdim for train
    ("styblinski_tang", generate_styblinski_tang_data, 20, 20, False),  # 20xdim for train
]


def plot_histograms(X_train, y_train, X_test, y_test, problem_name, dimensions, save_dir):
    """
    Create histogram plots for each feature and target variable.
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
        problem_name: Name of the problem
        dimensions: Number of dimensions
        save_dir: Directory to save plots
    """
    # Convert to numpy for plotting
    X_train_np = X_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()
    
    n_features = X_train.shape[1]
    
    # Create subplots: one row for each feature + one for target
    n_plots = n_features + 1
    n_cols = min(4, n_plots)  # Max 4 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    # Handle different subplot configurations
    if n_plots == 1:
        axes = np.array([axes])
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Plot each feature
    for i in range(n_features):
        ax = axes[i]
        
        # Get data for this feature
        train_data = X_train_np[:, i]
        test_data = X_test_np[:, i]
        
        # Create histograms
        ax.hist(test_data, bins=50, alpha=0.6, color='blue', label='Test', density=True)
        ax.hist(train_data, bins=50, alpha=0.6, color='orange', label='Train', density=True)
        
        ax.set_xlabel(f'Feature {i+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'{problem_name.capitalize()} - Feature {i+1} (dim={dimensions})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot target
    ax = axes[n_features]
    ax.hist(y_test_np, bins=50, alpha=0.6, color='blue', label='Test', density=True)
    ax.hist(y_train_np, bins=50, alpha=0.6, color='orange', label='Train', density=True)
    ax.set_xlabel('Target (y)')
    ax.set_ylabel('Density')
    ax.set_title(f'{problem_name.capitalize()} - Target (dim={dimensions})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / f"{problem_name}_{dimensions}D_histograms.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram plot to: {save_path}")
    plt.close()


def analyze_problem(problem_name, generate_func, dimensions, train_size_multiplier, is_sf=False):
    """
    Generate data and create histograms for a single problem.
    
    Args:
        problem_name: Name of the problem
        generate_func: Function to generate data
        dimensions: Number of dimensions
        train_size_multiplier: Multiplier for train size (train_size = multiplier * dimensions)
        is_sf: True if single-fidelity (uses MF function but only s0), False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {problem_name.upper()} (dimensions={dimensions})")
    print(f"{'='*60}")
    
    # Calculate train size
    n_train = train_size_multiplier * dimensions
    n_test = N_TEST
    
    print(f"Train size: {n_train}, Test size: {n_test}")
    
    # Generate data
    try:
        if is_sf:
            # Single-fidelity problems use MF functions but only s0 source
            if problem_name == "wing_SF":
                X_train, y_train, X_test, y_test = generate_func(
                    train_samples_per_source=[n_train, 0, 0, 0],
                    test_samples_per_source=[n_test, 0, 0, 0],
                    train_noise=0.0,
                    test_noise=0.0,
                    seed=SEED
                )
                # Drop the source column (11th column)
                if X_train.shape[1] == 11:
                    X_train = X_train[:, :10]
                if X_test.shape[1] == 11:
                    X_test = X_test[:, :10]
            elif problem_name == "buckling_SF":
                X_train, y_train, X_test, y_test = generate_func(
                    train_samples_per_source=[n_train, 0],
                    test_samples_per_source=[n_test, 0],
                    train_noise=[0.0, 0.0],
                    test_noise=[0.0, 0.0],
                    seed=SEED
                )
                # Drop the source column (5th column)
                if X_train.shape[1] == 5:
                    X_train = X_train[:, :4]
                if X_test.shape[1] == 5:
                    X_test = X_test[:, :4]
            elif problem_name == "borehole_SF":
                X_train, y_train, X_test, y_test = generate_func(
                    train_samples_per_source=[n_train, 0, 0, 0, 0],
                    test_samples_per_source=[n_test, 0, 0, 0, 0],
                    train_noise=0.0,
                    test_noise=0.0,
                    seed=SEED
                )
                # Drop the source column (9th column)
                if X_train.shape[1] == 9:
                    X_train = X_train[:, :8]
                if X_test.shape[1] == 9:
                    X_test = X_test[:, :8]
        else:
            # N-dimensional problems use standard generate functions
            X_train, y_train, X_test, y_test = generate_func(
                n_train=n_train,
                n_test=n_test,
                dimensions=dimensions,
                train_noise=0.0,
                test_noise=0.0,
                seed=SEED
            )
    except Exception as e:
        print(f"Error generating data for {problem_name}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Generated data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Print basic statistics
    print(f"\nTrain statistics:")
    print(f"  X_train - mean: {X_train.mean(dim=0)}, std: {X_train.std(dim=0)}")
    print(f"  y_train - mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"\nTest statistics:")
    print(f"  X_test - mean: {X_test.mean(dim=0)}, std: {X_test.std(dim=0)}")
    print(f"  y_test - mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
    
    # Create histograms
    plot_histograms(X_train, y_train, X_test, y_test, problem_name, dimensions, SAVE_DIR)


def main():
    """Main function to run variable analysis for all problems."""
    print("Variable Analysis for Analytic Problems")
    print("="*60)
    print(f"Test samples: {N_TEST}")
    print(f"Seed: {SEED}")
    print(f"Save directory: {SAVE_DIR}")
    
    # Analyze each problem
    for problem_config in PROBLEMS:
        try:
            if len(problem_config) == 5:
                problem_name, generate_func, dimensions, train_size_multiplier, is_sf = problem_config
            else:
                # Backward compatibility
                problem_name, generate_func, dimensions, train_size_multiplier = problem_config
                is_sf = False
            analyze_problem(problem_name, generate_func, dimensions, train_size_multiplier, is_sf)
        except Exception as e:
            print(f"Failed to analyze {problem_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Variable analysis complete!")
    print(f"Results saved to: {SAVE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

