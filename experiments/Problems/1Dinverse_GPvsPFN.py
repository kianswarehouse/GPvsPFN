import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import gpytorch
import time
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics, extract_parameter_statistics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from gpplus.training.optimizers import LBFGSScipy
from gpplus.training.parameter_initializer import DefaultParameterInitializer
from gpplus.likelihoods.multi_noise import MultiLikelihood

from data.load_experimental_data import generate_mf_inverse_1D_data

# import warnings
# warnings.filterwarnings("ignore")

def test_data_generation_and_visualize():
    """
    Test the 1D inverse data generation and create histograms to visualize the data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("=== Testing 1D Inverse Data Generation ===")
    
    # Generate test data
    set_seed(42)
    X_test, y_test = generate_mf_inverse_1D_data(samples_per_source=[10000], noise=[0.05], noise_type='gaussian')
    X_train, y_train = generate_mf_inverse_1D_data(samples_per_source=[10000], noise=[0.05], noise_type='gaussian')
    
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Train data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"y range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Create histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('1D Inverse Data Generation Analysis', fontsize=16)
    
    # Histogram of X features (all dimensions)
    axes[0, 0].hist(X_test[:, 0].numpy(), bins=50, alpha=0.7, label='X[:, 0]', color='blue')
    # axes[0, 0].hist(X_test[:, 1].numpy(), bins=50, alpha=0.7, label='X[:, 1]', color='red')
    # axes[0, 0].hist(X_test[:, 2].numpy(), bins=50, alpha=0.7, label='X[:, 2]', color='green')
    axes[0, 0].set_title('Input Features Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of y targets
    axes[0, 1].hist(y_test.numpy(), bins=50, alpha=0.7, color='purple')
    axes[0, 1].set_title('Target Values Distribution')
    axes[0, 1].set_xlabel('Target Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: X[:, 0] vs y
    axes[1, 0].scatter(X_test[:, 0].numpy(), y_test.numpy(), alpha=0.5, s=1)
    axes[1, 0].set_title('X[:, 0] vs Target')
    axes[1, 0].set_xlabel('X[:, 0]')
    axes[1, 0].set_ylabel('Target')
    axes[1, 0].grid(True, alpha=0.3)
    
    # # Scatter plot: X[:, 1] vs y
    # axes[1, 1].hist(X_train[:, 0].numpy(), bins=50, alpha=0.7, label='X[:, 0]', color='blue')
    # # axes[0, 0].hist(X_test[:, 1].numpy(), bins=50, alpha=0.7, label='X[:, 1]', color='red')
    # # axes[0, 0].hist(X_test[:, 2].numpy(), bins=50, alpha=0.7, label='X[:, 2]', color='green')
    # axes[1, 1].set_title('Input Features Distribution')
    # axes[1, 1].set_xlabel('Feature Value')
    # axes[1, 1].set_ylabel('Frequency')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('1Dinverse_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Histogram saved as '1Dinverse_data_analysis.png'")
    
    # Show basic statistics
    print("\n=== Data Statistics ===")
    print(f"X[:, 0] - Mean: {X_test[:, 0].mean():.3f}, Std: {X_test[:, 0].std():.3f}")
    # print(f"X[:, 1] - Mean: {X_test[:, 1].mean():.3f}, Std: {X_test[:, 1].std():.3f}")
    # print(f"X[:, 2] - Mean: {X_test[:, 2].mean():.3f}, Std: {X_test[:, 2].std():.3f}")
    print(f"y - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}")
    
    # # Check for any categorical patterns in X[:, 2]
    # unique_values = torch.unique(X_test[:, 2])
    # print(f"\nX[:, 2] unique values: {unique_values.tolist()}")
    # print(f"X[:, 2] appears to be {'categorical' if len(unique_values) < 10 else 'continuous'}")
    
    plt.show()
    
    return X_test, y_test, X_train, y_train


def inverse_1D_GPvsPFN(
        num_seeds=20,
        num_test=[2500, 2500],
        train_size=[10, 10], # total training size is train_size * number of X input dimensions
        num_runs=16, 
        num_epochs=10000, 
        lr=0.1, 
        convergence_patience=10,
        optimizer_class=LBFGSScipy,
        initializer_class=DefaultParameterInitializer,
        gp_device='cpu',
        amp_device='cuda',
        save_path='./results/buckling',
        title=None,
        standardize_X_gp=True,
        standardize_y_gp=True,
        noise_train=[0.0, 0.0],
        noise_test=[0.0, 0.0],
        noise_type='gaussian',
        encode_PFN_data=False,
    ):

    if title is None:
        title = f"1Dinverse_{train_size[0]}D_{num_epochs}epochs_{num_runs}runs_{lr}"
    else: 
        title = f"1Dinverse_train{train_size[0]}D_{title}"

    amp_dtype = torch.float32
    dtype = torch.float64
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    
    # Set the correct path to the TabPFN checkpoint
    tabpfn_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gpplus', 'tabpfn', 'model', 'tabpfn-v2-regressor.ckpt')
    tabpfn_model_path = os.path.abspath(tabpfn_model_path)
    print(f"TabPFN model path: {tabpfn_model_path}")
    print(f"File exists: {os.path.exists(tabpfn_model_path)}")
    
    regressor = VanillaDirectTabPFNRegressor(device=amp_device, model_path=tabpfn_model_path)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Generate data
    set_seed(0)
    
    # Calculate total samples needed - use number of sources based on train_size length
    num_sources = len(train_size)
    train_per_seed = np.array(train_size)  # Use all sources
    total_train = num_seeds * train_per_seed
    total_samples = sum(num_test[:num_sources]) + sum(total_train)  # Use corresponding test sizes
    
    # Generate all unique Sobol samples at once - multiple sources
    print(f"Generating {total_samples} unique Sobol samples for {num_sources} sources...")
    print(f"Train samples per source: {train_per_seed}")
    print(f"Test samples per source: {num_test[:num_sources]}")
    
    X_test, y_test = generate_mf_inverse_1D_data(samples_per_source=num_test[:num_sources], noise=noise_test[:num_sources], noise_type=noise_type)
    X_train_all, y_train_all = generate_mf_inverse_1D_data(samples_per_source=total_train, noise=noise_train[:num_sources], noise_type=noise_type)
    
    X = torch.cat([X_test, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    # Since we're using only one source, we don't need source column encoding
    X_enc_train_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, noncont_dict=qual_dict, source_col=None)
    X_enc_test, _, _, _ = encode_qual_data(X_test, noncont_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Multi-source randomization and splitting
    total_train_samples = len(X_train_all)
    train_indices = torch.arange(total_train_samples)
    train_indices = train_indices[torch.randperm(len(train_indices))]
    
    # Reshape indices for multi-source case
    if num_sources == 1:
        train_indices_2d = train_indices.reshape(num_seeds, train_per_seed[0])
    else:
        # For multiple sources, we need to handle the reshaping differently
        # Each seed gets samples from all sources
        samples_per_seed = sum(train_per_seed)
        train_indices_2d = train_indices.reshape(num_seeds, samples_per_seed)
        
    total_start_time = time.time()
    for i in range(num_seeds):
        seed = i  # deterministic seed per split
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()

        # Get training indices for this seed
        seed_train_indices = train_indices_2d[i]

        X_train = X_train_all[seed_train_indices]
        X_enc_train = X_enc_train_all[seed_train_indices]
        y_train = y_train_all[seed_train_indices]

        print(f"Training samples for seed {i}: {len(X_train)}")
        if num_sources > 1:
            # Print source distribution for multi-source case
            source_counts = {}
            for j in range(num_sources):
                source_mask = X_train[:, 1] == j  # Assuming source is in column 1
                source_counts[f"source_{j}"] = source_mask.sum().item()
            print(f"Source distribution: {source_counts}")

        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")
        
        # Reuse PFN split, convert to torch
        X_gp_train = X_enc_train.detach().clone().to(dtype=dtype)
        X_gp_test = X_enc_test.detach().clone().to(dtype=dtype)
        y_gp_train = y_train.detach().clone().to(dtype=dtype)
        y_gp_test = y_test.detach().clone().to(dtype=dtype)
        if standardize_X_gp:
            Xscaler = gpplus.utils.StandardScaler()
            Xscaler.fit(X_gp_train[:, cont_cols])
            X_gp_train[:, cont_cols] = Xscaler.transform(X_gp_train[:, cont_cols])
            X_gp_test[:, cont_cols] = Xscaler.transform(X_gp_test[:, cont_cols])

        # Normalize the GP data
        y_gp_train_mean = y_gp_train.mean()
        y_gp_train_std = y_gp_train.std()
        y_gp_train_normal = (y_gp_train - y_gp_train_mean) / y_gp_train_std

        # cat_cols was returned by the encoder; CombinedKernel expects only cat indices
        # print(cat_cols)
        # Use simple GaussianKernel since we're not using source information
        print(f"X_gp_train shape: {X_gp_train.shape}")
        kernel = gpplus.kernels.LogScaleKernel(gpplus.kernels.GaussianKernel(ard_num_dims=X_gp_train.shape[1]))    

        # Create GP model
        model = gpplus.models.GPR(
            X_gp_train,
            y_gp_train_normal if standardize_y_gp else y_gp_train,
            kernel_module=kernel,
            mean_module=gpytorch.means.ConstantMean(),
            likelihood=gpplus.likelihoods.LogGaussianLikelihood(),
        )
        if (i == 0) or (i == num_seeds - 1):
            print(model)

        # Create trainer
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp(
            model,
            X_gp_test,
            y_gp_test,
            num_epochs=num_epochs,
            seed=seed,
            num_runs=num_runs,
            lr=lr,
            convergence_patience=convergence_patience,
            optimizer_class=optimizer_class,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_gp_train_mean if standardize_y_gp else None,
            y_train_std=y_gp_train_std if standardize_y_gp else None,
        )
        GPPlus_metrics.append(gp_metric)

        print(f"\nGP Results (Seed {seed}) [{i+1}/{num_seeds}]")
        from gpplus.utils.metrics_functions import format_metric_value
        for k, v in gp_metric.items():
            print(f"  {k}: {format_metric_value(k, v)}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")
        
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_enc_train if encode_PFN_data else X_train,
            X_enc_test if encode_PFN_data else X_test,
            y_train,
            y_test,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            regressor=regressor,
        )
        TabPFN_metrics.append(tabpfn_metric)

        # Print results for this seed
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
        # Collect model info from first seed
        if i == 0:
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_gp_train.shape[1],
                "train_samples_per_source": train_per_seed.tolist(),
                "test_samples_per_source": num_test[:num_sources],
                "num_sources": num_sources,
                "y_train_mean": float(y_gp_train_mean.item()),
                "y_train_std": float(y_gp_train_std.item()),
                "standardize_X_gp": standardize_X_gp,
                "standardize_y_gp": standardize_y_gp,
                "dtype": str(dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_runs": num_runs,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None
            }
            tabpfn_model_info = {
                "model_path": regressor.model_path,
                "fit_mode": regressor.fit_mode,
                "device": str(regressor.device_),
                "inference_precision": regressor.inference_precision,
                "random_state": regressor.random_state,
                "use_autocast": regressor.use_autocast_,
                "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
                "encoded_data": encode_PFN_data,
            }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title)
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Extract parameter statistics from gp_parameters.json
            param_stats = extract_parameter_statistics("gp_parameters.json")
            
            # Combined single file: TabPFN data + GP data + GP model_info + parameter stats
            combined_data = {
                "gp_data": {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                    "parameter_statistics": param_stats
                },
                "tabpfn_data": {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info
                },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {num_seeds} seeds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X_gp}\n\ty_standardize: {standardize_y_gp}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tseeds: {num_seeds}\n\tsources: {num_sources}\n\ttrain per source: {train_per_seed.tolist()}\n\ttest per source: {num_test[:num_sources]}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # buckling_GPvsPFN(num_seeds=4, train_size=[10, 10], num_runs=4, num_epochs=10000, save_path="./results/buckling/temp")
    # buckling_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path='./results/buckling/temp', standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=True)
    # buckling_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=True)
    # buckling_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=False, standardize_y_gp=False)
    # buckling_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path=None, standardize_X_gp=True, standardize_y_gp=False)
    # inverse_1D_GPvsPFN(num_seeds=1, num_runs=2, num_epochs=10000, save_path='./results/1Dinverse/temp', encode_PFN_data=True)
    # test_data_generation_and_visualize()
    inverse_1D_GPvsPFN(num_seeds=10, num_runs=4, num_epochs=10000, save_path='./results/1Dinverse/temp', encode_PFN_data=False, amp_device='cpu')