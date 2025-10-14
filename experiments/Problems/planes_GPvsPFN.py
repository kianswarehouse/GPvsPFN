import torch
import os
import sys
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
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor


# import warnings
# warnings.filterwarnings("ignore")

def load_2dplanes_data(print_info=False):
    """
    Load the planes dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y) where X is features and y are targets
    """
    # Path to the preferred CSV and a fallback TSV if CSV is missing
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    csv_path = os.path.join(data_dir, 'data_2dplanes.csv')
    tsv_path = os.path.join(data_dir, '215_2dplanes.tsv')

    # Load the data with graceful fallback
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.isfile(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None)
    else:
        raise FileNotFoundError(f"Could not find 2dplanes data. Expected one of: {csv_path} or {tsv_path}")
    

    # Convert to numpy array for processing; if header-like first row exists, drop it
    try:
        arr = df.values.astype(np.float64)
    except ValueError:
        # Drop the first row and retry conversion
        df = df.iloc[1:].reset_index(drop=True)
        arr = df.values.astype(np.float64)
    
    
    X = arr[:, :-1]  # shape: (n_samples, 10)
    

    y = arr[:, -1]
    
    return X, y


# Simple data info prints
# print("Encoded dataset info:")
# print(f"  X shape: {tuple(X_enc.shape)}  | y shape: {tuple(y_enc.shape)}")
# print("  First 5 X rows:\n", X_enc[:5])
# print("  First 5 y:", y_enc[:5])
# print("  Last 5 X rows:\n", X_enc[-5:])
# print("  Last 5 y:", y_enc[-5:])
def planes_GPvsPFN(num_seeds=20,
        num_test=5000,
        train_size=10,
        num_runs=16, 
        num_epochs=10000, 
        lr=0.1, 
        convergence_patience=10,
        optimizer_class=torch.optim.Adam,
        initializer_class=None,
        gp_device='cpu',
        amp_device='cuda',
        encode_PFN_data=False,
        save_path='./results/2dplanes',
        title=None,
    ):
    if title is None:
        title = f"2dplanes_{train_size}D_{num_epochs}epochs_{num_runs}runs_{lr}"
    else: 
        title = f"2dplanes_{train_size}D_{title}"
    # Load the data
    X, y = load_2dplanes_data()
    device = gp_device
    amp_dtype = torch.float32
    dtype = torch.float64
    print(f" GP Device: {device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    # Initialize results storage

    print("="*60)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*60)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(torch.tensor(X, dtype=dtype))
    print(qual_dict)
    X_enc, cont_cols, cat_cols, source_cols = encode_qual_data(torch.tensor(X, dtype=dtype), noncont_dict=qual_dict, source_col=None)
    print(cat_cols)
    TabPFN_planes_metrics = []
    GPPlus_planes_metrics = []
    set_seed(0)

    # Simple approach: get all indices, take first 5k for test, rest for training
    N = X_enc.shape[0]
    train_per_seed = train_size * X.shape[1]
    total_train = train_per_seed * num_seeds
    total_needed = num_test + total_train
    
    if N < total_needed:
        raise ValueError(f"Not enough data: need {total_needed}, have {N}")
    
    # Get deterministic permutation of all indices
    rng = np.random.RandomState(1337)
    all_indices = rng.permutation(N)
    
    # First 5k are test
    test_indices = all_indices[:num_test]

    # Remaining indices for training (repeated to get enough for all seeds)
    train_indices = all_indices[num_test:num_test+total_train]
    
    

    X_test_enc = X_enc[test_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Reshape into (num_seeds, train_per_seed) array
    train_indices_2d = train_indices.reshape(num_seeds, train_per_seed)

    total_start_time = time.time()
    for i in range(num_seeds):
        seed = i  # deterministic seed per split
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()

        # Get training indices for this seed
        seed_train_indices = train_indices_2d[i]

        X_train = X[seed_train_indices]
        y_train = y[seed_train_indices]

        X_train_enc = X_enc[seed_train_indices]


        # =============================================================================
        # GP planes Section 
        # =============================================================================
        print(f"\n--- {title} GP Training ---")

        # Reuse PFN split, convert to torch
        X_gp_train = X_train_enc.detach().clone().to(dtype=dtype)
        X_gp_test = X_test_enc.detach().clone().to(dtype=dtype)
        y_gp_train = torch.tensor(y_train, dtype=dtype)
        y_gp_test = torch.tensor(y_test, dtype=dtype)

        # Normalize the GP data
        y_gp_train_mean = y_gp_train.mean()
        y_gp_train_std = y_gp_train.std()
        y_gp_train_normal = (y_gp_train - y_gp_train_mean) / y_gp_train_std

        # cat_cols was returned by the encoder; CombinedKernel expects only cat indices
        # print(cat_cols)
        kernel = gpplus.kernels.CombinedKernel(cat_cols=cat_cols)

        # Create GP model
        model = gpplus.models.GPR(
            X_gp_train,
            y_gp_train_normal,
            kernel_module=kernel,
            # likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6), noise_prior=gpytorch.priors.LogNormalPrior(loc=np.log(y_gp_train_std**2), scale=1.0)),
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
            device=device,
            y_train_mean=y_gp_train_mean,
            y_train_std=y_gp_train_std,
        )
        GPPlus_planes_metrics.append(gp_metric)

        print(f"\nGP Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in gp_metric.items():
            print(f"  {k}: {v:.4f}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")
        
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_train_enc if encode_PFN_data else X_train,
            X_test_enc if encode_PFN_data else X_test,
            y_train,
            y_test,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            regressor=regressor,
        )
        TabPFN_planes_metrics.append(tabpfn_metric)

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
                "train_samples": X_gp_train.shape[0],
                "test_samples": X_gp_test.shape[0],
                "y_train_mean": float(y_gp_train_mean.item()),
                "y_train_std": float(y_gp_train_std.item()),
                "dtype": str(dtype),
                "device": str(device),
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
                "input_dim": X_gp_train.shape[1],
                "train_samples": X_gp_train.shape[0],
                "test_samples": X_gp_test.shape[0],
            }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_planes_summary = analyze_metrics(TabPFN_planes_metrics, print_summary=True, label="TabPFN")
    GPPlus_planes_summary = analyze_metrics(GPPlus_planes_metrics, print_summary=True, label="GP")
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        plot_metrics(TabPFN_planes_metrics, GPPlus_planes_metrics, labels=["TabPFN", "GP"], title="planes", save_path=save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # TabPFN single file: summary + metrics
            tabpfn_data = {
                "summary": TabPFN_planes_summary,
                "metrics": TabPFN_planes_metrics,
                "pfn_model_info": tabpfn_model_info
            }
            (out_dir / f"tabpfn_{title}.json").write_text(json.dumps(tabpfn_data, indent=2))
            
            # GP single file: summary + model_info + metrics
            gp_data = {
                "summary": GPPlus_planes_summary,
                "metrics": GPPlus_planes_metrics,
                "gp_model_info": gp_model_info
            }
            (out_dir / f"gp_{title}.json").write_text(json.dumps(gp_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {num_seeds} seeds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"GP trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {device}\n\tinitializer: {initializer_class}\n\tcat_cols: {cat_cols}\n\tqual_dict: {qual_dict}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tseeds: {num_seeds}")

    return GPPlus_planes_metrics, TabPFN_planes_metrics


if __name__ == "__main__":
    planes_GPvsPFN(num_seeds=4, num_runs=4, num_epochs=10000, save_path=None)