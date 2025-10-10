import torch
import os
import sys
import pandas as pd
import numpy as np
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

# import warnings
# warnings.filterwarnings("ignore")

def load_2dplanes_data(print_info=False):
    """
    Load the M2AX dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y_porosity, y_hardness, label_encoders) where X is features and y are targets
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
    test_size=5000,
    train_size=10,
    num_runs=4, 
    num_epochs=10000, 
    lr=0.1, 
    convergence_patience=10,
    optimizer_class=torch.optim.Adam,
    initializer_class=None,
    gp_device='cpu',
    amp_device='cuda',
    save_path='./plots'):
    # Load the data
    X, y = load_2dplanes_data()
    title = f"2dplanes_{test_size}test"
    device = gp_device
    print(f"Device: {device}")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float32
    dtype = torch.float64

    # Initialize results storage

    print("="*60)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*60)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    _qual = learn_encodings(torch.tensor(X, dtype=dtype))

    X_enc, cat_cols, _source_cols = encode_qual_data(torch.tensor(X, dtype=dtype), noncont_dict=_qual, source_col=None)
    y_enc = torch.tensor(y, dtype=dtype)

    TabPFN_M2AX_metrics = []
    GPPlus_M2AX_metrics = []
    set_seed(0)

    # Build a deterministic fixed test set and expanded training pool
    N = X_enc.shape[0]
    # Interpret test_size as count if int, else as fraction
    if isinstance(test_size, float) and 0 < test_size < 1:
        num_test = int(round(test_size * N))
    else:
        num_test = int(test_size)
    num_test = min(num_test, N)

    rng_fixed_test = np.random.RandomState(1337)
    all_idx = np.arange(N)
    perm = rng_fixed_test.permutation(all_idx)
    test_indices = np.sort(perm[:num_test])
    train_base_indices = np.setdiff1d(all_idx, test_indices, assume_unique=False)
    base_train_len = int(train_base_indices.shape[0])
    if base_train_len <= 0:
        raise ValueError("No training data available after creating fixed test set.")

    # Expand training pool deterministically to train_size * base_train_len
    expanded_target = int(train_size) * base_train_len
    rng_train_pool = np.random.RandomState(2024)
    train_base_shuffled = rng_train_pool.permutation(train_base_indices)
    reps = int(np.ceil(expanded_target / float(base_train_len)))
    training_pool = np.tile(train_base_shuffled, reps)[:expanded_target]

    # Split training pool into num_seeds deterministic contiguous chunks
    chunk_sizes = [expanded_target // num_seeds] * num_seeds
    remainder = expanded_target % num_seeds
    for r in range(remainder):
        chunk_sizes[r] += 1
    starts = np.cumsum([0] + chunk_sizes[:-1])
    ends = np.cumsum(chunk_sizes)

    total_start_time = time.time()
    for i in range(num_seeds):
        seed = i  # deterministic seed per split
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()

        train_slice = slice(int(starts[i]), int(ends[i]))
        train_indices = training_pool[train_slice]

        X_train = X_enc.numpy()[train_indices]
        y_train = y_enc.numpy()[train_indices]
        X_test = X_enc.numpy()[test_indices]
        y_test = y_enc.numpy()[test_indices]


        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Training ---")
        
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            X_train,
            X_test,
            y_train,
            y_test,
            device=device,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
        )
        TabPFN_M2AX_metrics.append(tabpfn_metric)

        # Print results for this seed
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")

        # =============================================================================
        # GP M2AX Section (reuse encoded data and the same split)
        # =============================================================================
        print(f"\n--- {title} GP Training ---")

        # Reuse PFN split, convert to torch
        X_gp_train = torch.tensor(X_train, dtype=dtype)
        X_gp_test = torch.tensor(X_test, dtype=dtype)
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
        GPPlus_M2AX_metrics.append(gp_metric)

        print(f"\nGP Results (Seed {seed}) [{i+1}/{num_seeds}]")
        for k, v in gp_metric.items():
            print(f"  {k}: {v:.4f}")
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_M2AX_summary = analyze_metrics(TabPFN_M2AX_metrics, print_summary=True, label="TabPFN")
    GPPlus_M2AX_summary = analyze_metrics(GPPlus_M2AX_metrics, print_summary=True, label="GP")
    
    if save_path is not None:
        plot_metrics(TabPFN_M2AX_metrics, GPPlus_M2AX_metrics, labels=["TabPFN", "GP"], title="M2AX", save_path=save_path)
    print(f"\nTotal experiment time for {num_seeds} seeds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"GP trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {device}")
    print(f"Experiment details: \n\ttest size: {test_size} ({len(X_test)} test samples, {len(X_train)} train samples)\n\tseeds: {num_seeds}")

    return GPPlus_M2AX_metrics, TabPFN_M2AX_metrics


if __name__ == "__main__":
    planes_GPvsPFN(num_seeds=2, num_runs=1, num_epochs=10 )