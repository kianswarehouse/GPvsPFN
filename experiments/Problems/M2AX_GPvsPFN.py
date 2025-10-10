import torch
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import gpytorch
import time
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN

# import warnings
# warnings.filterwarnings("ignore")

def load_m2ax_data(print_info=False):
    """
    Load the M2AX dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y_porosity, y_hardness, label_encoders) where X is features and y are targets
    """
    # Path to the data file (from one folder back, "data/data_M.csv")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_M.csv')
    data_path = os.path.abspath(data_path)
    
    # Load the data
    df = pd.read_csv(data_path)

    # Label-encode the first three categorical columns before casting to float
    categorical_columns = df.columns[:3]
    df_encoded = df.copy()
    for col in categorical_columns:
        df_encoded[col] = pd.factorize(df[col])[0]

    # Convert to numpy array for processing
    arr = df_encoded.values.astype(np.float64)
    
    # Remove rows with any NaN values
    mask = ~np.isnan(arr).any(axis=1)
    arr = arr[mask]
    
    # Features: first 6 columns (3 encoded element names + 3 numerical properties)
    X = arr[:, :3]  # shape: (n_samples, 6)
    
    #y = arr[:, -1]
    #print("E, Young's Modulus")

    # y = arr[:, -2]
    # print("G, Shear Modulus")

    y = arr[:, -3]
    # print("B, Bulk Modulus")

    return X, y


# Simple data info prints
# print("Encoded dataset info:")
# print(f"  X shape: {tuple(X_enc.shape)}  | y shape: {tuple(y_enc.shape)}")
# print("  First 5 X rows:\n", X_enc[:5])
# print("  First 5 y:", y_enc[:5])
# print("  Last 5 X rows:\n", X_enc[-5:])
# print("  Last 5 y:", y_enc[-5:])
def M2AX_GPvsPFN(num_seeds=20,
    test_size=0.2, 
    num_runs=16, 
    num_epochs=10000, 
    lr=0.1, 
    convergence_patience=10,
    optimizer_class=torch.optim.Adam,
    initializer_class=None,
    gp_device='cpu',
    amp_device='cuda',
    save_path='./results'
    ):
    # Load the data
    X, y = load_m2ax_data()
    title = f"M2AX_{test_size}test"
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None
        
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
    set_seed(0)  # Set global seed for reproducible seeds
    seeds = np.random.RandomState(0).choice(10**6, size=num_seeds, replace=False).tolist()
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} {title} SEED {i+1}/{num_seeds}: {seed} {'='*20}")
        t0 = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X_enc.numpy(), y_enc.numpy(), test_size=test_size, random_state=seed)
        # =============================================================================
        # GP M2AX Section 
        # =============================================================================
        print(f"\n--- {title} GP Evaluation ---")

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
            # likelihood=gpytorch.likelihoods.GaussianLikelihood(
                # noise_constraint=gpytorch.constraints.GreaterThan(1e-6), 
                # noise_prior=gpytorch.priors.LogNormalPrior(loc=np.log(y_gp_train_std**2), scale=1.0)),

        )
        if (i == 0) or (i == len(seeds) - 1):
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

        print(f"\nGP Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        for k, v in gp_metric.items():
            print(f"  {k}: {v:.4f}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        print(f"\n--- {title} TabPFN Evaluation ---")
        
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
        print(f"\nTabPFN Results (Seed {seed}) [{i+1}/{len(seeds)}]")
        for k, v in tabpfn_metric.items():
            print(f"  {k}: {v:.4f}")
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_M2AX_summary = analyze_metrics(TabPFN_M2AX_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_M2AX_summary = analyze_metrics(GPPlus_M2AX_metrics, print_summary=True, label="GP", title=title)
    
    if save_path is not None:
        plot_metrics(TabPFN_M2AX_metrics, GPPlus_M2AX_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            (out_dir / f"tabpfn_metrics_{title}.json").write_text(json.dumps(TabPFN_M2AX_metrics, indent=2))
            (out_dir / f"gp_metrics_{title}.json").write_text(json.dumps(GPPlus_M2AX_metrics, indent=2))
            (out_dir / f"tabpfn_summary_{title}.json").write_text(json.dumps(TabPFN_M2AX_summary, indent=2))
            (out_dir / f"gp_summary_{title}.json").write_text(json.dumps(GPPlus_M2AX_summary, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {len(seeds)} seeds: {time.time() - t0:.2f}s")
    print("="*60)
    print(f"GP trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_runs}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {device}")
    print(f"Experiment details: \n\ttest size: {test_size} ({len(X_test)} test samples, {len(X_train)} train samples)\n\tseeds: {seeds}")

    return GPPlus_M2AX_metrics, TabPFN_M2AX_metrics,

if __name__ == "__main__":
    # M2AX_GPvsPFN()
    M2AX_GPvsPFN(num_seeds=2, num_runs=1, num_epochs=10)