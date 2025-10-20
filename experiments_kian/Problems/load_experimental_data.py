import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def load_m2ax_data(print_info=False):
    """
    Load the M2AX dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y) where X is features and y are targets
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
    
    # Features: first 3 columns (3 encoded element names)
    X = torch.tensor(arr[:, :3], dtype=torch.float64)  # shape: (n_samples, 3)
    
    # Target: Bulk Modulus (column -3)
    y = torch.tensor(arr[:, -3], dtype=torch.float64)

    return X, y


def generate_hartmann_data(n_samples=10000, noise_level=0.0, noise_type='gaussian'):
    """
    Generate data for the 6D Hartmann function using Sobol sequences.
    
    The 6D Hartmann function is defined as:
    f(x) = -sum_{i=1}^{4} α_i * exp(-sum_{j=1}^{6} A_{ij} * (x_j - P_{ij})^2)
    
    where x ∈ [0,1]^6
    
    Args:
        n_samples (int): Number of samples to generate
        noise_level (float): Noise level as a fraction of the standard deviation of y
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        
    Returns:
        X (torch.Tensor): Input samples of shape (n_samples, 6)
        y (torch.Tensor): Function values of shape (n_samples,)
    """
    # Hartmann 6D function constants
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float64)
    
    A = torch.tensor([
        [10,   3,   17,  3.5, 1.7, 8],
        [0.05, 10,  17,  0.1, 8,   14],
        [3,    3.5, 1.7, 10,  17,  8],
        [17,   8,   0.05, 10,  0.1, 14]
    ], dtype=torch.float64)
    
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124,  8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=torch.float64)
    
    # Generate Sobol samples in [0,1]^6
    sobol = torch.quasirandom.SobolEngine(dimension=6, scramble=True)
    X = sobol.draw(n_samples).to(dtype=torch.float64)
    
    # Compute Hartmann function values using vectorized operations
    y = torch.zeros(n_samples, dtype=torch.float64)
    
    for k in range(4):  # 4 terms in the sum
        # Compute (x_j - P_{kj})^2 for all samples and dimensions
        diff = X - P[k:k+1, :]  # Broadcasting: (n_samples, 6) - (1, 6)
        squared_diff = diff ** 2  # (n_samples, 6)
        
        # Compute sum over dimensions: sum_j A_{kj} * (x_j - P_{kj})^2
        inner_sum = torch.sum(A[k:k+1, :] * squared_diff, dim=1)  # (n_samples,)
        
        # Add alpha_k * exp(-inner_sum) to the total
        y += alpha[k] * torch.exp(-inner_sum)
    
    # Apply negative sign
    y = -y
    
    # Add noise if specified
    if noise_level > 0:
        y_std = y.std()
        noise_scale = noise_level * y_std
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(y) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y) - 0.5) * 2 * noise_scale
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        
        y = y + noise
    
    return X, y


def load_am_dataset_data(print_info=False):
    """
    Load the AM dataset from the .pt file.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y_porosity, y_hardness) where X is features and y are targets
    """
    # Path to the data file (from one folder back, "data/am_data.pt")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'am_data.pt')
    data_path = os.path.abspath(data_path)
    
    # Load the data
    data = torch.load(data_path)
    
    arr = data
    mask = ~torch.isnan(arr).any(dim=1)
    
    # Apply mask to filter out all rows with any NaN
    arr = arr[mask]
    
    X = arr[:, :6]   # shape: (540, 6) or (539, 6) after removing NaN sample
    
    # Targets: column 6 = Porosity, column 7 = Hardness
    y_porosity = arr[:, 6]
    y_hardness = arr[:, 7]
    
    return X, y_porosity, y_hardness


def wing_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Local copy of the wing function with source-specific variants.
    X shape: (n, 10)
    """
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (torch.pi / 180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    cos_Gama = torch.cos(Gama)

    if source == "s0":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
    elif source == "s1":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s2":
        return (
            0.036
            * Sw**0.8
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s3":
        return (
            0.036
            * Sw**0.9
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 0 * Wp
        )
    else:
        raise ValueError(f"Unknown source: {source}")


def generate_mf_wing_data(samples_per_source: list[int], seed: int = 0, noise: list[float] = None, noise_type: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Wing data locally and return:
      - X with 11 features: 10 continuous + 1 source class column in {0,1,2,3}
      - y as target values

    Args:
        samples_per_source: List of number of samples from each source
        seed: Random seed
        noise: List of noise levels for each source
        noise_type: Type of noise ('gaussian' or 'uniform')
    """
    torch.manual_seed(seed)

    # Set default noise levels
    if noise is None:
        noise = [0.0] * 4

    # Bounds for the 10 continuous features (same as data_gen)
    l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025], dtype=torch.float64)
    u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08], dtype=torch.float64)

    sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=seed)

    counts = samples_per_source
    sources = ["s0", "s1", "s2", "s3"]

    X_list = []
    y_list = []

    for idx, (src, n) in enumerate(zip(sources, counts)):
        if n == 0:
            continue
        x_raw = sobol.draw(n).to(dtype=torch.float64)
        x_raw = x_raw * (u_bound - l_bound) + l_bound
        y = wing_mixed_variables(x_raw, source=src)
        
        # Add noise based on source
        noise_level = noise[idx]
        if noise_level > 0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y) * noise_level
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y) - 0.5) * 2 * noise_level
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y = y + noise
        
        # Append source class as the 11th feature
        src_col = torch.full((n, 1), float(idx), dtype=torch.float64)
        X_list.append(torch.cat([x_raw, src_col], dim=1))
        y_list.append(y)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y


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
    
    
    X = torch.tensor(arr[:, :-1], dtype=torch.float64)  # shape: (n_samples, 10)
    

    y = torch.tensor(arr[:, -1], dtype=torch.float64)
    
    return X, y


def buckling_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute buckling load given input variables.
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, 4] with columns:
            0: L (length of the beam, m)
            1: E (Young's modulus, Pa) 
            2: K (shear modulus, Pa)
            3: I (moment of inertia, m^4)
        source (str): Source of the data ('s0' or 's1')
    Returns:
        torch.Tensor: Buckling load values for each input sample
    """
    L = X[..., 0]
    E = X[..., 1]
    K = X[..., 2]
    I = X[..., 3]

    # Buckling load calculation
    if source == "s0":
        P = torch.pi * E * I / (L * K) ** 2
    elif source == "s1":
        P = ((torch.pi * E * I / (L * K) ** 2) + L) ** 1.1
    else:
        raise ValueError(f"Unknown source: {source}. Only 's0' and 's1' are supported for buckling.")

    return P


def generate_mf_buckling_data(samples_per_source: list[int], seed: int = 0, noise: list[float] = None, noise_type: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data locally and return:
      - X with 5 features: 4 continuous + 1 source class column in {0,1}
      - y as target values

    Args:
        samples_per_source: List of number of samples from each source (2 sources: s0, s1)
        seed: Random seed
        noise: List of noise levels for each source
        noise_type: Type of noise ('gaussian' or 'uniform')
    """
    torch.manual_seed(seed)

    # Set default noise levels
    if noise is None:
        noise = [0.0] * 2

    # Bounds for the 4 continuous features (L, E, K, I)
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=torch.float64)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=torch.float64)

    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)

    counts = samples_per_source
    sources = ["s0", "s1"]

    X_list = []
    y_list = []

    for idx, (src, n) in enumerate(zip(sources, counts)):
        if n == 0:
            continue
        x_raw = sobol.draw(n).to(dtype=torch.float64)
        x_raw = x_raw * (u_bound - l_bound) + l_bound
        y = buckling_mixed_variables(x_raw, source=src)
        
        # Add noise based on source
        noise_level = noise[idx]
        if noise_level > 0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y) * noise_level
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y) - 0.5) * 2 * noise_level
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y = y + noise
        
        # Append source class as the 5th feature
        src_col = torch.full((n, 1), float(idx), dtype=torch.float64)
        X_list.append(torch.cat([x_raw, src_col], dim=1))
        y_list.append(y)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y