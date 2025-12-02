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


def generate_mf_wing_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                         seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                         noise_type: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Wing data by drawing a single Sobol batch (no repeats),
    then splitting into test/train per source. Compute test std after the split
    and scale both train and test noise by that std.

    Returns:
      - X_train, y_train: Training data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
      - X_test, y_test: Test data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (4 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0, 0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0, 0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 4
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 4
    if len(train_noise) != 4 or len(test_noise) != 4:
        raise ValueError("train_noise and test_noise must be length-4 (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3"]

    # Bounds for the 10 continuous features
    l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025], dtype=torch.float64)
    u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08], dtype=torch.float64)

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once (scrambled => randomized QMC) and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Assign contiguous blocks per source (no repeats globally)
    src_indices = []
    start = 0
    for idx, n in enumerate(total_per_source):
        src_indices.extend([idx] * n)
        start += n
    src_indices_tensor = torch.tensor(src_indices, dtype=torch.long)

    # Compute clean targets once per source
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    offset = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[offset:offset + n]
        y_clean_all[offset:offset + n] = wing_mixed_variables(x_block, source=src)
        offset += n

    # Split per source into test then train; get test std after split
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)

        # Test std after split (per source) as a Python float
        test_std_value: float
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append source id as 11th feature
        if n_train > 0:
            src_col_train = torch.full((n_train, 1), float(idx), dtype=torch.float64)
            X_train_list.append(torch.cat([x_train_block, src_col_train], dim=1))
            y_train_list.append(y_train_block)

        if n_test > 0:
            src_col_test = torch.full((n_test, 1), float(idx), dtype=torch.float64)
            X_test_list.append(torch.cat([x_test_block, src_col_test], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 11), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 11), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


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


def generate_mf_buckling_data_with_folds(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                                         num_folds: int = 4, seed: int = None, train_noise: list[float] = None, 
                                         test_noise: list[float] = None, noise_type: str = 'gaussian', 
                                         return_categorical: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data with pre-stratified folds:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Generate train data directly as num_folds with even categorical distributions
      - Generate test data with even categorical distributions
      - Each fold has perfectly balanced categorical distributions
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Default noise values
    if train_noise is None:
        train_noise = [0.0] * len(train_samples_per_source)
    if test_noise is None:
        test_noise = [0.0] * len(test_samples_per_source)
    
    # Validate inputs
    if len(train_samples_per_source) != len(test_samples_per_source):
        raise ValueError("train_samples_per_source and test_samples_per_source must have same length")
    if len(train_noise) != len(train_samples_per_source):
        raise ValueError("train_noise must be length-2 (one scalar per source)")
    if len(test_noise) != len(test_samples_per_source):
        raise ValueError("test_noise must be length-2 (one scalar per source)")
    
    sources = ['s0', 's1']  # Two sources
    total_n = sum(train_samples_per_source) + sum(test_samples_per_source)
    
    # Categorical index values (0-based) and positive physical mappings
    # Use strictly positive physical values to avoid 0/0 or division-by-zero in buckling formula
    E_values = torch.tensor([0.0, 1.0], dtype=torch.float64)  # indices
    K_values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)  # indices
    I_values = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)  # indices
    E_phys = torch.tensor([1.0, 2.0], dtype=torch.float64)
    K_phys = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    I_phys = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    
    # Generate test data first (one block per source)
    X_test_list = []
    y_test_list = []
    test_std_per_source = {}  # Store test std for each source
    
    for src_idx, (src, n_test) in enumerate(zip(sources, test_samples_per_source)):
        if n_test == 0:
            continue
            
        # Generate continuous variables using Sobol
        sobol_seed = (seed + src_idx * 1000) if seed is not None else None
        sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=sobol_seed)
        L_vals = sobol.draw(n_test).squeeze() * 0.4 + 0.6  # L in [0.6, 1.0]
        
        # Create test block
        x_test_block = torch.zeros((n_test, 4), dtype=torch.float64)
        x_test_block[:, 0] = L_vals
        
        # Generate even categorical distributions for TEST
        # E values (2 options)
        n_per_E_test = n_test // len(E_values)
        remaining_E_test = n_test % len(E_values)
        E_indices_test = []
        for i in range(len(E_values)):
            count = n_per_E_test + (1 if i < remaining_E_test else 0)
            E_indices_test.append(torch.full((count,), i))
        E_indices_test = torch.cat(E_indices_test)
        E_indices_test = E_indices_test[torch.randperm(n_test)]
        x_test_block[:, 1] = E_phys[E_indices_test]

        # K values (4 options)
        n_per_K_test = n_test // len(K_values)
        remaining_K_test = n_test % len(K_values)
        K_indices_test = []
        for i in range(len(K_values)):
            count = n_per_K_test + (1 if i < remaining_K_test else 0)
            K_indices_test.append(torch.full((count,), i))
        K_indices_test = torch.cat(K_indices_test)
        K_indices_test = K_indices_test[torch.randperm(n_test)]
        x_test_block[:, 2] = K_phys[K_indices_test]

        # I values (3 options)
        n_per_I_test = n_test // len(I_values)
        remaining_I_test = n_test % len(I_values)
        I_indices_test = []
        for i in range(len(I_values)):
            count = n_per_I_test + (1 if i < remaining_I_test else 0)
            I_indices_test.append(torch.full((count,), i))
        I_indices_test = torch.cat(I_indices_test)
        I_indices_test = I_indices_test[torch.randperm(n_test)]
        x_test_block[:, 3] = I_phys[I_indices_test]

        # Compute targets
        y_test_clean = buckling_mixed_variables(x_test_block, source=src)
        
        # Compute test std for noise scaling
        if y_test_clean.numel() > 1:
            test_std_value = float(y_test_clean.std().item())
        else:
            test_std_value = 0.0
        test_std_per_source[src_idx] = test_std_value
        
        # Add noise to test data
        y_test_block = y_test_clean.clone()
        if n_test > 0 and test_noise[src_idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[src_idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[src_idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise
        
        # Store categorical indices if requested
        if return_categorical:
            x_test_block[:, 1] = E_indices_test.to(torch.float64)
            x_test_block[:, 2] = K_indices_test.to(torch.float64)
            x_test_block[:, 3] = I_indices_test.to(torch.float64)
        
        source_column = torch.full((x_test_block.shape[0], 1), src_idx, dtype=torch.float64)
        X_test_list.append(torch.cat([x_test_block, source_column], dim=1))
        y_test_list.append(y_test_block)
    
    # Combine test data
    X_test_all = torch.cat(X_test_list, dim=0)
    y_test_all = torch.cat(y_test_list, dim=0)
    
    # Generate train data as folds
    X_train_folds = []
    y_train_folds = []
    
    total_train_samples = sum(train_samples_per_source)
    if total_train_samples == 0:
        return X_train_folds, y_train_folds, X_test_all, y_test_all
    
    # Pre-allocate lists for all folds
    for _ in range(num_folds):
        X_train_folds.append([])
        y_train_folds.append([])
    
    for src_idx, (src, n_train) in enumerate(zip(sources, train_samples_per_source)):
        if n_train == 0:
            continue
        
        # Calculate target samples per fold
        target_per_fold = n_train // num_folds
        remainder = n_train % num_folds
        
        # Calculate EXACT target counts per categorical value per fold using MATH
        num_E = len(E_values)  # 2
        num_K = len(K_values)  # 4
        num_I = len(I_values)  # 3
        
        # Generate each fold separately with EXACT categorical distributions
        for fold in range(num_folds):
            fold_target = target_per_fold + (1 if fold < remainder else 0)
            
            # Calculate EXACT counts for each categorical value in this fold
            # E: fold_target / 2
            E_base = fold_target // num_E
            E_rem = fold_target % num_E
            E_counts = [E_base + (1 if i < E_rem else 0) for i in range(num_E)]
            
            # K: fold_target / 4
            K_base = fold_target // num_K
            K_rem = fold_target % num_K
            K_counts = [K_base + (1 if i < K_rem else 0) for i in range(num_K)]
            
            # I: fold_target / 3 - ROTATE which I value gets extra across folds
            I_base = fold_target // num_I
            I_rem = fold_target % num_I
            # Rotate which I value gets the remainder based on fold number
            # This ensures different folds have different I distributions
            # For example, if I_rem=2: fold 0 gets (7,7,6), fold 1 gets (7,6,7), fold 2 gets (6,7,7)
            rotation_offset = (fold + src_idx * num_folds) % num_I
            I_counts = [I_base + (1 if (i + rotation_offset) % num_I < I_rem else 0) for i in range(num_I)]
            
            # Generate samples for this fold with EXACT categorical distributions
            # Build list of categorical assignments that satisfy exact counts
            cat_assignments = []
            
            # Create assignments for E values (exact counts)
            for e_idx in range(num_E):
                for _ in range(E_counts[e_idx]):
                    cat_assignments.append({'e': e_idx})
            
            # Create assignments for K values (exact counts) - distribute round-robin
            k_list = []
            for k_idx in range(num_K):
                for _ in range(K_counts[k_idx]):
                    k_list.append(k_idx)
            # Shuffle K assignments
            if seed is not None:
                torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 1)
            k_perm = torch.randperm(len(k_list))
            k_list = [k_list[i] for i in k_perm.tolist()]
            
            # Create assignments for I values (exact counts) - distribute round-robin
            i_list = []
            for i_idx in range(num_I):
                for _ in range(I_counts[i_idx]):
                    i_list.append(i_idx)
            # Shuffle I assignments
            if seed is not None:
                torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 2)
            i_perm = torch.randperm(len(i_list))
            i_list = [i_list[i] for i in i_perm.tolist()]
            
            # Combine assignments
            for i in range(fold_target):
                cat_assignments[i]['k'] = k_list[i]
                cat_assignments[i]['i'] = i_list[i]
            
            # Shuffle final order to randomize (E, K, I) combinations
            if seed is not None:
                torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 3)
            perm = torch.randperm(len(cat_assignments))
            cat_assignments = [cat_assignments[i] for i in perm.tolist()]
            
            # Generate Sobol samples for continuous L variable
            sobol_seed = (seed + src_idx * 2000 + fold * 100) if seed is not None else None
            sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=sobol_seed)
            L_vals = sobol.draw(fold_target).squeeze() * 0.4 + 0.6
            
            # Create samples - ALWAYS use physical values for buckling computation
            samples_X = []
            for i, assignment in enumerate(cat_assignments):
                x_sample = torch.zeros((1, 4), dtype=torch.float64)
                x_sample[0, 0] = L_vals[i] if L_vals.dim() > 0 else L_vals
                # Always use physical values for computation
                x_sample[0, 1] = E_phys[assignment['e']]
                x_sample[0, 2] = K_phys[assignment['k']]
                x_sample[0, 3] = I_phys[assignment['i']]
                
                samples_X.append(x_sample)
            
            # Convert to tensors
            if len(samples_X) > 0:
                x_fold = torch.cat(samples_X, dim=0)
                # Compute targets using physical values
                y_fold = buckling_mixed_variables(x_fold, source=src)
                
                # Convert to categorical indices if requested (AFTER computing y)
                if return_categorical:
                    for i, assignment in enumerate(cat_assignments):
                        x_fold[i, 1] = float(assignment['e'])
                        x_fold[i, 2] = float(assignment['k'])
                        x_fold[i, 3] = float(assignment['i'])
                
                # Add noise if needed
                test_std_value = test_std_per_source.get(src_idx, 0.0)
                if train_noise[src_idx] > 0 and test_std_value > 0.0:
                    if noise_type == 'gaussian':
                        noise = torch.randn_like(y_fold) * (train_noise[src_idx] * test_std_value)
                    elif noise_type == 'uniform':
                        noise = (torch.rand_like(y_fold) - 0.5) * 2 * (train_noise[src_idx] * test_std_value)
                    else:
                        raise ValueError(f"Unknown noise_type: {noise_type}")
                    y_fold = y_fold + noise
                
                # Add source column
                source_column = torch.full((x_fold.shape[0], 1), src_idx, dtype=torch.float64)
                x_fold_with_source = torch.cat([x_fold, source_column], dim=1)
                
                X_train_folds[fold].append(x_fold_with_source)
                y_train_folds[fold].append(y_fold)
    
    # Concatenate folds from all sources
    for fold in range(num_folds):
        if len(X_train_folds[fold]) > 0:
            X_train_folds[fold] = torch.cat(X_train_folds[fold], dim=0)
            y_train_folds[fold] = torch.cat(y_train_folds[fold], dim=0)
        else:
            # If fold is empty, create empty tensor with correct shape
            X_train_folds[fold] = torch.empty((0, 5), dtype=torch.float64)
            y_train_folds[fold] = torch.empty((0,), dtype=torch.float64)
    
    return X_train_folds, y_train_folds, X_test_all, y_test_all


def generate_mf_buckling_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                              seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                              noise_type: str = 'gaussian', return_categorical: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data following the data_gen.py method:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Draw a single Sobol batch per source (no repeats globally)
      - Split into test then train per source
      - Compute per-source test std after the split
      - Scale both train and test additive noise by that test std

    Returns:
      - X_train, y_train: Training data with 5 features (4 continuous + 1 source class column in {0,1})
        Where columns are [L (cont), E, K, I (categorical or values per return_categorical), source]
      - X_test, y_test: Test data with 5 features (same schema as X_train)
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (2 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 2
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 2
    if len(train_noise) != 2 or len(test_noise) != 2:
        raise ValueError("train_noise and test_noise must be length-2 (one scalar per source)")

    sources = ["s0", "s1"]

    # Bounds for the 4 continuous features (L, E, K, I)
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=torch.float64)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=torch.float64)

    # Define specific categorical values (same as data_gen.py)
    E_values = torch.tensor([73.1, 200.0], dtype=torch.float64)  # Column 1: E can only be 73.1 or 200
    K_values = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)  # Column 2: K can only be 0.5, 0.7, 1, or 2
    I_values = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)  # Column 3: I can only be 9.49, 12.1, or 29.5

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets once per source in contiguous blocks
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    X_src_col_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        
        # Generate TEST data with even distribution
        if n_test > 0:
            x_test_block = X_raw_all[cursor:cursor + n_test].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TEST
            n_per_E_test = n_test // len(E_values)
            remaining_E_test = n_test % len(E_values)
            E_indices_test = []
            for i in range(len(E_values)):
                count = n_per_E_test + (1 if i < remaining_E_test else 0)
                E_indices_test.append(torch.full((count,), i))
            E_indices_test = torch.cat(E_indices_test)
            E_indices_test = E_indices_test[torch.randperm(n_test)]
            x_test_block[:, 1] = E_values[E_indices_test]

            # Column 2: K values (4 options) - ensure even distribution for TEST
            n_per_K_test = n_test // len(K_values)
            remaining_K_test = n_test % len(K_values)
            K_indices_test = []
            for i in range(len(K_values)):
                count = n_per_K_test + (1 if i < remaining_K_test else 0)
                K_indices_test.append(torch.full((count,), i))
            K_indices_test = torch.cat(K_indices_test)
            K_indices_test = K_indices_test[torch.randperm(n_test)]
            x_test_block[:, 2] = K_values[K_indices_test]

            # Column 3: I values (3 options) - ensure even distribution for TEST
            n_per_I_test = n_test // len(I_values)
            remaining_I_test = n_test % len(I_values)
            I_indices_test = []
            for i in range(len(I_values)):
                count = n_per_I_test + (1 if i < remaining_I_test else 0)
                I_indices_test.append(torch.full((count,), i))
            I_indices_test = torch.cat(I_indices_test)
            I_indices_test = I_indices_test[torch.randperm(n_test)]
            x_test_block[:, 3] = I_values[I_indices_test]

            # Compute targets for test data
            y_test_clean = buckling_mixed_variables(x_test_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_test_block[:, 1] = E_indices_test.to(torch.float64)
                x_test_block[:, 2] = K_indices_test.to(torch.float64)
                x_test_block[:, 3] = I_indices_test.to(torch.float64)
            
            # Store test data
            y_clean_all[cursor:cursor + n_test] = y_test_clean
            X_raw_all[cursor:cursor + n_test] = x_test_block
            X_src_col_all[cursor:cursor + n_test, 0] = float(idx)

        # Generate TRAIN data with even distribution
        if n_train > 0:
            x_train_block = X_raw_all[cursor + n_test:cursor + n_total].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TRAIN
            n_per_E_train = n_train // len(E_values)
            remaining_E_train = n_train % len(E_values)
            E_indices_train = []
            for i in range(len(E_values)):
                count = n_per_E_train + (1 if i < remaining_E_train else 0)
                E_indices_train.append(torch.full((count,), i))
            E_indices_train = torch.cat(E_indices_train)
            E_indices_train = E_indices_train[torch.randperm(n_train)]
            x_train_block[:, 1] = E_values[E_indices_train]

            # Column 2: K values (4 options) - ensure even distribution for TRAIN
            n_per_K_train = n_train // len(K_values)
            remaining_K_train = n_train % len(K_values)
            K_indices_train = []
            for i in range(len(K_values)):
                count = n_per_K_train + (1 if i < remaining_K_train else 0)
                K_indices_train.append(torch.full((count,), i))
            K_indices_train = torch.cat(K_indices_train)
            K_indices_train = K_indices_train[torch.randperm(n_train)]
            x_train_block[:, 2] = K_values[K_indices_train]

            # Column 3: I values (3 options) - ensure even distribution for TRAIN
            n_per_I_train = n_train // len(I_values)
            remaining_I_train = n_train % len(I_values)
            I_indices_train = []
            for i in range(len(I_values)):
                count = n_per_I_train + (1 if i < remaining_I_train else 0)
                I_indices_train.append(torch.full((count,), i))
            I_indices_train = torch.cat(I_indices_train)
            I_indices_train = I_indices_train[torch.randperm(n_train)]
            x_train_block[:, 3] = I_values[I_indices_train]

            # Compute targets for train data
            y_train_clean = buckling_mixed_variables(x_train_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_train_block[:, 1] = E_indices_train.to(torch.float64)
                x_train_block[:, 2] = K_indices_train.to(torch.float64)
                x_train_block[:, 3] = I_indices_train.to(torch.float64)
            
            # Store train data
            y_clean_all[cursor + n_test:cursor + n_total] = y_train_clean
            X_raw_all[cursor + n_test:cursor + n_total] = x_train_block
            X_src_col_all[cursor + n_test:cursor + n_total, 0] = float(idx)

        cursor += n_total

    # Split per source into test then train; get test std after split and add noise scaled by it
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        src_block = X_src_col_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        src_test_block = src_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        src_train_block = src_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Test std after split (per source)
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append source column and collect
        X_test_list.append(torch.cat([x_test_block, src_test_block], dim=1))
        y_test_list.append(y_test_block)
        X_train_list.append(torch.cat([x_train_block, src_train_block], dim=1))
        y_train_list.append(y_train_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 5), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 5), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def generate_mf_buckling_data_v2(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                              seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                              noise_type: str = 'gaussian', return_categorical: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data in the same way as wing MF:
      - Draw a single Sobol batch per source (no repeats globally)
      - Split into test then train per source
      - Compute per-source test std after the split
      - Scale both train and test additive noise by that test std

    Returns:
      - X_train, y_train: Training data with 5 features (4 continuous + 1 source class column in {0,1})
        Where columns are [L (cont), E, K, I (categorical or values per return_categorical), source]
      - X_test, y_test: Test data with 5 features (same schema as X_train)
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (2 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 2
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 2
    if len(train_noise) != 2 or len(test_noise) != 2:
        raise ValueError("train_noise and test_noise must be length-2 (one scalar per source)")

    sources = ["s0", "s1"]

    # Bounds for the 4 continuous features (L, E, K, I)
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=torch.float64)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=torch.float64)

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Discrete value sets for categorical variables: E (col 1), K (col 2), I (col 3)
    E_values = torch.tensor([73.1, 200.0], dtype=torch.float64)
    K_values = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)
    I_values = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)

    # Compute clean targets once per source in contiguous blocks
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    X_src_col_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n]

        # Find nearest categorical indices for E, K, I
        diffs = (x_block[:, 1:2] - E_values.unsqueeze(0))**2
        nearest_idx_E = diffs.argmin(dim=1)
        diffs = (x_block[:, 2:3] - K_values.unsqueeze(0))**2
        nearest_idx_K = diffs.argmin(dim=1)
        diffs = (x_block[:, 3:4] - I_values.unsqueeze(0))**2
        nearest_idx_I = diffs.argmin(dim=1)

        # Build a values-based view for computing y
        x_vals = x_block.clone()
        x_vals[:, 1] = E_values[nearest_idx_E]
        x_vals[:, 2] = K_values[nearest_idx_K]
        x_vals[:, 3] = I_values[nearest_idx_I]

        # Compute targets using physical values, not indices
        y_clean_all[cursor:cursor + n] = buckling_mixed_variables(x_vals, source=src)

        # Store either indices or values in the master block according to flag
        if return_categorical:
            x_block[:, 1] = nearest_idx_E.to(torch.float64)
            x_block[:, 2] = nearest_idx_K.to(torch.float64)
            x_block[:, 3] = nearest_idx_I.to(torch.float64)
        else:
            x_block[:, 1] = x_vals[:, 1]
            x_block[:, 2] = x_vals[:, 2]
            x_block[:, 3] = x_vals[:, 3]
        X_src_col_all[cursor:cursor + n, 0] = float(idx)
        cursor += n

    # Split per source into test then train; get test std after split and add noise scaled by it
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        src_block = X_src_col_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        src_test_block = src_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        src_train_block = src_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Test std after split (per source)
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append source column and collect
        X_test_list.append(torch.cat([x_test_block, src_test_block], dim=1))
        y_test_list.append(y_test_block)
        X_train_list.append(torch.cat([x_train_block, src_train_block], dim=1))
        y_train_list.append(y_train_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 5), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 5), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def borehole_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute borehole water flow rate given input variables (torch implementation).

    Args:
        X (torch.Tensor): Input array of shape [n_samples, 8] with columns:
            0: rw (radius of borehole, m)
            1: r (radius of influence, m)
            2: Tu (transmissivity of upper aquifer, m^2/yr)
            3: Hu (potentiometric head of upper aquifer, m)
            4: Tl (transmissivity of lower aquifer, m^2/yr)
            5: Hl (potentiometric head of lower aquifer, m)
            6: L (length of borehole, m)
            7: Kw (hydraulic conductivity of borehole, m/yr)
        source (str): Source/fidelity level ("s0".."s4")
    Returns:
        torch.Tensor: Flow rate values for each input sample
    """
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3]
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]

    if source == "s0":
        numerator = 2 * torch.pi * Tu * (Hu - Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s1":
        numerator = 2 * torch.pi * Tu * (Hu - 0.8 * Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s2":
        numerator = 2 * torch.pi * Tu * (Hu - 3 * Hl)
        denominator = torch.log(r / rw) * (1 + 8 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + 0.75 * Tu / Tl)
        result = numerator / denominator
    elif source == "s3":
        numerator = 2 * torch.pi * Tu * (1.1 * Hu - Hl)
        denominator = torch.log(4 * r / rw) * (1 + 3 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s4":
        numerator = 2 * torch.pi * Tu * (1.05 * Hu - Hl)
        denominator = torch.log(2 * r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    else:
        raise ValueError(f"Unknown source: {source}. Only s0..s4 are supported for borehole.")

    return result


def generate_mf_borehole_data(
    train_samples_per_source: list[int],
    test_samples_per_source: list[int],
    *,
    seed: int | None = None,
    train_noise: list[float] | float | None = None,
    test_noise: list[float] | float | None = None,
    noise_type: str = 'gaussian',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Borehole data consistent with wing/buckling helpers.

    - Draw a single Sobol batch across all sources (no repeats globally)
    - Split per source: first test, remaining train
    - Scale additive noise by per-source test std (post-split), like wing/buckling

    Returns:
      - X_train, y_train with 9 features (8 continuous + 1 numeric source column)
      - X_test, y_test with same schema
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (5 sources)
    num_sources = 5
    if train_noise is None:
        train_noise = [0.0] * num_sources
    if test_noise is None:
        test_noise = [0.0] * num_sources
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * num_sources
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * num_sources
    if len(train_noise) != num_sources or len(test_noise) != num_sources:
        raise ValueError(f"train_noise and test_noise must be length-{num_sources} (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3", "s4"]

    # Bounds for the 8 continuous features
    l_bound = torch.tensor([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0], dtype=torch.float64)
    u_bound = torch.tensor([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0], dtype=torch.float64)

    # Totals and Sobol draws
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    sobol = torch.quasirandom.SobolEngine(dimension=8, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets per contiguous source block
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    src_ids_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n]
        y_clean_all[cursor:cursor + n] = borehole_mixed_variables(x_block, source=src)
        src_ids_all[cursor:cursor + n, 0] = float(idx)
        cursor += n

    # Split into test then train per source; add scaled noise
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        s_block = src_ids_all[cursor:cursor + n_total]

        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        s_test_block = s_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        s_train_block = s_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Per-source test std
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append numeric source id as 9th feature
        if n_train > 0:
            X_train_list.append(torch.cat([x_train_block, s_train_block], dim=1))
            y_train_list.append(y_train_block)
        if n_test > 0:
            X_test_list.append(torch.cat([x_test_block, s_test_block], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 9), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 9), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def ackley_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Ackley function for given input variables.
    
    The Ackley function is defined as:
    f(x) = -20 * exp(-0.2 * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    
    where x ∈ [-32.768, 32.768]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Ackley function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Constants
    a = 20.0
    b = 0.2
    c = 2 * torch.pi
    
    # Compute the two main terms
    sum_squares = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(c * X), dim=1)
    
    # Ackley function
    term1 = -a * torch.exp(-b * torch.sqrt(sum_squares / dimensions))
    term2 = -torch.exp(sum_cos / dimensions)
    result = term1 + term2 + a + torch.e
    
    return result


def generate_ackley_data(n_train: int, n_test: int, dimensions: int = 2, train_noise: float = 0.0, 
                        test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None, V2: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Ackley function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Ackley function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = -5
    u_bound = 10
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Ackley bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Ackley function values
    y_all = ackley_function(X_all, dimensions)

    if V2:
        y_all = torch.log(y_all+1)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test
