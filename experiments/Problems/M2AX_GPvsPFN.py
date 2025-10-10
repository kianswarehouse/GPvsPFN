import torch
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tabpfn_wrapper import VanillaDirectTabPFNRegressor
import torch.nn.functional as F


def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
    """
    data: (N, D) torch tensor (float or int)
    Returns:
      encoded_data: torch.Tensor with integer columns one-hot encoded
      column_info: list of dicts describing how each original column was treated
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float64)

    num_rows, num_cols = data.shape
    encoded_columns = []
    column_info = []

    for col_idx in range(num_cols):
        col = data[:, col_idx]

        is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
        if is_integer_like:
            col_long = torch.round(col).to(torch.long)
            min_val = int(col_long.min().item())
            max_val = int(col_long.max().item())
            num_classes = max_val - min_val + 1

            # Shift so the smallest value maps to 0, then one-hot
            class_indices = col_long - min_val
            ohe = F.one_hot(class_indices, num_classes=num_classes).to(data.dtype)
            encoded_columns.append(ohe)

            column_info.append({
                "column": col_idx,
                "type": "one_hot",
                "min_value": min_val,
                "max_value": max_val,
                "num_classes": num_classes
            })
        else:
            encoded_columns.append(col.unsqueeze(1))
            column_info.append({
                "column": col_idx,
                "type": "continuous"
            })

    encoded_data = torch.cat(encoded_columns, dim=1)
    return encoded_data, column_info


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
    
    # Get the first three columns (element names) for label encoding
    categorical_columns = df.columns[:3]  # Msiteelement, Asiteelement, Xsiteelement
    
    # Create label encoders for the first three columns
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
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
    
    return X, y, label_encoders

# Load the data
X, y, label_encoders = load_m2ax_data()


# Choose target variable (y_porosity or y_hardness)
seed = 42

# Set up device and dtype for autocast
device = "cuda" if torch.cuda.is_available() else "cpu"
amp_device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float64
dtype = torch.float64

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
single_eval_pos = len(X_train)


# 4. Stack context and candidate/query points
X_all_np = np.concatenate([X_train, X_test], axis=0)
Y_all_np = np.concatenate([y_train, np.zeros_like(y_test)], axis=0)

# 5. GITBO expects [N, 1, D] and [N, 1, 1]
X_all = torch.tensor(X_all_np, dtype=amp_dtype).unsqueeze(1)      # [N, 1, D]
Y_all = torch.tensor(Y_all_np, dtype=amp_dtype).reshape(-1, 1, 1) # [N, 1, 1]

regressor = VanillaDirectTabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")

with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
    out             = regressor.forward(X_all, Y_all, single_eval_pos)
    logits          = out["standard"]
    y_pred     = regressor.predict_mean(logits)                   # [ (single_eval_pos+N_PENDING), N_CANDIDATES ]
    output_variance = regressor.predict_variance(logits)               # same shape
y_pred = y_pred.detach().cpu().numpy().reshape(-1)
output_variance = output_variance.detach().cpu().numpy().reshape(-1)
output_std = output_variance **0.5



# =============================================================================
# GP M2AX Section - Copy data tensors and apply one-hot encoding
# =============================================================================

print("\n" + "="*50)
print("GP M2AX Section")
print("="*50)

# Copy the original data tensors for GP processing
X_gp = X.copy()
y_gp = y.copy()

# Convert to torch tensors
X_gp = torch.tensor(X_gp, dtype=dtype)
y_gp = torch.tensor(y_gp, dtype=dtype)

print(f"Original GP data shape: {X_gp.shape}")

# Apply one-hot encoding to integer columns
X_gp_encoded, column_info = one_hot_encode_integer_columns(X_gp)

print(f"One-hot encoded GP data shape: {X_gp_encoded.shape}")
print("Column encoding info:")
for i, info in enumerate(column_info):
    if info["type"] == "one_hot":
        print(f"  Column {i}: {info['num_classes']} classes (values {info['min_value']}-{info['max_value']})")
    else:
        print(f"  Column {i}: continuous")

# Split the GP data
X_gp_train, X_gp_test, y_gp_train, y_gp_test = train_test_split(
    X_gp_encoded.numpy(), y_gp.numpy(), test_size=0.2, random_state=seed
)

print(f"GP train shape: {X_gp_train.shape}")
print(f"GP test shape: {X_gp_test.shape}")

# Convert back to torch tensors for GP processing
X_gp_train = torch.tensor(X_gp_train, dtype=dtype)
X_gp_test = torch.tensor(X_gp_test, dtype=dtype)
y_gp_train = torch.tensor(y_gp_train, dtype=dtype)
y_gp_test = torch.tensor(y_gp_test, dtype=dtype)

print("GP M2AX data preparation complete!")
print("="*50)



