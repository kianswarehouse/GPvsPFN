"""Actually generate SF and MF data and compare the 5000 test samples"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from load_experimental_data import generate_mf_wing_data
from gpplus.utils import set_seed

# Use the same parameters as the actual functions
num_seeds = 20
num_test = 5000
train_size = 10

# SF parameters
set_seed(0)
train_per_seed_sf = train_size * 10
total_train_sf = num_seeds * train_per_seed_sf

print("=" * 80)
print("GENERATING SF DATA")
print("=" * 80)
print(f"train_samples_per_source: [{total_train_sf}, 0, 0, 0]")
print(f"test_samples_per_source: [{num_test}, 0, 0, 0]")

X_train_sf, y_train_sf, X_test_sf, y_test_sf = generate_mf_wing_data(
    train_samples_per_source=[total_train_sf, 0, 0, 0], 
    test_samples_per_source=[num_test, 0, 0, 0], 
    train_noise=0.0, 
    test_noise=0.0, 
    noise_type='gaussian'
)

# Drop source column for SF
X_train_sf = X_train_sf[:, :10]
X_test_sf = X_test_sf[:, :10]

print(f"SF X_train shape: {X_train_sf.shape}")
print(f"SF y_train shape: {y_train_sf.shape}")
print(f"SF X_test shape: {X_test_sf.shape}")
print(f"SF y_test shape: {y_test_sf.shape}")

# MF parameters
set_seed(0)  # Reset seed
train_size_mf = [10, 10, 10, 10]
num_test_mf = [5000, 1000, 1000, 1000]

train_per_seed_mf = torch.tensor(train_size_mf) * 10
total_train_mf = num_seeds * train_per_seed_mf

print("\n" + "=" * 80)
print("GENERATING MF DATA")
print("=" * 80)
print(f"train_samples_per_source: {total_train_mf.tolist()}")
print(f"test_samples_per_source: {num_test_mf}")

X_train_mf, y_train_mf, X_test_mf, y_test_mf = generate_mf_wing_data(
    train_samples_per_source=total_train_mf.tolist(), 
    test_samples_per_source=num_test_mf, 
    train_noise=[0.0, 0.0, 0.0, 0.0], 
    test_noise=[0.0, 0.0, 0.0, 0.0], 
    noise_type='gaussian'
)

print(f"MF X_train shape: {X_train_mf.shape}")
print(f"MF y_train shape: {y_train_mf.shape}")
print(f"MF X_test shape: {X_test_mf.shape}")
print(f"MF y_test shape: {y_test_mf.shape}")

# Extract HF (source 0) data from MF
# Source column is the last column (index -1)
source_col_train = X_train_mf[:, -1]
source_col_test = X_test_mf[:, -1]
hf_mask_train = source_col_train == 0.0
hf_mask_test = source_col_test == 0.0
X_train_mf_hf = X_train_mf[hf_mask_train][:, :10]  # Drop source column
y_train_mf_hf = y_train_mf[hf_mask_train]
X_test_mf_hf = X_test_mf[hf_mask_test][:, :10]  # Drop source column
y_test_mf_hf = y_test_mf[hf_mask_test]

print(f"\nMF HF (source 0) data:")
print(f"  X_train_mf_hf shape: {X_train_mf_hf.shape}")
print(f"  y_train_mf_hf shape: {y_train_mf_hf.shape}")
print(f"  X_test_mf_hf shape: {X_test_mf_hf.shape}")
print(f"  y_test_mf_hf shape: {y_test_mf_hf.shape}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nX_test shapes match: {X_test_sf.shape == X_test_mf_hf.shape}")
print(f"y_test shapes match: {y_test_sf.shape == y_test_mf_hf.shape}")

if X_test_sf.shape == X_test_mf_hf.shape:
    x_match = torch.allclose(X_test_sf, X_test_mf_hf, rtol=1e-9, atol=1e-9)
    x_max_diff = (X_test_sf - X_test_mf_hf).abs().max().item()
    print(f"\nX_test (features) are identical: {x_match}")
    print(f"Max difference in X_test: {x_max_diff}")
    
    if not x_match:
        # Find where they differ
        diff_mask = ~torch.isclose(X_test_sf, X_test_mf_hf, rtol=1e-9, atol=1e-9)
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print(f"Number of differing elements: {len(diff_indices)}")
        if len(diff_indices) > 0:
            print(f"First few differing positions: {diff_indices[:5]}")
            idx = diff_indices[0]
            print(f"  SF value: {X_test_sf[idx[0], idx[1]]}")
            print(f"  MF value: {X_test_mf_hf[idx[0], idx[1]]}")
            print(f"  Difference: {X_test_sf[idx[0], idx[1]] - X_test_mf_hf[idx[0], idx[1]]}")

if y_test_sf.shape == y_test_mf_hf.shape:
    y_match = torch.allclose(y_test_sf, y_test_mf_hf, rtol=1e-9, atol=1e-9)
    y_max_diff = (y_test_sf - y_test_mf_hf).abs().max().item()
    print(f"\ny_test (targets) are identical: {y_match}")
    print(f"Max difference in y_test: {y_max_diff}")
    
    if not y_match:
        # Find where they differ
        diff_mask = ~torch.isclose(y_test_sf, y_test_mf_hf, rtol=1e-9, atol=1e-9)
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print(f"Number of differing elements: {len(diff_indices)}")
        if len(diff_indices) > 0:
            print(f"First few differing positions: {diff_indices[:5]}")
            idx = diff_indices[0]
            print(f"  SF value: {y_test_sf[idx[0]]}")
            print(f"  MF value: {y_test_mf_hf[idx[0]]}")
            print(f"  Difference: {y_test_sf[idx[0]] - y_test_mf_hf[idx[0]]}")

# Print first 5 and last 5 values
print("\n" + "=" * 80)
print("FIRST 5 AND LAST 5 VALUES - PROOF")
print("=" * 80)

print("\n--- TEST DATA ---")
print("\nSF X_test - First 5:")
print(X_test_sf[:5])
print("SF X_test - Last 5:")
print(X_test_sf[-5:])
print("\nMF HF X_test - First 5:")
print(X_test_mf_hf[:5])
print("MF HF X_test - Last 5:")
print(X_test_mf_hf[-5:])

print("\nSF y_test - First 5:")
print(y_test_sf[:5])
print("SF y_test - Last 5:")
print(y_test_sf[-5:])
print("\nMF HF y_test - First 5:")
print(y_test_mf_hf[:5])
print("MF HF y_test - Last 5:")
print(y_test_mf_hf[-5:])

print("\n--- TRAINING DATA ---")
print("\nSF X_train - First 5:")
print(X_train_sf[:5])
print("SF X_train - Last 5:")
print(X_train_sf[-5:])
print("\nMF HF X_train - First 5:")
print(X_train_mf_hf[:5])
print("MF HF X_train - Last 5:")
print(X_train_mf_hf[-5:])

print("\nSF y_train - First 5:")
print(y_train_sf[:5])
print("SF y_train - Last 5:")
print(y_train_sf[-5:])
print("\nMF HF y_train - First 5:")
print(y_train_mf_hf[:5])
print("MF HF y_train - Last 5:")
print(y_train_mf_hf[-5:])

# Compare training data
print("\n" + "=" * 80)
print("TRAINING DATA COMPARISON")
print("=" * 80)

print(f"\nX_train shapes match: {X_train_sf.shape == X_train_mf_hf.shape}")
print(f"y_train shapes match: {y_train_sf.shape == y_train_mf_hf.shape}")

if X_train_sf.shape == X_train_mf_hf.shape:
    x_train_match = torch.allclose(X_train_sf, X_train_mf_hf, rtol=1e-9, atol=1e-9)
    x_train_max_diff = (X_train_sf - X_train_mf_hf).abs().max().item()
    print(f"\nX_train (features) are identical: {x_train_match}")
    print(f"Max difference in X_train: {x_train_max_diff}")

if y_train_sf.shape == y_train_mf_hf.shape:
    y_train_match = torch.allclose(y_train_sf, y_train_mf_hf, rtol=1e-9, atol=1e-9)
    y_train_max_diff = (y_train_sf - y_train_mf_hf).abs().max().item()
    print(f"\ny_train (targets) are identical: {y_train_match}")
    print(f"Max difference in y_train: {y_train_max_diff}")

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)
test_match = X_test_sf.shape == X_test_mf_hf.shape and y_test_sf.shape == y_test_mf_hf.shape and x_match and y_match
train_match = X_train_sf.shape == X_train_mf_hf.shape and y_train_sf.shape == y_train_mf_hf.shape and x_train_match and y_train_match

if test_match:
    print("✓ YES: The 5000 test data (features and targets) in SF and MF HF are IDENTICAL")
else:
    print("✗ NO: The 5000 test data (features and targets) in SF and MF HF are DIFFERENT")
    print(f"   Features match: {x_match}, Targets match: {y_match}")

if train_match:
    print("✓ YES: The training data (features and targets) in SF and MF HF are IDENTICAL")
else:
    print("✗ NO: The training data (features and targets) in SF and MF HF are DIFFERENT")
    print(f"   Features match: {x_train_match}, Targets match: {y_train_match}")

