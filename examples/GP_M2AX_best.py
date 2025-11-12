import os

os.environ["PYTHONHASHSEED"] = "0"
import argparse
import random
import sys

# from _GITBO import *
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# from gpplus.utils.model_inspector import print_model_parameters
# from gpplus.training.callbacks import PrintInitializedParametersCallback

warnings.filterwarnings("ignore")

import time
from datetime import datetime

import gpytorch

import gpplus
from gpplus.models import GPR
# from gpplus.TestProblems_Utils import *
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.encoders import MatrixEncoder, NeuralEncoder
# from gpplus.utils.one_hot_to_latent_nn import OneHotToLatent

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def compute_metrics(y_true, y_hat, output_std=None):
    # Basic metrics
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    metrics = {
        "Time": time.time() - t1,
        "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
        "MSE": mean_squared_error(y_true, y_hat),
        # "MAE": mean_absolute_error(y_true, y_hat),
        # "R2": r2_score(y_true, y_hat)
    }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96  # or 1.96 for 95% CI
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics


import torch.nn.functional as F


def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
    """
    data: (N, D) torch tensor (float or int)
    Returns:
      encoded_data: torch.Tensor with integer columns one-hot encoded
      column_info: list of dicts describing how each original column was treated
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float32)

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

            column_info.append(
                {
                    "column": col_idx,
                    "type": "one_hot",
                    "min_value": min_val,
                    "max_value": max_val,
                    "num_classes": num_classes,
                }
            )
        else:
            encoded_columns.append(col.unsqueeze(1))
            column_info.append({"column": col_idx, "type": "continuous"})

    encoded_data = torch.cat(encoded_columns, dim=1)
    return encoded_data, column_info


import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

print("M2AX Dataset, GP")

tkwargs = {"device": torch.device(device), "dtype": torch.float64}
amp_dtype = tkwargs["dtype"]
amp_device = device
model_dtype = torch.float64
data1 = pd.read_csv("C:/Users/kianb/Repos/Datasets/data_M.csv")

cat_cols = ["Msiteelement", "Asiteelement", "Xsiteelement"]
data = data1
# Factorize each categorical column separately
for col in cat_cols:
    data[col], _ = pd.factorize(data[col])

# print(data.head())

data = torch.tensor(data.to_numpy(), dtype=amp_dtype)
# data_no_header = torch.tensor(data.values, dtype=amp_dtype)  # removes headers automatically

# Now call your one-hot encoder
encoded_cols, classes = one_hot_encode_integer_columns(data[:, :3])

encoded_data = torch.cat([encoded_cols, data[:, 3:]], dim=1)
# print(encoded_data.shape)
# print(encoded_data)
# data_all = torch.cat((data1_t, data2_t, data3_t), dim=0)  # dim=0 → stack rows
t00 = time.time()


for c in classes:
    print(c)
print("Final shape:", tuple(encoded_cols.shape))

# Convert to tensor
arr = torch.tensor(encoded_data, dtype=amp_dtype)


X = arr[:, : len(encoded_cols[0])]
xnp = np.array(X)
# y = arr[:, -1]
# print("E, Young's Modulus")
# y = arr[:, -2]
# print("G, Shear Modulus")
y = arr[:, -3]
print("B, Bulk Modulus")

ynp = np.array(y)
num_epochs = 10000
num_runs = 4
lr = 0.1
y_train_noise = 0

convergence_patience = 20

i = 0
full_metrics = []
t0 = time.time()
for seed in range(42, 52):
    t1 = time.time()
    i += 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # %%

    # cat_encoder1 = OneHotToLatent(
    #     input_dim=classes[0]['num_classes'], # 10
    #     architecture_config={
    #         'hidden_dims': [4],
    #         'activation': 'hardtanh',
    #         'dropout': 0.0
    #     },
    #     z_dim=2
    # )
    cat_encoder1 = MatrixEncoder(input_dim=classes[0]["num_classes"], z_dim=2, initialization="normal", init_std=0.01)

    # cat_encoder2 = OneHotToLatent(
    #     input_dim=classes[1]['num_classes'], # 12
    #     architecture_config={
    #         'hidden_dims': [4],
    #         'activation': 'hardtanh',
    #         'dropout': 0.0
    #     },
    #     z_dim=2
    # )
    cat_encoder2 = MatrixEncoder(input_dim=classes[1]["num_classes"], z_dim=2, initialization="normal", init_std=0.01)

    # cat_encoder3 = OneHotToLatent(
    #     input_dim=classes[2]['num_classes'], # 2
    #     architecture_config={
    #         'hidden_dims': [2],
    #         'activation': 'hardtanh',
    #         'dropout': 0.0
    #     },
    #     z_dim=2
    # )
    cat_encoder3 = MatrixEncoder(input_dim=classes[2]["num_classes"], z_dim=2, initialization="normal", init_std=0.01)

    # kernel = gpplus.kernels.LogScaleKernel(gpplus.kernels.CombinedKernel(
    kernel = gpplus.kernels.CombinedKernel(
        cat_cols=[np.arange(0, 10), np.arange(10, 22), np.arange(22, 24)],
        # cat_encoder=[cat_encoder1, cat_encoder2, cat_encoder3],
        # cat_encoder = 'nn',
        # cat_encoder = 'matrix'
        # cat_combination_method="product",
        # cat_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
    )
    # )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train = X_train.to(amp_dtype)
    X_test = X_test.to(amp_dtype)
    y_train = y_train.to(amp_dtype)
    y_test = y_test.to(amp_dtype)

    y_train = y_train + (y_train_noise * torch.randn_like(y_train))

    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train_normal = (y_train - y_train_mean) / y_train_std

    # y_train_min = y_train.min()
    # y_train_std2 = y_train.max() - y_train.min()
    # y_train_normal = (y_train - y_train_min) / y_train_std2

    # y_train_std_prior = y_train.std().item()
    # prior_mean = np.log(y_train_std_prior**2)
    # noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
    # from gpytorch.constraints import GreaterThan

    model = GPR(
        X_train,
        y_train_normal,
        # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
        # kernel_module = gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
        # kernel_module = gpytorch.kernels.RBFKernel(),
        kernel_module=kernel,
        # mean_module=gpytorch.means.ZeroMean(),
        likelihood=gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=GreaterThan(1e-6)), #, noise_prior=noise_prior),
        # seed=seed,
        # dtype=model_dtype
    )

    trainer = gpplus.training.GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs,  # <-----
        optimizer_kwargs = {
            'max_iter': 2000,
            'max_eval': 5000,
            'tolerance_grad': 1e-5,
            'tolerance_change': 1e-9,
            'history_size': 10,
        },
        # convergence_patience=convergence_patience,
        optimizer_class=gpplus.training.optimizers.LBFGSScipy,
        # optimizer_class=torch.optim.LBFGS,
        device="cpu",
        # map_prior=True,
        # scheduler = .95
        # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
        # initializer_class=initializer_class,  # Use the chosen initializer
    )

    if i == 1:
        print(model)
        # print(f"Sample input: {X_train[0]}")
        print(f"Input data dtype = {model.train_inputs[0].dtype}")
        print(f"Target data dtype = {model.train_targets[0].dtype}, added noise: {y_train_noise}")
        # print(f"model dtype = {model.dtype}")
        print(f"Optimizer = {trainer.optimizer_class}")
        print(f"Optimizer_kwargs = {trainer.optimizer_kwargs}")
        print(f"convergence_patience: {trainer.convergence_patience}")
        print(f"{num_runs} runs. Lr = {lr}. {num_epochs} epochs")
    # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
    results = trainer.train()  # Returns a dict of results; you might store if needed.
    y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)

    y_pred = (y_pred * y_train_std) + y_train_mean
    output_std = output_std * y_train_std

    # y_pred = (y_pred * y_train_std2) + y_train_min
    # output_std = output_std * y_train_std2

    metric = compute_metrics(y_test, y_pred, output_std)
    print(f"\nMetrics (Run # {i}):")
    for k, v in metric.items():
        print(f"{k}: {v:.4f}")
    full_metrics.append(metric)
print(f"Input data dtype = {model.train_inputs[0].dtype}")
print(f"Target data dtype = {model.train_targets[0].dtype}, added noise: {y_train_noise}")
# print(f"model dtype = {model.dtype}")
print(f"Optimizer: {trainer.optimizer_class}")
# print(f"Optimizer_kwargs: {trainer.optimizer_kwargs}")
# print(f"convergence_patience: {trainer.convergence_patience}")
print(f"Time: {time.time() - t0:.2f} s\n{num_runs} runs. Lr = {lr}. {num_epochs} epochs")
df_metrics = pd.DataFrame(full_metrics)

# Calculate the mean for each metric across all seeds
avg_metrics = df_metrics.mean().to_dict()
std_metrics = df_metrics.std().to_dict()


# Print averages
print("Average metrics (± std):")
# print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
for metric in avg_metrics.keys():
    mean_val = avg_metrics[metric]
    std_val = std_metrics[metric]
    print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
print("\n")


# i = 0
# full_metrics = []
# t0 = time.time()
# for seed in range(42, 52):
#     t1 = time.time()
#     i += 1
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     # Combine X
#     # Initialize architectures
#     # cat_architecture = {
#     #     'hidden_dims': [5],
#     #     'activation': 'hardtanh',
#     #     'dropout': 0.0
#     # }

#     # Initialize encoders with custom architectures
#     # cat_encoder = None
#     # if data['column_indices']['categorical']:
#     # cat_encoder = OneHotToLatent(
#     #     input_dim=len(data['column_indices']['categorical']),
#     #     architecture_config=cat_architecture,
#     #     z_dim=2
#     # )
#     # cat_encoder = MatrixEncoder(
#     #     input_dim=len(arr()),
#     #     z_dim=2,
#     #     initialization='orthogonal',
#     #     init_std=0.1
#     #     )

#     kernel = gpplus.kernels.CombinedKernel_MVMF(
#         cat_cols=[np.arange(0, 10), np.arange(10, 22), np.arange(22, 24)],
#         # cat_encoder=cat_encoder,
#         # cat_combination_method="product",
#         # cat_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
#     )
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
#     X_train = X_train.to(amp_dtype)
#     X_test = X_test.to(amp_dtype)
#     y_train = y_train.to(amp_dtype)
#     y_test = y_test.to(amp_dtype)

#     y_train_mean = y_train.mean()
#     y_train_std = y_train.std()
#     y_train_normal = (y_train - y_train_mean) / y_train_std


#     y_train_std_prior = y_train.std().item()
#     prior_mean = np.log(y_train_std_prior**2)
#     noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#     from gpytorch.constraints import GreaterThan

#     model = GPR(
#         X_train,
#         y_train_normal,
#         # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
#         # kernel_module = gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
#         # kernel_module = gpytorch.kernels.RBFKernel(),
#         kernel_module=kernel,
#         # mean_module=gpytorch.means.ZeroMean(),
#         likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6), noise_prior=noise_prior),
#         seed=seed,
#         dtype=model_dtype
#     )

#     trainer = gpplus.training.GPTrainer(
#         model=model,
#         num_epochs=num_epochs,
#         seed=seed,
#         num_runs=num_runs,  # <-----
#         optimizer_kwargs={
#             "lr": lr,
#             # 'weight_decay': 1e-4,
#             # 'eps': 1e-8,
#         },
#         convergence_patience=convergence_patience,
#         optimizer_class=torch.optim.Adam,
#         # optimizer_class=torch.optim.LBFGS,
#         device="cpu",
#         # map_prior=True,
#         # scheduler = .95
#         # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
#         # initializer_class=initializer_class,  # Use the chosen initializer
#     )
#     if i == 1:
#         print(model)
#         print(f"Input data dtype = {model.train_inputs[0].dtype}")
#         print(f"Target data dtype = {model.train_targets[0].dtype}")
#         print(f"model dtype = {model.dtype}")
#         print(f"Optimizer = {trainer.optimizer_class}")
#         print(f"Optimizer_kwargs = {trainer.optimizer_kwargs}")
#         print(f"convergence_patience: {trainer.convergence_patience}")
#         print(f"{num_runs} runs. Lr = {lr}. {num_epochs} epochs")
#     # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
#     results = trainer.train()  # Returns a dict of results; you might store if needed.
#     y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)

#     y_pred = (y_pred * y_train_std) + y_train_mean
#     output_std = output_std * y_train_std

#     metric = compute_metrics(y_test, y_pred, output_std)
#     print(f"\nMetrics (Run # {i}):")
#     for k, v in metric.items():
#         print(f"{k}: {v:.4f}")
#     full_metrics.append(metric)
# print(f"Input data dtype = {model.train_inputs[0].dtype}")
# print(f"Target data dtype = {model.train_targets[0].dtype}")
# print(f"model dtype = {model.dtype}")
# print(f"Optimizer: {trainer.optimizer_class}")
# print(f"Optimizer_kwargs: {trainer.optimizer_kwargs}")
# print(f"convergence_patience: {trainer.convergence_patience}")
# print(f"Time: {time.time() - t0:.2f} s ({num_runs} runs. Lr = {lr})")
# df_metrics = pd.DataFrame(full_metrics)

# # Calculate the mean for each metric across all seeds
# avg_metrics = df_metrics.mean().to_dict()
# std_metrics = df_metrics.std().to_dict()


# # Print averages
# print("Average metrics (± std):")
# # print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
# for metric in avg_metrics.keys():
#     mean_val = avg_metrics[metric]
#     std_val = std_metrics[metric]
#     print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
# print("\n")
