import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import gpplus
from examples.data.data_gen import load_data_buckling_MF
from gpplus.models import GPR
from gpplus.training.callbacks import PrintInitialParametersCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils import set_seed
from gpplus.utils.latent_reps import get_latent_representations, plot_encoders


source_encoder = gpplus.utils.MatrixEncoder(input_dim=2, initialization="normal", init_std=0.1, z_dim=2)

source_encoder2 = gpplus.utils.NeuralEncoder(
    input_dim=2, architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0}, z_dim=2
)

cat_encoder0 = gpplus.utils.NeuralEncoder(
    input_dim=len(cat_cols[0]),
    architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
    z_dim=2,
)

cat_encoder1 = gpplus.utils.NeuralEncoder(
    input_dim=len(cat_cols[1]),
    architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
    z_dim=2,
)

cat_encoder2 = gpplus.utils.NeuralEncoder(
    input_dim=len(cat_cols[2]),
    architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
    z_dim=2,
)
cat_encoders = [cat_encoder0, cat_encoder1, cat_encoder2]

# Create model
kernel = gpplus.kernels.CombinedKernel(
    cont_cols=cont_cols,
    cat_cols=cat_cols,
    source_cols=source_cols,
    cat_encoder="matrix",
    # cat_encoder=cat_encoders,
    # source_encoder=source_encoder,
    # source_encoder=source_encoder2,
)

model = GPR(
    X_train,
    y_train,
    kernel_module=kernel,
    mean_module=gpplus.means.MultiMean(encoded_cols=source_cols),
    likelihood=gpplus.likelihoods.MultiLikelihood(encoded_cols=source_cols, training_data=X_train),
    # likelihood=gpytorch.likelihoods.GaussianLikelihood(),
)

   