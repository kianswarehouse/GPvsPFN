import gpplus
import torch

X_STANDARDIZE_METHOD = 2 # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
STANDARDIZE_X = True
STANDARDIZE_Y = True

SF_kernel = None
SF_mean = None
SF_likelihood = None
NUM_FOLDS = 20
TRAINER_LR = None # Using LBFGS
TRAINER_MIN_EPOCHS = 0  # Do not consider early stopping until at least this many epochs
TRAINER_NUM_EPOCHS = 200 
TRAINER_NUM_RUNS = 16
TRAINER_CONVERGENCE_PATIENCE = 10
# TRAINER_CHOLESKY_JITTER = 0
# TRAINER_CHOLESKY_JITTER = 1e-12
TRAINER_CHOLESKY_JITTER = 1e-6
TRAINER_MIN_LOSS_CHANGE = 1e-7
# TRAINER_OPTIMIZER_CLASS = torch.optim.Adam
TRAINER_OPTIMIZER_CLASS = gpplus.training.optimizers.LBFGSScipy
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 2000, "max_eval": 5000, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #1
TRAINER_OPTIMIZER_KWARGS = {"max_iter": 20}   # 0 
# TRAINER_OPTIMIZER_KWARGS = None
TRAINER_INITIALIZER_CLASS = gpplus.training.parameter_initializer.DefaultParameterInitializer
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32
NOISE_TYPE = 'gaussian'


SEED = 42
SEED_TRAINER = None

def MF_kernel(
    cont_cols=None,
    cat_cols=None,
    source_cols=None,
    cont_kernel=None,
    cat_kernel=None,
    source_kernel=None,
    cat_encoder=None,
    source_encoder=None,
    z_dim=2,
    fix_lengthscale_cat=False,
    fix_lengthscale_source=False,
    **kwargs
):
    """Factory function that creates and wraps the MF kernel."""
    return gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            source_cols=source_cols,
            cont_kernel=cont_kernel,
            cat_kernel=cat_kernel,
            source_kernel=source_kernel,
            cat_encoder=cat_encoder,
            source_encoder=source_encoder,
            z_dim=z_dim,
            fix_lengthscale_cat=fix_lengthscale_cat,
            fix_lengthscale_source=fix_lengthscale_source,
            **kwargs
        )
    )

# For current experiments, we are not using the following MF methods
# MF_mean = gpplus.means.MultiMean
# MF_likelihood = gpplus.likelihoods.MultiLikelihood
# MF_STANDARDIZATION_METHOD = 2 # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently