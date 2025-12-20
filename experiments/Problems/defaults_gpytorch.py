import torch

# Note: These defaults are for gpytorch-based training scripts.
# The shared training helper lives in `experiments/Problems/gpytorch_train_eval.py`
# as `train_eval_gp_gpytorch_default` (defaults to LBFGS unless overridden).

SF_kernel = None
SF_mean = None
SF_likelihood = None

# MF settings (not typically used in gpytorch scripts, but kept for consistency)
MF_mean = None  # gpytorch uses ConstantMean
MF_likelihood = None  # gpytorch uses GaussianLikelihood
def MF_kernel(*args, **kwargs):
    """Not used in gpytorch scripts - they use ScaleKernel(RBFKernel)"""
    return None

MF_STANDARDIZATION_METHOD = 2

NUM_FOLDS = 20
TRAINER_LR = 0.1  # LBFGS learning rate (default for LBFGS)
TRAINER_NUM_EPOCHS = 100
# TRAINER_NUM_EPOCHS = 10000
TRAINER_NUM_RUNS = 16
TRAINER_CONVERGENCE_PATIENCE = 20
TRAINER_OPTIMIZER_CLASS = torch.optim.LBFGS
# TRAINER_OPTIMIZER_CLASS = torch.optim.Adam  # Supported by `train_eval_gp_gpytorch_default` as well
TRAINER_INITIALIZER_CLASS = None  # Not used in gpytorch scripts
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'

SEED = 42
SEED_TRAINER = None

DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32

# LBFGS-specific settings (used in train_eval_gp_gpytorch_default)
LBFGS_MAX_ITER = 20  # Max iterations per LBFGS step
LBFGS_LR = 0.1  # Learning rate for LBFGS
