import torch

# Note: These defaults are for gpytorch-based training scripts.
# The shared training helper lives in `gpytorch_train_eval.py`
# as `train_eval_gp_gpytorch_default` (defaults to LBFGS unless overridden).

# TabPFN input preprocessing: True = same X/y scaling as GP; False = raw encoded X and y (SI paper style)
PREPROCESS_PFN = False

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

NUM_RUNS = 10
TRAINER_LR = 0.1  # LBFGS learning rate (default for LBFGS)
TRAINER_NUM_EPOCHS = 1  # One torch LBFGS step per init (gpplus uses TRAINER_NUM_EPOCHS=1)
# TRAINER_NUM_EPOCHS = 100  # legacy: chunked LBFGS with small max_iter per step
TRAINER_NUM_INITS = 16
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

# LBFGS-specific settings (used in train_eval_gp_gpytorch_default).
# Mirrors gpplus presentation defaults (experiments_opt_study/defaults.py):
#   max_iter=2000, max_eval=2500 per LBFGS step.
LBFGS_MAX_ITER = 2000
LBFGS_MAX_EVAL = 2500
LBFGS_TOLERANCE_GRAD = 1e-5
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE = 10
LBFGS_LR = 1  # Line-search step size for torch.optim.LBFGS

# Adam-specific settings (train_eval_gp_gpytorch_default)
ADAM_LR = 1.0
ADAM_NUM_EPOCHS = 1000
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
