# %% Import GPyTorch Analytic Problems ------------------------------------------------------------------------------------------------
import time

from A1_wing_SF_GPvsPFN_gpytorch import wing_SF_GPvsPFN as wing_GPvsPFN
from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN as buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN_gpytorch import borehole_SF_GPvsPFN as borehole_GPvsPFN
from A4_Ackley_GPvsPFN_gpytorch import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN_gpytorch import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN_gpytorch import rosenbrock_GPvsPFN
from M2AX_GPvsPFN_gpytorch import M2AX_GPvsPFN as m2ax_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics

date = "12_16"
# Note: optimizer_class and initializer_class parameters are accepted but ignored
# GPyTorch versions use pure gpytorch defaults: Adam optimizer with lr=0.001

num_folds = 20
num_runs = 16
num_runs_large_dim = 1
folder = "gpytorch_default_results"
start_time = time.time()

# %% Wing ------------------------------------------------------------------------------------------------
# save_path_wing = f"./{folder}/wing/{date}"
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing)

# save_path_wing_noise = f"./{folder}/wing/{date}"
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)

# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)

# # %% Buckling ------------------------------------------------------------------------------------------------

# save_path_buckling = f"./{folder}/buckling/{date}"
# # # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling)
# # # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling)
# # # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling)

# save_path_buckling_noise = f"./{folder}/buckling/{date}"
# # # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# # buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)

# # %% Rosenbrock 5Dx------------------------------------------------------------------------------------------------

# save_path_rosenbrock = f"./{folder}/rosenbrock/{date}/5Dx"
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_rosenbrock)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_rosenbrock)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_rosenbrock)

# save_path_rosenbrock_noise = f"./{folder}/rosenbrock/{date}/5Dx"
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)

# # %% Rosenbrock 20Dx------------------------------------------------------------------------------------------------

# save_path_rosenbrock = f"./{folder}/rosenbrock/{date}/20Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock)
# save_path_rosenbrock_noise = f"./{folder}/rosenbrock/{date}/20Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)

# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)

# # # %% Rosenbrock 40Dx------------------------------------------------------------------------------------------------

# save_path_rosenbrock = f"./{folder}/rosenbrock/{date}/40Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock)

# save_path_rosenbrock_noise = f"./{folder}/rosenbrock/{date}/40Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)

# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)

# # %% Rosenbrock 80Dx------------------------------------------------------------------------------------------------

# save_path_rosenbrock = f"./{folder}/rosenbrock/{date}/80Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=80, save_path=save_path_rosenbrock)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock)

# save_path_rosenbrock_noise = f"./{folder}/rosenbrock/{date}/80Dx"
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.05, noise_test=0.05)

# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_noise, noise_train=0.005, noise_test=0.005)

# # %% Rastrigin 5Dx------------------------------------------------------------------------------------------------

# save_path_rastrigin = f"./{folder}/rastrigin/{date}/5Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/5Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/5Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)

# # %% Rastrigin 40Dx------------------------------------------------------------------------------------------------

# save_path_rastrigin = f"./{folder}/rastrigin/{date}/40Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/40Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/40Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)

# # %% Rastrigin 80Dx------------------------------------------------------------------------------------------------

# save_path_rastrigin = f"./{folder}/rastrigin/{date}/80Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rastrigin)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/80Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.05, noise_test=0.05)

# save_path_rastrigin_noise = f"./{folder}/rastrigin/{date}/80Dx"
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=10, dimensions=80, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs_large_dim, train_size=20, dimensions=80, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_noise, noise_train=0.005, noise_test=0.005)

# %% Borehole ------------------------------------------------------------------------------------------------

# save_path_borehole = f"./{folder}/borehole/{date}"
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole)

# save_path_borehole_noise = f"./{folder}/borehole/{date}"
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)

# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)

# %% Ackley 40Dx ------------------------------------------------------------------------------------------------

# save_path_ackley = f"./{folder}/ackley/{date}/40Dx"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_ackley)

save_path_ackley_noise = f"./{folder}/ackley/{date}/40Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)

ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)

# %% Ackley 20Dx ------------------------------------------------------------------------------------------------

save_path_ackley = f"./{folder}/ackley/{date}/20Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley)

save_path_ackley_noise = f"./{folder}/ackley/{date}/20Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)

ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)

# # %% Ackley 10Dx ------------------------------------------------------------------------------------------------

save_path_ackley = f"./{folder}/ackley/{date}/10Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley)

save_path_ackley_noise = f"./{folder}/ackley/{date}/10Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)

ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)

# # %% Ackley 5Dx------------------------------------------------------------------------------------------------

save_path_ackley = f"./{folder}/ackley/{date}/5Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley)

save_path_ackley_noise = f"./{folder}/ackley/{date}/5Dx"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.05, noise_test=0.05)

ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_noise, noise_train=0.005, noise_test=0.005)

# %% M2AX ------------------------------------------------------------------------------------------------

# save_path_m2ax = f"./{folder}/m2ax/{date}"
# m2ax_GPvsPFN(num_seeds=num_folds, num_runs=num_runs, test_size=0.103, save_path=save_path_m2ax)
# m2ax_GPvsPFN(num_seeds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_m2ax)
# m2ax_GPvsPFN(num_seeds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_m2ax)
# m2ax_GPvsPFN(num_seeds=num_folds, num_runs=num_runs, test_size=0.9, save_path=save_path_m2ax)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")
# %%
