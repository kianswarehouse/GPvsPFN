# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "11_17"

from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam
num_folds = 20
num_runs = 4
# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[1, 2, 3, 4], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[5, 5, 5, 5], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[5, 10, 10, 10], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 20, 20, 20], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40], save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80], save_path=save_path_wing, optimizer_class=optimizer)

# save_path_wing = f"./results/wing/{date}/x_not_standardized"
# gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)

# save_path_wing_noise = f"./results/wing/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[4, 3, 2, 1], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[4, 3, 2, 1], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)


# %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = f"./results/buckling/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80], save_path=save_path_buckling, optimizer_class=optimizer)

save_path_buckling_noise = f"./results/buckling/{date}/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)

# %% Borehole ------------------------------------------------------------------------------------------------

# save_path_borehole = f"./results/borehole/{date}/default"
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole, optimizer_class=optimizer)

# save_path_borehole_noise = f"./results/borehole/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

# num_runs = 64
# save_path_borehole = f"./results/borehole/{date}/default"
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole, optimizer_class=optimizer)

# save_path_borehole_noise = f"./results/borehole/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# # gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
