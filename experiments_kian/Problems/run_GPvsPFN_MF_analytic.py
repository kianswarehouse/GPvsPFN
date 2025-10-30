# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "10_29/no_logscalekernel"

from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam
# num_seeds = 4
# num_runs = 4
# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing, optimizer_class=optimizer)
gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing, optimizer_class=optimizer)

save_path_wing = f"./results/wing/{date}/x_not_standardized"
# gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing, standardize_X=False, optimizer_class=optimizer)

save_path_wing_noise = f"./results/wing/{date}/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.05, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)  
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(train_size=[80, 80, 80, 80], save_path=save_path_wing_noise, noise_train=[0.005, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0], optimizer_class=optimizer)


# %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = f"./results/buckling/{date}/default"
# gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(train_size=10, save_path=save_path_buckling, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = buckling_GPvsPFN(train_size=80, save_path=save_path_buckling, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling, optimizer_class=optimizer)
gp_metrics_20, tabpfn_metrics_20 = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling, optimizer_class=optimizer)

save_path_buckling_noise = f"./results/buckling/{date}/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.0, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(train_size=[80, 80], save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = f"./results/borehole/{date}/default"
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(train_size=10, save_path=save_path_borehole, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = borehole_GPvsPFN(train_size=80, save_path=save_path_borehole, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer)
gp_metrics_20, tabpfn_metrics_20 = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole, optimizer_class=optimizer)

save_path_borehole_noise = f"./results/borehole/{date}/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.05, 0.0, 0.0, 0.0, 0.0], noise_test=[0.05, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.0, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.0, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)    

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(train_size=[40, 40, 40, 40, 40], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(train_size=[80, 80, 80, 80, 80], save_path=save_path_borehole_noise, noise_train=[0.005, 0.0, 0.0, 0.0, 0.0], noise_test=[0.005, 0.0, 0.0, 0.0, 0.0], optimizer_class=optimizer)
