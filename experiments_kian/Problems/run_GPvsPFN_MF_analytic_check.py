# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "11_12test"

from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam
num_seeds = 1
num_runs = 1
# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing, optimizer_class=optimizer)

# %% Buckling ------------------------------------------------------------------------------------------------
save_path_buckling = f"./results/buckling/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], save_path=save_path_buckling, optimizer_class=optimizer)

# %% Borehole ------------------------------------------------------------------------------------------------
save_path_borehole = f"./results/borehole/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer)
