import torch
from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN
from M2AX_GPvsPFN import M2AX_GPvsPFN
from M2AX_GPvsPFN_binaryM import M2AX_GPvsPFN as M2AX_GPvsPFN_binaryM
from M2AX_GPvsPFN_allcont import M2AX_GPvsPFN as M2AX_GPvsPFN_allcont
from M2AX_GPvsPFN_zdim import M2AX_GPvsPFN as M2AX_GPvsPFN_zdim
from M2AX_GPvsPFN_binaryM_zdim import M2AX_GPvsPFN as M2AX_GPvsPFN_binaryM_zdim
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics

date = "12_01"
from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# # optimizer = torch.optim.Adam
num_seeds = 5
num_runs = 8
title = "m1"
s_method = 1
# # %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
# save_path_borehole = f"./results/borehole/{date}/default"
save_path_buckling = f"./results/buckling/{date}/default"
buckling_SF_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.005, noise_test=0.005, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.05, noise_test=0.05, save_path=save_path_buckling, optimizer_class=optimizer)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=5, save_path=save_path_wing, optimizer_class=optimizer)
wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=5, noise_train=0.005, noise_test=0.005, save_path=save_path_wing, optimizer_class=optimizer)
wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=5, noise_train=0.05, noise_test=0.05, save_path=save_path_wing, optimizer_class=optimizer)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

title = "m2"
s_method = 2

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[2, 1, 1, 1], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
