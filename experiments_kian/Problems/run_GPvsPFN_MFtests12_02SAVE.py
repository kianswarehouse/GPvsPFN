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

date = "12_02eval1"
from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# # optimizer = torch.optim.Adam
num_seeds = 10
num_runs = 16
title = "m1"
s_method = 1
# # %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
save_path_borehole = f"./results/borehole/{date}/default"
save_path_buckling = f"./results/buckling/{date}/default"

wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing, optimizer_class=optimizer)
wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.005, noise_test=0.005, save_path=save_path_wing, optimizer_class=optimizer)
wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.05, noise_test=0.05, save_path=save_path_wing, optimizer_class=optimizer)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, noise_train=0.005, noise_test=0.005, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, noise_train=0.05, noise_test=0.05, save_path=save_path_buckling, optimizer_class=optimizer)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.005, noise_test=0.005, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.05, noise_test=0.05, save_path=save_path_borehole, optimizer_class=optimizer)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)

title = "m2"
s_method = 2
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)
wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, optimizer_class=optimizer, standardization_method=s_method)

buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, noise_train=0.005, noise_test=0.005, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=20, noise_train=0.05, noise_test=0.05, save_path=save_path_buckling, optimizer_class=optimizer)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)
buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, optimizer_class=optimizer, standardization_method=s_method)

borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.005, noise_test=0.005, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, noise_train=0.05, noise_test=0.05, save_path=save_path_borehole, optimizer_class=optimizer)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)

borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)
borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, optimizer_class=optimizer, standardization_method=s_method)



# date = "11_201"
# save_path_wing = f"./results/wing/{date}/default"
# save_path_borehole = f"./results/borehole/{date}/default"
# save_path_buckling = f"./results/buckling/{date}/default"
# save_path_buckling_noise = f"./results/buckling/{date}/noise"


# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=80, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
# wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[1, 2, 3, 4], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)

# gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)

# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[1, 2, 3, 4, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5, 5, 5, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)

# date = "11_192"
# save_path_wing = f"./results/wing/{date}/default"
# save_path_borehole = f"./results/borehole/{date}/default"
# save_path_buckling = f"./results/buckling/{date}/default"
# save_path_buckling_noise = f"./results/buckling/{date}/noise"

# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer)
# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=80, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 5, 3, 2], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[1, 2, 3, 4], num_test=[5000,1000,1000,1000],save_path=save_path_wing, optimizer_class=optimizer)

# gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.05, 0.0], noise_test=[0.05, 0.0], optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10], num_test=[5000,1000],save_path=save_path_buckling_noise, noise_train=[0.005, 0.0], noise_test=[0.005, 0.0], optimizer_class=optimizer)

# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[1, 2, 3, 4, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 1, 1, 1, 1], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5, 5, 5, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)


# # wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer)
# # buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_buckling, optimizer_class=optimizer)
# # borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_borehole, optimizer_class=optimizer)
# # wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[4, 3, 2, 1], num_test=[5000, 1000, 1000, 1000], save_path=save_path_wing, optimizer_class=optimizer)
# # buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000, 1000],save_path=save_path_buckling, optimizer_class=optimizer)
# # borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5, 5, 5, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=80, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer)
# wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20, 20, 20], num_test=[5000, 1000, 1000, 1000], save_path=save_path_wing, optimizer_class=optimizer)

# date = "11_18adam"
# save_path_wing = f"./results/wing/{date}/default"
# save_path_borehole = f"./results/borehole/{date}/default"
# save_path_buckling = f"./results/buckling/{date}/default"
# save_path_buckling_noise = f"./results/buckling/{date}/noise"

# optimizer = torch.optim.Adam
# # wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer)
# # buckling_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_buckling, optimizer_class=optimizer)
# # borehole_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=10, num_test=5000, save_path=save_path_borehole, optimizer_class=optimizer)
# # wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[4, 3, 2, 1], num_test=[5000, 1000, 1000, 1000], save_path=save_path_wing, optimizer_class=optimizer)
# # buckling_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5], num_test=[5000, 1000],save_path=save_path_buckling, optimizer_class=optimizer)
# # borehole_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[5, 5, 5, 5, 5], num_test=[5000, 750, 750, 750, 750], save_path=save_path_borehole, optimizer_class=optimizer)
# wing_SF_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=80, num_test=5000, save_path=save_path_wing, optimizer_class=optimizer)
# wing_GPvsPFN(title=title, num_seeds=num_seeds, num_runs=num_runs, train_size=[20, 20, 20, 20], num_test=[5000, 1000, 1000, 1000], save_path=save_path_wing, optimizer_class=optimizer)

