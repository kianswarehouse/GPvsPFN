# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN as wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN as buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN as borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "11_07"
# optimizer = torch.optim.LBFGS  # This is causing the slowdown!
from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam

num_seeds = 5
num_runs = 16
# # %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing, optimizer_class=optimizer)

# save_path_wing_noise = f"./results/wing/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)  
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)


# %% Buckling ------------------------------------------------------------------------------------------------

# save_path_buckling = f"./results/buckling/{date}/default"
# # gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# # gp_metrics_80, tabpfn_metrics_80 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, optimizer_class=optimizer)

save_path_buckling_noise = f"./results/buckling/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)

# # %% Borehole ------------------------------------------------------------------------------------------------

# save_path_borehole = f"./results/borehole/{date}/default"
# # gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# # gp_metrics_80, tabpfn_metrics_80 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, num_runs=num_runs, num_seeds=num_seeds, optimizer_class=optimizer)
# gp_metrics_10, tabpfn_metrics_10 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, optimizer_class=optimizer)

# save_path_borehole_noise = f"./results/borehole/{date}/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer) 
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer) 
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, optimizer_class=optimizer)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)

# # %% Ackley ------------------------------------------------------------------------------------------------

# save_path_ackley = f"./results/ackley/{date}/default/5D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)

# save_path_ackley = f"./results/ackley/{date}/default/10D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)

# save_path_ackley = f"./results/ackley/{date}/default/20D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
# # gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/default/5D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/default/10D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/default/20D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)

# save_path_ackley = f"./results/ackley/{date}/x_standardized/5D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)

# save_path_ackley = f"./results/ackley/{date}/x_standardized/10D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)

# save_path_ackley = f"./results/ackley/{date}/x_standardized/20D"
# gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)
# # gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, standardize_X=True, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/x_standardized/5D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/x_standardized/10D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/x_standardized/20D"
# gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)
# gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(num_seeds=num_seeds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, standardize_X=True, optimizer_class=optimizer)


