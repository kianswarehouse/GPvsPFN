# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN as wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN as buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN as borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "12_03"
# optimizer = torch.optim.LBFGS  # This is causing the slowdown!
from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam

num_folds = 20
num_runs = 16
# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./results/wing/{date}/default"
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, optimizer_class=optimizer)

save_path_wing_noise = f"./results/wing/{date}/noise"
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)


# # %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = f"./results/buckling/{date}"
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, optimizer_class=optimizer)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = f"./results/borehole/{date}"
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, optimizer_class=optimizer)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, optimizer_class=optimizer)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, optimizer_class=optimizer)

# # %% Ackley ------------------------------------------------------------------------------------------------

# save_path_ackley = f"./results/ackley/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer)

save_path_ackley = f"./results/ackley/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer)

save_path_ackley = f"./results/ackley/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)

save_path_ackley_V2 = f"./results/ackleyV2/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)

save_path_ackley_V2 = f"./results/ackleyV2/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer)


# # %% Ackley with noise ------------------------------------------------------------------------------------------------
# save_path_ackley = f"./results/ackley/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)

save_path_ackley = f"./results/ackley/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)

save_path_ackley = f"./results/ackley/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)

save_path_ackley_V2 = f"./results/ackleyV2/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)

save_path_ackley_V2 = f"./results/ackleyV2/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, optimizer_class=optimizer, noise_train=0.005, noise_test=0.005)