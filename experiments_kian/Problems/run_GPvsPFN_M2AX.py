# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
import torch

from M2AX_GPvsPFN import M2AX_GPvsPFN
from M2AX_GPvsPFN_binaryM import M2AX_GPvsPFN as M2AX_GPvsPFN_binaryM
from M2AX_GPvsPFN_allcont import M2AX_GPvsPFN as M2AX_GPvsPFN_allcont
from M2AX_GPvsPFN_zdim import M2AX_GPvsPFN as M2AX_GPvsPFN_zdim
from M2AX_GPvsPFN_binaryM_zdim import M2AX_GPvsPFN as M2AX_GPvsPFN_binaryM_zdim
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
date = "11_17"

from gpplus.training.optimizers import LBFGSScipy
optimizer = LBFGSScipy
# optimizer = torch.optim.Adam
num_seeds = 20
# num_runs = 16

gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}")
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}")
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}")

# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}")

# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_allcont(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_allcont(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_allcont(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}")
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_allcont(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}")

gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}", z_dim=3)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}", z_dim=3)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}", z_dim=3)
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}", z_dim=3)

gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}", z_dim=4)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}", z_dim=4)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}", z_dim=4)
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_zdim(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}", z_dim=4)

gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}", z_dim=3)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}", z_dim=3)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}", z_dim=3)
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}", z_dim=3)

gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=1, save_path=f"./results/M2AX/{date}", z_dim=4)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=4, save_path=f"./results/M2AX/{date}", z_dim=4)
gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=16, save_path=f"./results/M2AX/{date}", z_dim=4)
# gp_metrics_M2AX, tabpfn_metrics_M2AX = M2AX_GPvsPFN_binaryM_zdim(num_seeds=num_seeds, num_runs=64, save_path=f"./results/M2AX/{date}", z_dim=4)

