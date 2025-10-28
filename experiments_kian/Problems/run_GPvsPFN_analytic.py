# %% Import Analytic Problems ------------------------------------------------------------------------------------------------
from A1_wing_SF_GPvsPFN import wing_GPvsPFN as wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_GPvsPFN as buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_GPvsPFN as borehole_SF_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics

# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = "./results/wing/10_28/default"
# gp_metrics_10, tabpfn_metrics_10 = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing)
gp_metrics_10, tabpfn_metrics_10 = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, num_runs=4)
# gp_metrics_20, tabpfn_metrics_20 = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing)
# gp_metrics_20, tabpfn_metrics_20 = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing)
# gp_metrics_40, tabpfn_metrics_40 = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing)
# gp_metrics_80, tabpfn_metrics_80 = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing)
gp_metrics_80, tabpfn_metrics_80 = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, num_runs=4)

# save_path_wing = "./results/wing/10_28/x_not_standardized"
# gp_metrics_10, tabpfn_metrics_10 = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, standardize_X=False)
# gp_metrics_20, tabpfn_metrics_20 = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing, standardize_X=False)
# gp_metrics_40, tabpfn_metrics_40 = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing, standardize_X=False)
# gp_metrics_80, tabpfn_metrics_80 = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, standardize_X=False)

# save_path_wing_noise = "./results/wing/10_28/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005)  
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005)


# %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = "./results/buckling/10_28/default"
# gp_metrics_10, tabpfn_metrics_10 = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling)
gp_metrics_10, tabpfn_metrics_10 = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, num_runs=4)
# gp_metrics_20, tabpfn_metrics_20 = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling)
# gp_metrics_20, tabpfn_metrics_20 = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling)
# gp_metrics_40, tabpfn_metrics_40 = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling)
# gp_metrics_80, tabpfn_metrics_80 = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling)
gp_metrics_80, tabpfn_metrics_80 = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, num_runs=4)

# analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="buckling_10D")
# analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="buckling_10D")
# analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="buckling_20D")
# analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="buckling_20D")
# analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="buckling_40D")
# analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="buckling_40D")
# analyze_metrics(gp_metrics_80, print_summary=True, label="GP", title="buckling_80D")
# analyze_metrics(tabpfn_metrics_80, print_summary=True, label="TabPFN", title="buckling_80D")


# save_path_buckling_noise = "./results/buckling/10_28/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005)

# analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
# analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
# analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
# analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
# analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
# analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
# analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
# analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = "./results/borehole/10_28/default"
# gp_metrics_10, tabpfn_metrics_10 = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole)
gp_metrics_10, tabpfn_metrics_10 = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, num_runs=4)
# gp_metrics_20, tabpfn_metrics_20 = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole)
# gp_metrics_40, tabpfn_metrics_40 = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole)
# gp_metrics_80, tabpfn_metrics_80 = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole)
gp_metrics_80, tabpfn_metrics_80 = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, num_runs=4)



# save_path_borehole_noise = "./results/borehole/10_28/noise"
# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0) 
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0)



# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05)


# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05)



# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0) 
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0)


# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005)

# gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)
# gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005)


# %% Ackley ------------------------------------------------------------------------------------------------

save_path_ackley = "./results/ackley/10_28/default/5D"
gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley)
gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley)
gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley)
gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley)

save_path_ackley = "./result/ackley/10_28/default/10D"
gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley)
gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley)
gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley)
gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley)

save_path_ackley_V2 = "./results/ackleyV2/10_28/default/5D"
gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True)
gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True)
gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True)
gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True)

save_path_ackley_V2 = "./result/ackleyV2/10_28/default/10D"
gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True)
gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True)
gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True)
gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True)

save_path_ackley = "./results/ackley/10_28/x_standardized/5D"
gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley, standardize_X=True)
gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley, standardize_X=True)
gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley, standardize_X=True)
gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley, standardize_X=True)

save_path_ackley = "./result/ackley/10_28/x_standardized/10D"
gp_metrics_10, tabpfn_metrics_10 = ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley, standardize_X=True)
gp_metrics_20, tabpfn_metrics_20 = ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley, standardize_X=True)
gp_metrics_40, tabpfn_metrics_40 = ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley, standardize_X=True)
gp_metrics_80, tabpfn_metrics_80 = ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley, standardize_X=True)

save_path_ackley_V2 = "./results/ackleyV2/10_28/x_standardized/5D"
gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, standardize_X=True)

save_path_ackley_V2 = "./result/ackleyV2/10_28/x_standardized/10D"
gp_metrics_10_V2, tabpfn_metrics_10_V2 = ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_20_V2, tabpfn_metrics_20_V2 = ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_40_V2, tabpfn_metrics_40_V2 = ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
gp_metrics_80_V2, tabpfn_metrics_80_V2 = ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, standardize_X=True)
