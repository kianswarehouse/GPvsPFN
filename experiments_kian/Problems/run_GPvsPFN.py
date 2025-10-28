from M2AX_GPvsPFN import M2AX_GPvsPFN
from planes_GPvsPFN import planes_GPvsPFN
from hartmann_GPvsPFN import hartmann_GPvsPFN
from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.training.parameter_initializer_kian import KianParameterInitializer as KianInitializer

# initializer_class = KianInitializer
initializer_class = None

# %% Buckling ------------------------------------------------------------------------------------------------
save_path_buckling = "./results/buckling/10_17/defaults"

gp_metrics_10, tabpfn_metrics_10 = buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling, initializer_class=initializer_class)
gp_metrics_20, tabpfn_metrics_20 = buckling_GPvsPFN(train_size=[20, 20], save_path=save_path_buckling, initializer_class=initializer_class)
gp_metrics_30, tabpfn_metrics_30 = buckling_GPvsPFN(train_size=[30, 30], save_path=save_path_buckling, initializer_class=initializer_class)
gp_metrics_40, tabpfn_metrics_40 = buckling_GPvsPFN(train_size=[40, 40], save_path=save_path_buckling, initializer_class=initializer_class)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="buckling_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="buckling_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="buckling_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="buckling_20D")
analyze_metrics(gp_metrics_30, print_summary=True, label="GP", title="buckling_30D")
analyze_metrics(tabpfn_metrics_30, print_summary=True, label="TabPFN", title="buckling_30D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="buckling_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="buckling_40D")

# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = "./results/wing/10_17/defaults"

gp_metrics_10, tabpfn_metrics_10 = wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing, initializer_class=initializer_class)
gp_metrics_20, tabpfn_metrics_20 = wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing, initializer_class=initializer_class)
gp_metrics_30, tabpfn_metrics_30 = wing_GPvsPFN(train_size=[30, 30, 30, 30], save_path=save_path_wing, initializer_class=initializer_class)
gp_metrics_40, tabpfn_metrics_40 = wing_GPvsPFN(train_size=[40, 40, 40, 40], save_path=save_path_wing, initializer_class=initializer_class)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="wing_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="wing_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="wing_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="wing_20D")
analyze_metrics(gp_metrics_30, print_summary=True, label="GP", title="wing_30D")
analyze_metrics(tabpfn_metrics_30, print_summary=True, label="TabPFN", title="wing_30D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="wing_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="wing_40D")


# %% M2AX ------------------------------------------------------------------------------------------------
num_epochs_list = [10, 10000, 10, 10000, 10, 10000]
num_runs_list = [1, 1, 4, 4, 16, 16]
lr = 0.1
save_path_m2ax = "./results/m2ax/10_17/defaults"
for i in range(len(num_epochs_list)):
    num_epochs = num_epochs_list[i]
    num_runs = num_runs_list[i]
    gp_metrics_2, tabpfn_metrics_2 = M2AX_GPvsPFN(test_size=0.2, num_epochs=num_epochs, num_runs=num_runs, lr=lr, save_path=save_path_m2ax, initializer_class=initializer_class)
    gp_metrics_4, tabpfn_metrics_4 = M2AX_GPvsPFN(test_size=0.4, num_epochs=num_epochs, num_runs=num_runs, lr=lr, save_path=save_path_m2ax, initializer_class=initializer_class)
    gp_metrics_6, tabpfn_metrics_6 = M2AX_GPvsPFN(test_size=0.6, num_epochs=num_epochs, num_runs=num_runs, lr=lr, save_path=save_path_m2ax, initializer_class=initializer_class)
    gp_metrics_8, tabpfn_metrics_8 = M2AX_GPvsPFN(test_size=0.8, num_epochs=num_epochs, num_runs=num_runs, lr=lr, save_path=save_path_m2ax, initializer_class=initializer_class)

    analyze_metrics(gp_metrics_2, print_summary=True, label="GP", title="M2AX_0.2test")
    analyze_metrics(tabpfn_metrics_2, print_summary=True, label="TabPFN", title="M2AX_0.2test")
    analyze_metrics(gp_metrics_4, print_summary=True, label="GP", title="M2AX_0.4test")
    analyze_metrics(tabpfn_metrics_4, print_summary=True, label="TabPFN", title="M2AX_0.4test")
    analyze_metrics(gp_metrics_6, print_summary=True, label="GP", title="M2AX_0.6test")
    analyze_metrics(tabpfn_metrics_6, print_summary=True, label="TabPFN", title="M2AX_0.6test")
    analyze_metrics(gp_metrics_8, print_summary=True, label="GP", title="M2AX_0.8test")
    analyze_metrics(tabpfn_metrics_8, print_summary=True, label="TabPFN", title="M2AX_0.8test")

# %% Planes ------------------------------------------------------------------------------------------------

save_path_planes = "./results/planes/10_17/defaults"

gp_metrics_10, tabpfn_metrics_10 = planes_GPvsPFN(train_size=10, save_path=save_path_planes, initializer_class=initializer_class)
gp_metrics_20, tabpfn_metrics_20 = planes_GPvsPFN(train_size=20, save_path=save_path_planes, initializer_class=initializer_class)
gp_metrics_30, tabpfn_metrics_30 = planes_GPvsPFN(train_size=30, save_path=save_path_planes, initializer_class=initializer_class)
gp_metrics_40, tabpfn_metrics_40 = planes_GPvsPFN(train_size=40, save_path=save_path_planes, initializer_class=initializer_class)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="planes_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="planes_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="planes_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="planes_20D")
analyze_metrics(gp_metrics_30, print_summary=True, label="GP", title="planes_30D")
analyze_metrics(tabpfn_metrics_30, print_summary=True, label="TabPFN", title="planes_30D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="planes_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="planes_40D")



# %% hartmann ------------------------------------------------------------------------------------------------
save_path_hartmann = "./results/hartmann/10_17/default"

gp_metrics_10, tabpfn_metrics_10 = hartmann_GPvsPFN(train_size=10, num_runs=16, save_path=save_path_hartmann, initializer_class=initializer_class, standardize_X_gp=True)
gp_metrics_20, tabpfn_metrics_20 = hartmann_GPvsPFN(train_size=20, num_runs=16, save_path=save_path_hartmann, initializer_class=initializer_class, standardize_X_gp=True)
gp_metrics_30, tabpfn_metrics_30 = hartmann_GPvsPFN(train_size=30, num_runs=16, save_path=save_path_hartmann, initializer_class=initializer_class, standardize_X_gp=True)
gp_metrics_40, tabpfn_metrics_40 = hartmann_GPvsPFN(train_size=40, num_runs=16, save_path=save_path_hartmann, initializer_class=initializer_class, standardize_X_gp=True)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="hartmann_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="hartmann_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="hartmann_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="hartmann_20D")
analyze_metrics(gp_metrics_30, print_summary=True, label="GP", title="hartmann_30D")
analyze_metrics(tabpfn_metrics_30, print_summary=True, label="TabPFN", title="hartmann_30D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="hartmann_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="hartmann_40D")

