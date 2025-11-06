from M2AX_GPvsPFN import M2AX_GPvsPFN
from planes_GPvsPFN import planes_GPvsPFN
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.training.parameter_initializer import DefaultParameterInitializer
from gpplus.training.optimizers import LBFGSScipy

initializer_class = DefaultParameterInitializer
# %% M2AX ------------------------------------------------------------------------------------------------
# num_epochs_list = [10, 10000, 10, 10000, 10, 10000]
num_epochs_list = [10000]
num_runs_list = [8]
lr = 0.1
save_path_m2ax = "./results/m2ax/MatrixEncoder"
for i in range(len(num_epochs_list)):
    num_epochs = num_epochs_list[i]
    num_runs = num_runs_list[i]
    gp_metrics_2, tabpfn_metrics_2 = M2AX_GPvsPFN(num_seeds=20, test_size=0.2, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_m2ax, initializer_class=initializer_class)
    # # gp_metrics_4, tabpfn_metrics_4 = M2AX_GPvsPFN(num_seeds=20, test_size=0.4, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_m2ax, initializer_class=initializer_class)
    # gp_metrics_6, tabpfn_metrics_6 = M2AX_GPvsPFN(num_seeds=20, test_size=0.6, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_m2ax, initializer_class=initializer_class)
    # gp_metrics_8, tabpfn_metrics_8 = M2AX_GPvsPFN(num_seeds=20, test_size=0.8, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_m2ax, initializer_class=initializer_class)

    analyze_metrics(gp_metrics_2, print_summary=True, label="GP", title="M2AX_0.2test")
    analyze_metrics(tabpfn_metrics_2, print_summary=True, label="TabPFN", title="M2AX_0.2test")
    # # analyze_metrics(gp_metrics_4, print_summary=True, label="GP", title="M2AX_0.4test")
    # analyze_metrics(tabpfn_metrics_4, print_summary=True, label="TabPFN", title="M2AX_0.4test")
    # analyze_metrics(gp_metrics_6, print_summary=True, label="GP", title="M2AX_0.6test")
    # analyze_metrics(tabpfn_metrics_6, print_summary=True, label="TabPFN", title="M2AX_0.6test")
    # analyze_metrics(gp_metrics_8, print_summary=True, label="GP", title="M2AX_0.8test")
    # analyze_metrics(tabpfn_metrics_8, print_summary=True, label="TabPFN", title="M2AX_0.8test")
 
# %% Planes ------------------------------------------------------------------------------------------------

# save_path_planes = "./results/planes/MatrixEncoder"
# initializer_class = None
# for i in range(len(num_runs_list)):
#     num_runs = num_runs_list[i]
#     num_epochs = num_epochs_list[i]
#     gp_metrics_10, tabpfn_metrics_10 = planes_GPvsPFN(train_size=10, num_runs=num_runs, num_epochs=num_epochs, save_path=save_path_planes, optimizer_class=LBFGSScipy, initializer_class=initializer_class)
#     gp_metrics_20, tabpfn_metrics_20 = planes_GPvsPFN(train_size=20, num_runs=num_runs, num_epochs=num_epochs, save_path=save_path_planes, optimizer_class=LBFGSScipy, initializer_class=initializer_class)
#     # gp_metrics_30, tabpfn_metrics_30 = planes_GPvsPFN(train_size=30, save_path=save_path_planes, optimizer_class=LBFGSScipy, initializer_class=initializer_class)
    # gp_metrics_40, tabpfn_metrics_40 = planes_GPvsPFN(train_size=40, save_path=save_path_planes, optimizer_class=LBFGSScipy, initializer_class=initializer_class)
    # gp_metrics_80, tabpfn_metrics_80 = planes_GPvsPFN(train_size=80, save_path=save_path_planes, optimizer_class=LBFGSScipy, initializer_class=initializer_class)
    # analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="planes_10D")
    # analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="planes_10D")
    # analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="planes_20D")
    # analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="planes_20D")
    # analyze_metrics(gp_metrics_30, print_summary=True, label="GP", title="planes_30D")
    # analyze_metrics(tabpfn_metrics_30, print_summary=True, label="TabPFN", title="planes_30D")
    # analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="planes_40D")
    # analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="planes_40D")



# %%
