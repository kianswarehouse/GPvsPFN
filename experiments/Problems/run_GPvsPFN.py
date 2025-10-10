from M2AX_GPvsPFN import M2AX_GPvsPFN
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.training.parameter_initializer_kian import DefaultParameterInitializer
save_path = "./results/kian_initializer_no_prior_check"

gp_metrics_2, tabpfn_metrics_2 = M2AX_GPvsPFN(test_size=0.2, save_path=save_path, initializer_class=DefaultParameterInitializer)
# gp_metrics_4, tabpfn_metrics_4 = M2AX_GPvsPFN(test_size=0.4, save_path=save_path, initializer_class=DefaultParameterInitializer)
# gp_metrics_6, tabpfn_metrics_6 = M2AX_GPvsPFN(test_size=0.6, save_path=save_path, initializer_class=DefaultParameterInitializer)
# gp_metrics_8, tabpfn_metrics_8 = M2AX_GPvsPFN(test_size=0.8, save_path=save_path, initializer_class=DefaultParameterInitializer)

analyze_metrics(gp_metrics_2, print_summary=True, label="GP", title="M2AX_0.2test")
analyze_metrics(tabpfn_metrics_2, print_summary=True, label="TabPFN", title="M2AX_0.2test")
# analyze_metrics(gp_metrics_4, print_summary=True, label="GP", title="M2AX_0.4test")
# analyze_metrics(tabpfn_metrics_4, print_summary=True, label="TabPFN", title="M2AX_0.4test")
# analyze_metrics(gp_metrics_6, print_summary=True, label="GP", title="M2AX_0.6test")
# analyze_metrics(tabpfn_metrics_6, print_summary=True, label="TabPFN", title="M2AX_0.6test")
# analyze_metrics(gp_metrics_8, print_summary=True, label="GP", title="M2AX_0.8test")
# analyze_metrics(tabpfn_metrics_8, print_summary=True, label="TabPFN", title="M2AX_0.8test")

