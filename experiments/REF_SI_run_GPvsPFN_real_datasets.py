from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A10_tabpfn1d_abs_x_GPvsPFN import tabpfn1d_abs_x_GPvsPFN
from A12_tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN import tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN
from A14_tabpfn1d_step_GPvsPFN import tabpfn1d_step_GPvsPFN
from A15_tabpfn1d_x_squared_GPvsPFN import tabpfn1d_x_squared_GPvsPFN
from A18_tabpfn1d_linear_homoscedastic_GPvsPFN import tabpfn1d_linear_homoscedastic_GPvsPFN
from A20_pumadyn32_GPvsPFN import pumadyn32_GPvsPFN
from A21_elevators_GPvsPFN import elevators_GPvsPFN

folder = "results"
label = "B3_real_datasets"

save_path_pumadyn32 = f"./{folder}/{label}/A20_pumadyn32/"
save_path_elevators = f"./{folder}/{label}/A21_elevators/"

num_test = 5000
run_models = None
# run_models = 'gp'
# run_models = 'pfn'

title = None

# Extrapolation controls for A10-A17:
# If explicit_test_x_bounds is set, it is used directly.
# Otherwise test domain = [x_bounds[0]-test_outside_margin, x_bounds[1]+test_outside_margin].
# test_outside_margin = 0.0
test_outside_margin = 0.3
explicit_test_x_bounds = None

num_runs = 20

wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=10, save_path=None, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models)

# Pumadyn32 ------------------------------------------------------------------------------------------------
pumadyn32_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_pumadyn32, run_models=run_models)
pumadyn32_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_pumadyn32, run_models=run_models)
pumadyn32_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_pumadyn32, run_models=run_models)

# Elevators ------------------------------------------------------------------------------------------------
elevators_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_elevators, run_models=run_models)
elevators_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_elevators, run_models=run_models)
elevators_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_elevators, run_models=run_models)