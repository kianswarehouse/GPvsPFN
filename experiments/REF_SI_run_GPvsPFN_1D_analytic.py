from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A10_tabpfn1d_abs_x_GPvsPFN import tabpfn1d_abs_x_GPvsPFN
from A12_tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN import tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN
from A14_tabpfn1d_step_GPvsPFN import tabpfn1d_step_GPvsPFN
from A15_tabpfn1d_x_squared_GPvsPFN import tabpfn1d_x_squared_GPvsPFN
from A18_tabpfn1d_linear_homoscedastic_GPvsPFN import tabpfn1d_linear_homoscedastic_GPvsPFN

folder = "results"
label = "B2_analytic_problems"

save_path_abs_x = f"./{folder}/{label}/A10_abs_x/"
save_path_0p3sin_6p5pi_x_plus_0p5x = f"./{folder}/{label}/A12_0p3sin_6p5pi_x_plus_0p5x/"
save_path_step = f"./{folder}/{label}/A14_step/"
save_path_x_squared = f"./{folder}/{label}/A15_x_squared/"
save_path_linear_homoscedastic = f"./{folder}/{label}/A18_linear_homoscedastic/"

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

num_runs = 5

wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=10, save_path=None, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models)

# Abs x ------------------------------------------------------------------------------------------------
tabpfn1d_abs_x_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_abs_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% 0.3sin(6.5pi*x) + 0.5x ------------------------------------------------------------------------------------------------
tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_0p3sin_6p5pi_x_plus_0p5x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Step ------------------------------------------------------------------------------------------------
tabpfn1d_step_GPvsPFN(title=title, num_runs=num_runs, train_size=44, save_path=save_path_step, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% X squared ------------------------------------------------------------------------------------------------
tabpfn1d_x_squared_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_x_squared, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Linear homoscedastic ------------------------------------------------------------------------------------------------
tabpfn1d_linear_homoscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=300, save_path=save_path_linear_homoscedastic, noise_train=0.35, noise_test=0.35, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
