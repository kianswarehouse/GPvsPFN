from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A10_tabpfn1d_abs_x_GPvsPFN import tabpfn1d_abs_x_GPvsPFN
from A11_tabpfn1d_sin_2pi_x_GPvsPFN import tabpfn1d_sin_2pi_x_GPvsPFN
from A12_tabpfn1d_sin_2pi_x_plus_x_GPvsPFN import tabpfn1d_sin_2pi_x_plus_x_GPvsPFN
from A13_tabpfn1d_sin_plus_x_GPvsPFN import tabpfn1d_sin_plus_x_GPvsPFN
from A14_tabpfn1d_step_GPvsPFN import tabpfn1d_step_GPvsPFN
from A15_tabpfn1d_x_squared_GPvsPFN import tabpfn1d_x_squared_GPvsPFN
from A16_tabpfn1d_sin_2pi_x_windowed_GPvsPFN import tabpfn1d_sin_2pi_x_windowed_GPvsPFN
from A17_tabpfn1d_sin_2pi_x_plus_x_windowed_GPvsPFN import tabpfn1d_sin_2pi_x_plus_x_windowed_GPvsPFN
from A18_tabpfn1d_linear_homoscedastic_GPvsPFN import tabpfn1d_linear_homoscedastic_GPvsPFN
from A19_tabpfn1d_linear_heteroscedastic_GPvsPFN import tabpfn1d_linear_heteroscedastic_GPvsPFN

folder = "results_April09"
date = "10_runs_logging_full_Gaussian"
# date = "test96"

save_path_abs_x = f"./{folder}/{date}/abs_x/"
save_path_sin_2pi_x = f"./{folder}/{date}/sin_2pi_x/"
save_path_sin_2pi_x_plus_x = f"./{folder}/{date}/sin_2pi_x_plus_x/"
save_path_sin_plus_x = f"./{folder}/{date}/sin_plus_x/"
save_path_step = f"./{folder}/{date}/step/"
save_path_x_squared = f"./{folder}/{date}/x_squared/"
save_path_sin_2pi_x_windowed = f"./{folder}/{date}/sin_2pi_x_windowed/"
save_path_sin_2pi_x_plus_x_windowed = f"./{folder}/{date}/sin_2pi_x_plus_x_windowed/"
save_path_linear_homoscedastic = f"./{folder}/{date}/linear_homoscedastic/"
save_path_linear_heteroscedastic = f"./{folder}/{date}/linear_heteroscedastic/"

num_test = 5000
run_models = None
# run_models = 'gp'
# run_models = 'pfn'
# title = "2.5"
# title = 'V3'
title = None
# Extrapolation controls for A10-A17:
# If explicit_test_x_bounds is set, it is used directly.
# Otherwise test domain = [x_bounds[0]-test_outside_margin, x_bounds[1]+test_outside_margin].
# test_outside_margin = 0.0
test_outside_margin = 0.3
explicit_test_x_bounds = None


num_runs = 10

wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=10, save_path=None, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models)

# Abs x ------------------------------------------------------------------------------------------------
tabpfn1d_abs_x_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_abs_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_abs_x_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_abs_x, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Sin 2pi x ------------------------------------------------------------------------------------------------
tabpfn1d_sin_2pi_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_2pi_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_sin_2pi_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_2pi_x, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Sin 2pi x plus x ------------------------------------------------------------------------------------------------
tabpfn1d_sin_2pi_x_plus_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_2pi_x_plus_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_sin_2pi_x_plus_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_2pi_x_plus_x, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Sin plus x ------------------------------------------------------------------------------------------------
tabpfn1d_sin_plus_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_plus_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_sin_plus_x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_sin_plus_x, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Step ------------------------------------------------------------------------------------------------
tabpfn1d_step_GPvsPFN(title=title, num_runs=num_runs, train_size=44, save_path=save_path_step, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_step_GPvsPFN(title=title, num_runs=num_runs, train_size=44, save_path=save_path_step, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% X squared ------------------------------------------------------------------------------------------------
tabpfn1d_x_squared_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_x_squared, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_x_squared_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_x_squared, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Sin 2pi x windowed (0 outside [-1, 1]) ------------------------------------------------------------------------------------------------
tabpfn1d_sin_2pi_x_windowed_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_sin_2pi_x_windowed, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_sin_2pi_x_windowed_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_sin_2pi_x_windowed, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Sin 2pi x plus x windowed (0 outside [-1, 1]) ------------------------------------------------------------------------------------------------
tabpfn1d_sin_2pi_x_plus_x_windowed_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_sin_2pi_x_plus_x_windowed, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_sin_2pi_x_plus_x_windowed_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_sin_2pi_x_plus_x_windowed, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Linear homoscedastic ------------------------------------------------------------------------------------------------
tabpfn1d_linear_homoscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_linear_homoscedastic, noise_train=0.15, noise_test=0.15, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_linear_homoscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_linear_homoscedastic, noise_train=0.45, noise_test=0.45, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Linear heteroscedastic (noise increases left->right) --------------------------------------------------------------
tabpfn1d_linear_heteroscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_linear_heteroscedastic, noise_train=0.15, noise_test=0.15, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)
tabpfn1d_linear_heteroscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_linear_heteroscedastic, noise_train=0.45, noise_test=0.45, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)