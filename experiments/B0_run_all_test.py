# Run this script before running '00_run_all_experiments.py' to check that all the experiments run without errors.

from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN
from A10_tabpfn1d_abs_x_GPvsPFN import tabpfn1d_abs_x_GPvsPFN
from A12_tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN import tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN
from A14_tabpfn1d_step_GPvsPFN import tabpfn1d_step_GPvsPFN
from A15_tabpfn1d_x_squared_GPvsPFN import tabpfn1d_x_squared_GPvsPFN
from A18_tabpfn1d_linear_homoscedastic_GPvsPFN import tabpfn1d_linear_homoscedastic_GPvsPFN
from A20_pumadyn32_GPvsPFN import pumadyn32_GPvsPFN
from A21_elevators_GPvsPFN import elevators_GPvsPFN

folder = "results"
label = "B0_test"

save_path_wing = f"./{folder}/{label}/A1_wing/"
save_path_buckling = f"./{folder}/{label}/A2_buckling/"
save_path_ackley = f"./{folder}/{label}/A4_ackley/"
save_path_rosenbrock = f"./{folder}/{label}/A6_rosenbrock/"
save_path_griewank = f"./{folder}/{label}/A8_griewank/"
save_path_dixon_price = f"./{folder}/{label}/A9_dixon_price/"
save_path_abs_x = f"./{folder}/{label}/A10_abs_x/"
save_path_0p3sin_6p5pi_x_plus_0p5x = f"./{folder}/{label}/A12_0p3sin_6p5pi_x_plus_0p5x/"
save_path_step = f"./{folder}/{label}/A14_step/"
save_path_x_squared = f"./{folder}/{label}/A15_x_squared/"
save_path_linear_homoscedastic = f"./{folder}/{label}/A18_linear_homoscedastic/"
save_path_pumadyn32 = f"./{folder}/{label}/A20_pumadyn32/"
save_path_elevators = f"./{folder}/{label}/A21_elevators/"

num_test = 5000
run_models = None # Both models will be used
# run_models = 'gp' # Only GP+ model will be used
# run_models = 'pfn' # Only PFN model will be used
title = None

test_outside_margin = 0.3 # Extrapolation control for A10-A17
explicit_test_x_bounds = None # Extrapolation control for A10-A17


num_runs = 1

wing_SF_GPvsPFN(title="warmup", num_runs=num_runs, train_size=10, save_path=None, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Wing SF ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Buckling SF ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Ackley ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Rosenbrock ------------------------------------------------------------------------------------------------
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Griewank ------------------------------------------------------------------------------------------------
griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Dixon-Price ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_dixon_price, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Abs x ------------------------------------------------------------------------------------------------
tabpfn1d_abs_x_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_abs_x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% 0.3sin(6.5pi*x) + 0.5x ------------------------------------------------------------------------------------------------
tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN(title=title, num_runs=num_runs, train_size=41, save_path=save_path_0p3sin_6p5pi_x_plus_0p5x, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Step ------------------------------------------------------------------------------------------------
tabpfn1d_step_GPvsPFN(title=title, num_runs=num_runs, train_size=44, save_path=save_path_step, noise_train=0.08, noise_test=0.08, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% X squared ------------------------------------------------------------------------------------------------
tabpfn1d_x_squared_GPvsPFN(title=title, num_runs=num_runs, train_size=11, save_path=save_path_x_squared, noise_train=0.002, noise_test=0.002, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# %% Linear homoscedastic ------------------------------------------------------------------------------------------------
tabpfn1d_linear_homoscedastic_GPvsPFN(title=title, num_runs=num_runs, train_size=100, save_path=save_path_linear_homoscedastic, noise_train=0.15, noise_test=0.15, num_test=num_test, run_models=run_models, test_outside_margin=test_outside_margin, test_x_bounds=explicit_test_x_bounds)

# Pumadyn32 ------------------------------------------------------------------------------------------------
pumadyn32_GPvsPFN(title=title, num_runs=num_runs, train_size=20, train_pool_size=5192, test_pool_size=3000, split_seed=42, save_path=save_path_pumadyn32, run_models=run_models)

# Elevators ------------------------------------------------------------------------------------------------
elevators_GPvsPFN(title=title, num_runs=num_runs, train_size=20, test_pool_size=3000, split_seed=42, save_path=save_path_elevators, run_models=run_models)