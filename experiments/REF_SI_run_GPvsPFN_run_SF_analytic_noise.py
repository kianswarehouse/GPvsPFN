from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN

folder = "results"
label = "B1_analytic_problems"

save_path_wing = f"./{folder}/{label}/A1_wing/"
save_path_buckling = f"./{folder}/{label}/A2_buckling/"
save_path_ackley = f"./{folder}/{label}/A4_ackley/"
save_path_rosenbrock = f"./{folder}/{label}/A6_rosenbrock/"
save_path_griewank = f"./{folder}/{label}/A8_griewank/"
save_path_dixon_price = f"./{folder}/{label}/A9_dixon_price/"

num_test = 5000
run_models = None # Both models will be used
# run_models = 'gp' # Only GP+ model will be used
# run_models = 'pfn' # Only PFN model will be used
title = None

num_runs = 20

wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=10, save_path=None, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# Wing SF (dx = 10) ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# Buckling SF (dx = 4) ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# Ackley (dx = 20) ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# Rosenbrock (dx = 20) ------------------------------------------------------------------------------------------------
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# Griewank (dx = 40) ------------------------------------------------------------------------------------------------
griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# Dixon-Price (dx = 40) ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_dixon_price, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_dixon_price, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_dixon_price, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_dixon_price, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
