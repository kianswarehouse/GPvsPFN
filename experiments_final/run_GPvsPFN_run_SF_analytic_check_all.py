from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN

folder = "results_check"
date = "temp4"

save_path_wing = f"./{folder}/{date}/wing/"
save_path_buckling = f"./{folder}/{date}/buckling/"
save_path_borehole = f"./{folder}/{date}/borehole/"
save_path_ackley = f"./{folder}/{date}/ackley/"
save_path_ackley_V2 = f"./{folder}/{date}/ackleyV2/"
save_path_rosenbrock = f"./{folder}/{date}/rosenbrock/"
save_path_rastrigin = f"./{folder}/{date}/rastrigin/"
save_path_zakharov = f"./{folder}/{date}/zakharov/"
save_path_griewank = f"./{folder}/{date}/griewank/"
save_path_dixon_price = f"./{folder}/{date}/dixon_price/"

num_runs = 3
num_inits = 2
run_models = 'gp'

# %% Wing ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(title="warmup", num_runs=2, num_inits=num_inits, train_size=10, save_path=None, run_models=run_models)
# wing_SF_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, save_path=save_path_wing, run_models=run_models)

# # # %% Buckling ------------------------------------------------------------------------------------------------
# buckling_SF_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, save_path=save_path_buckling, run_models=run_models)

# # %% Borehole ------------------------------------------------------------------------------------------------
# borehole_SF_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, save_path=save_path_borehole, run_models=run_models)

# # %% 5Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Ackley ------------------------------------------------------------------------------------------------
# ackley_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_ackley, run_models=run_models)

# # # %% Ackley V2 ------------------------------------------------------------------------------------------------
# # ackley_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_ackley, V2=True) 

# # %% Rastrigin ------------------------------------------------------------------------------------------------
# rastrigin_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)

# # %% Rosenbrock ------------------------------------------------------------------------------------------------
# rosenbrock_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)

# # %% Zakharov ------------------------------------------------------------------------------------------------
# zakharov_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank ------------------------------------------------------------------------------------------------
griewank_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN(num_runs=num_runs, num_inits=num_inits, train_size=10, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)