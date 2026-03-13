from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN

folder = "results_final"
date = "pfnv2_5_IQR_500test"

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

num_test = 500
# run_models = 'gp'
run_models = 'pfn'

# num_runs = 4

# %% Wing ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(num_test=num_test, train_size=10, save_path=save_path_wing, run_models=run_models)
wing_SF_GPvsPFN(num_test=num_test, train_size=20, save_path=save_path_wing, run_models=run_models)
wing_SF_GPvsPFN(num_test=num_test, train_size=40, save_path=save_path_wing, run_models=run_models)
wing_SF_GPvsPFN(num_test=num_test, train_size=80, save_path=save_path_wing, run_models=run_models)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(num_test=num_test, train_size=10, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(num_test=num_test, train_size=20, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(num_test=num_test, train_size=40, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(num_test=num_test, train_size=80, save_path=save_path_buckling, run_models=run_models)

buckling_SF_GPvsPFN(title="SF", MF_kernel=False, num_test=num_test, train_size=10, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, num_test=num_test, train_size=20, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, num_test=num_test, train_size=40, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, num_test=num_test, train_size=80, save_path=save_path_buckling, run_models=run_models)

# %% Borehole ------------------------------------------------------------------------------------------------
borehole_SF_GPvsPFN(num_test=num_test, train_size=10, save_path=save_path_borehole, run_models=run_models)
borehole_SF_GPvsPFN(num_test=num_test, train_size=20, save_path=save_path_borehole, run_models=run_models)
borehole_SF_GPvsPFN(num_test=num_test, train_size=40, save_path=save_path_borehole, run_models=run_models)
borehole_SF_GPvsPFN(num_test=num_test, train_size=80, save_path=save_path_borehole, run_models=run_models)

# %% 5Dx Problems ------------------------------------------------------------------------------------------------
# # %% Ackley ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_ackley, run_models=run_models)

# # %% Ackley V2 ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_ackley, V2=True, run_models=run_models) 
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_ackley, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_ackley, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_ackley, V2=True, run_models=run_models)

# %% Rastrigin ------------------------------------------------------------------------------------------------
rastrigin_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)

# %% Rosenbrock ------------------------------------------------------------------------------------------------
rosenbrock_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)

# # # %% Zakharov ------------------------------------------------------------------------------------------------
zakharov_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank ------------------------------------------------------------------------------------------------
griewank_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN(num_test=num_test, train_size=10, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=20, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=40, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=80, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)

# %% 10 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_ackley, run_models=run_models)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, run_models=run_models)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_rastrigin, run_models=run_models)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_rosenbrock, run_models=run_models)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(num_test=num_test, train_size=10, dimensions=10, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=20, dimensions=10, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=40, dimensions=10, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=80, dimensions=10, save_path=save_path_dixon_price, run_models=run_models)

# %% 20 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_ackley, run_models=run_models)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, run_models=run_models)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_rastrigin, run_models=run_models)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_rosenbrock, run_models=run_models)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(num_test=num_test, train_size=10, dimensions=20, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=20, dimensions=20, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=40, dimensions=20, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=80, dimensions=20, save_path=save_path_dixon_price, run_models=run_models)

# %% 40 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_ackley, run_models=run_models)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_ackley_V2, V2=True, run_models=run_models)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_rastrigin, run_models=run_models)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_rosenbrock, run_models=run_models)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(num_test=num_test, train_size=10, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=20, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=40, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)

# %% 80 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_ackley, run_models=run_models)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_ackley_V2, V2=True, run_models=run_models)
ackley_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_ackley_V2, V2=True, run_models=run_models)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_rastrigin, run_models=run_models)
rastrigin_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_rastrigin, run_models=run_models)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)
rosenbrock_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(num_test=num_test, train_size=10, dimensions=80, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN(num_test=num_test, train_size=20, dimensions=80, save_path=save_path_dixon_price, run_models=run_models)