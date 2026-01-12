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
date = "1_11"

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
save_path_styblinski = f"./{folder}/{date}/styblinski/"

# %% Wing ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling)

buckling_SF_GPvsPFN(title="SF", MF_kernel=False, train_size=10, save_path=save_path_buckling)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, train_size=20, save_path=save_path_buckling)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, train_size=40, save_path=save_path_buckling)
buckling_SF_GPvsPFN(title="SF", MF_kernel=False, train_size=80, save_path=save_path_buckling)

# %% Borehole ------------------------------------------------------------------------------------------------
borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole)

# %% 5Dx Problems ------------------------------------------------------------------------------------------------
# # %% Ackley ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley)

# # %% Ackley V2 ------------------------------------------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley, V2=True) 
ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley, V2=True)
ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley, V2=True)
ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley, V2=True)

# %% Rastrigin ------------------------------------------------------------------------------------------------
rastrigin_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock)
rastrigin_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock)
rastrigin_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock)
rastrigin_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock)

# %% Rosenbrock ------------------------------------------------------------------------------------------------
rosenbrock_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock)

# # # %% Zakharov ------------------------------------------------------------------------------------------------
zakharov_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_zakharov)

# %% Griewank ------------------------------------------------------------------------------------------------
griewank_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_griewank)

# %% Dixon Price ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_dixon_price)

# %% 10 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_rastrigin)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_rosenbrock)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_zakharov)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_griewank)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(train_size=10, dimensions=10, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=20, dimensions=10, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=40, dimensions=10, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=80, dimensions=10, save_path=save_path_dixon_price)

# %% 20 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_ackley)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_rastrigin)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_rosenbrock)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_zakharov)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_griewank)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(train_size=10, dimensions=20, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=20, dimensions=20, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=40, dimensions=20, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=80, dimensions=20, save_path=save_path_dixon_price)

# %% 40 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_ackley)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_ackley_V2, V2=True)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_rastrigin)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_rosenbrock)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_zakharov)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_griewank)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=40, dimensions=40, save_path=save_path_dixon_price)

# %% 80 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_ackley)

# %% AckleyV2 -------------------------------------------------------------
ackley_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_ackley_V2, V2=True)

# %% Rastrigin --------------------------------------------------------------
rastrigin_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_rastrigin)
rastrigin_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_rastrigin)

# %% Rosenbrock -------------------------------------------------------------
rosenbrock_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_rosenbrock)
rosenbrock_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_rosenbrock)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_zakharov)
zakharov_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_zakharov)

# %% Griewank -------------------------------------------------------------
griewank_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_griewank)
griewank_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_griewank)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(train_size=10, dimensions=80, save_path=save_path_dixon_price)
dixon_price_GPvsPFN(train_size=20, dimensions=80, save_path=save_path_dixon_price)