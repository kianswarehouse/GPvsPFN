import time
from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN
from A10_styblinski_tang_GPvsPFN import styblinski_tang_GPvsPFN

num_folds = 20
num_runs = 16
folder = "results"
date = "12_14"

start_time = time.time()
save_path_wing = f"./{folder}/wing/{date}"
save_path_borehole = f"./{folder}/borehole/{date}"
save_path_buckling = f"./{folder}/buckling/{date}"
save_path_rosenbrock = f"./{folder}/rosenbrock/{date}"
save_path_rastrigin = f"./{folder}/rastrigin/{date}"
save_path_zakharov = f"./{folder}/zakharov/{date}"
save_path_griewank = f"./{folder}/griewank/{date}"
save_path_dixon_price = f"./{folder}/dixon_price/{date}"
save_path_styblinski = f"./{folder}/styblinski/{date}"

# %% Wing ------------------------------------------------------------------------------------------------

# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing)

# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)

# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)

# # %% Buckling ------------------------------------------------------------------------------------------------
# title ="log1p"
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling)

# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)

# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)

# # %% Borehole ------------------------------------------------------------------------------------------------

# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole)

# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)

# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)

# # # %% Ackley ------------------------------------------------------------------------------------------------

# save_path_ackley = f"./{folder}/ackley/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley)

# save_path_ackley = f"./{folder}/ackley/{date}/10D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley)

# save_path_ackley = f"./{folder}/ackley/{date}/20D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley)

# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True)

# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/10D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True)

# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/20D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True)


# # # %% Ackley with noise ------------------------------------------------------------------------------------------------
# # save_path_ackley = f"./{folder}/ackley/{date}/5D"
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

# save_path_ackley = f"./{folder}/ackley/{date}/10D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

# save_path_ackley = f"./{folder}/ackley/{date}/20D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)


# # save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/10D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/20D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# # ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)



# # # %% Wing ------------------------------------------------------------------------------------------------
# title="fixedm2"
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# # Below is probably too computationaly heavy
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], save_path=save_path_wing)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
# # wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# # %% Buckling ------------------------------------------------------------------------------------------------
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# # %% Borehole  -----------------------------------------------------------------------------------------------
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# Below is probably too computationaly heavy
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

# save_path_ackley = f"./{folder}/ackley/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley)


# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True)



# # # %% Ackley with noise ------------------------------------------------------------------------------------------------
# save_path_ackley = f"./{folder}/ackley/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)


# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# save_path_ackley = f"./{folder}/ackley/{date}/20D"
# save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/20D"

# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# Rastrigin 5xdim
# save_path_rastrigin_5d = f"{save_path_rastrigin}/5xdim"
# No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)

# # Rastrigin 20xdim
# save_path_rastrigin_20d = f"{save_path_rastrigin}/20xdim"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)

# # Rastrigin 40xdim
# save_path_rastrigin_40d = f"{save_path_rastrigin}/40xdim"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)


# # Shifted Rastrigin 5xdim
# save_path_rastrigin_5d_shifted = f"{save_path_rastrigin}/5xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# Shifted Rastrigin 20xdim
# save_path_rastrigin_20d_shifted = f"{save_path_rastrigin}/20xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# # Shifted Rastrigin 40xdim
# save_path_rastrigin_40d_shifted = f"{save_path_rastrigin}/40xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# x_bounds = [-5.12, 5.12]
# Rastrigin 5xdim
# save_path_rastrigin_5d = f"{save_path_rastrigin}/5xdim"
# No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)

# Rastrigin 20xdim
# save_path_rastrigin_20d = f"{save_path_rastrigin}/20xdim"
# No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.05, noise_test=0.05)

# # Rastrigin 40xdim
# save_path_rastrigin_40d = f"{save_path_rastrigin}/40xdim"
# # No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.05, noise_test=0.05)


# # Shifted Rastrigin 5xdim
# save_path_rastrigin_5d_shifted = f"{save_path_rastrigin}/5xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=160, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# # Shifted Rastrigin 20xdim
# save_path_rastrigin_20d_shifted = f"{save_path_rastrigin}/20xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# # Shifted Rastrigin 40xdim
# save_path_rastrigin_40d_shifted = f"{save_path_rastrigin}/40xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(x_bounds=x_bounds, num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# %% Rosenbrock ------------------------------------------------------------------------------------------------

# # Rosenbrock 5xdim
# save_path_rosenbrock_5d = f"{save_path_rosenbrock}/5xdim"
# # No noise
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
# # # 0.05 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)

# # Rosenbrock 20xdim
# save_path_rosenbrock_20d = f"{save_path_rosenbrock}/20xdim"
# # No noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)


# # Rosenbrock 40xdim
# save_path_rosenbrock_40d = f"{save_path_rosenbrock}/40xdim"
# # No noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# # # 0.005 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# # # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# # # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# # # 0.05 noise
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# # # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# # # rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)


# # %% Zakharov ------------------------------------------------------------------------------------------------
# save_path_zakharov_5d = f"{save_path_zakharov}/5xdim"
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)

# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)

# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)

# # 40xdim
# save_path_zakharov_40d = f"{save_path_zakharov}/40xdim"
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.0, noise_test=0.0)
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.0, noise_test=0.0)
# # zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.0, noise_test=0.0)

# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.005, noise_test=0.005)

# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.05, noise_test=0.05)

# %% Griewank ------------------------------------------------------------------------------------------------
save_path_griewank_5d = f"{save_path_griewank}/5xdim"
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
# #0.005 Noise
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# #0.05 Noise
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)

save_path_griewank_40d = f"{save_path_griewank}/40xdim"
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
# #0.005 Noise
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# #0.05 Noise
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)

# %% Dixon Price ------------------------------------------------------------------------------------------------
save_path_dixon_price_5d = f"{save_path_dixon_price}/5xdim"
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 Noise
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)

# 40xdim
save_path_dixon_price_40d = f"{save_path_dixon_price}/40xdim"
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.0, noise_test=0.0)
# # 0.005 Noise
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.05, noise_test=0.05)


# %% Styblinksi
save_path_styblinski = f"{save_path_styblinski}/5xdim"
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_styblinski, noise_train=0.0, noise_test=0.0)
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_styblinski, noise_train=0.0, noise_test=0.0)
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_styblinski, noise_train=0.0, noise_test=0.0)
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_styblinski, noise_train=0.0, noise_test=0.0)
# # 0.005 Noise
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_styblinski, noise_train=0.005, noise_test=0.005)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_styblinski, noise_train=0.005, noise_test=0.005)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_styblinski, noise_train=0.005, noise_test=0.005)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_styblinski, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_styblinski, noise_train=0.05, noise_test=0.05)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_styblinski, noise_train=0.05, noise_test=0.05)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_styblinski, noise_train=0.05, noise_test=0.05)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_styblinski, noise_train=0.05, noise_test=0.05)

save_path_styblinski_40d = f"{save_path_styblinski}/40xdim"
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.0, noise_test=0.0)
styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.0, noise_test=0.0)
# # 0.005 Noise
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.005, noise_test=0.005)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.05, noise_test=0.05)
# styblinski_tang_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_styblinski_40d, noise_train=0.05, noise_test=0.05)


end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")