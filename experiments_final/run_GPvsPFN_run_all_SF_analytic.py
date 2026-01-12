from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
# from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
# from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
# from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN



folder = "results"
date = "1_06"

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
save_path_am_porosity = f"./{folder}/{date}/am_data_porosity/"
save_path_am_hardness = f"./{folder}/{date}/am_data_hardness/"



# %% Wing ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing)
wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing)

# wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
# wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)

# wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
# wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)


# # %% Buckling ------------------------------------------------------------------------------------------------

buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling)
buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling)

# buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
# buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)

# buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
# buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)

# %% Borehole ------------------------------------------------------------------------------------------------

borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole)
borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole)

# borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
# borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)

# borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
# borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)

# # %% Ackley ------------------------------------------------------------------------------------------------

ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley)

ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True)

ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley)
ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley)

ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley_V2, V2=True)


# # %% Ackley with noise ------------------------------------------------------------------------------------------------
# save_path_ackley = f"./results/ackley/{date}/5D"
# ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)

# ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

# save_path_ackley = f"./results/ackley/{date}/40D"
# ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)

# ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/5D"
# ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# save_path_ackley_V2 = f"./results/ackleyV2/{date}/40D"
# ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
# ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)

# ackley_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
# ackley_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# %% Rastrigin ------------------------------------------------------------------------------------------------
# Rastrigin 5xdim
# No noise original bounds [-10, 10]
rastrigin_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
# No noise benchmark bounds [-5.12, 5.12]
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)


# 0.005 noise
# rastrigin_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)

# # Rastrigin 40xdim
# # No noise original bounds [-10, 10]
rastrigin_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
# # No noise benchmark bounds [-5.12, 5.12]
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=10, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rastrigin_GPvsPFN(bounds=[-5.12, 5.12], train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
# 0.005 noise
# rastrigin_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)


# Rosenbrock 5xdim
# No noise
rosenbrock_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.0, noise_test=0.0)
# 0.005 noise
# rosenbrock_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.005, noise_test=0.005)
# # # 0.05 noise
# rosenbrock_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_rosenbrock, noise_train=0.05, noise_test=0.05)


# # Rosenbrock 40xdim
save_path_rosenbrock_40d = f"{save_path_rosenbrock}/40xdim"
# # No noise
rosenbrock_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rosenbrock_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# # # 0.05 noise
# rosenbrock_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)


# # # %% Zakharov ------------------------------------------------------------------------------------------------
save_path_zakharov_5d = f"{save_path_zakharov}/5xdim"
zakharov_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
zakharov_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
zakharov_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)
zakharov_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.0, noise_test=0.0)

# zakharov_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.005, noise_test=0.005)

# zakharov_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_zakharov_5d, noise_train=0.05, noise_test=0.05)

# # 40xdim
save_path_zakharov_40d = f"{save_path_zakharov}/40xdim"
zakharov_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.0, noise_test=0.0)
zakharov_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.0, noise_test=0.0)

# zakharov_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.005, noise_test=0.005)
# zakharov_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.005, noise_test=0.005)

# zakharov_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.05, noise_test=0.05)
# zakharov_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_zakharov_40d, noise_train=0.05, noise_test=0.05)

# %% Griewank ------------------------------------------------------------------------------------------------
save_path_griewank_5d = f"{save_path_griewank}/5xdim"
griewank_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.0, noise_test=0.0)
#0.005 Noise
# griewank_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.005, noise_test=0.005)
# #0.05 Noise
# griewank_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_griewank_5d, noise_train=0.05, noise_test=0.05)

save_path_griewank_40d = f"{save_path_griewank}/40xdim"
griewank_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.0, noise_test=0.0)
griewank_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.0, noise_test=0.0)
#0.005 Noise
# griewank_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.005, noise_test=0.005)
# griewank_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.005, noise_test=0.005)
# #0.05 Noise
# griewank_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.05, noise_test=0.05)
# griewank_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_griewank_40d, noise_train=0.05, noise_test=0.05)

# %% Dixon Price ------------------------------------------------------------------------------------------------
save_path_dixon_price_5d = f"{save_path_dixon_price}/5xdim"
dixon_price_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.0, noise_test=0.0)
# 0.005 Noise
# dixon_price_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# dixon_price_GPvsPFN(train_size=10, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(train_size=20, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(train_size=40, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(train_size=80, dimensions=5, save_path=save_path_dixon_price_5d, noise_train=0.05, noise_test=0.05)

# # 40xdim
save_path_dixon_price_40d = f"{save_path_dixon_price}/40xdim"
dixon_price_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.0, noise_test=0.0)
dixon_price_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.0, noise_test=0.0)
# 0.005 Noise
# dixon_price_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.005, noise_test=0.005)
# dixon_price_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.005, noise_test=0.005)
# # 0.05 Noise
# dixon_price_GPvsPFN(train_size=10, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.05, noise_test=0.05)
# dixon_price_GPvsPFN(train_size=20, dimensions=40, save_path=save_path_dixon_price_40d, noise_train=0.05, noise_test=0.05)
