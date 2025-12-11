import time
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_keane_bump_GPvsPFN import keane_bump_GPvsPFN

num_folds = 20
num_runs = 16
folder = "results"
date = "12_10"

start_time = time.time()

save_path_rastrigin = f"./{folder}/rastrigin/{date}"
save_path_rosenbrock = f"./{folder}/rosenbrock/{date}"
save_path_keane_bump = f"./{folder}/keane_bump/{date}"
save_path_ackley = f"./{folder}/ackley/{date}"
save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}"
# %% Rastrigin ------------------------------------------------------------------------------------------------

# # Rastrigin 5xdim
# save_path_rastrigin_5d = f"{save_path_rastrigin}/5xdim"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d, noise_train=0.05, noise_test=0.05)

# # Rastrigin 20xdim
# save_path_rastrigin_20d = f"{save_path_rastrigin}/20xdim"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d, noise_train=0.0, noise_test=0.0)
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
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d, noise_train=0.0, noise_test=0.0)
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
# # 0.005 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# # 0.05 noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rastrigin_5d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# # Shifted Rastrigin 20xdim
# save_path_rastrigin_20d_shifted = f"{save_path_rastrigin}/20xdim_shifted"
# # No noise
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# # rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rastrigin_20d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
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

# Shifted Rastrigin 40xdim
save_path_rastrigin_40d_shifted = f"{save_path_rastrigin}/40xdim_shifted"
# No noise
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.0, noise_test=0.0, shift=2.0)
# 0.005 noise
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.005, noise_test=0.005, shift=2.0)
# 0.05 noise
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)
# rastrigin_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rastrigin_40d_shifted, noise_train=0.05, noise_test=0.05, shift=2.0)

# %% Rosenbrock ------------------------------------------------------------------------------------------------

# Rosenbrock 5xdim
save_path_rosenbrock_5d = f"{save_path_rosenbrock}/5xdim"
# No noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_rosenbrock_5d, noise_train=0.05, noise_test=0.05)

# Rosenbrock 20xdim
save_path_rosenbrock_20d = f"{save_path_rosenbrock}/20xdim"
# No noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_rosenbrock_20d, noise_train=0.05, noise_test=0.05)

# Rosenbrock 40xdim
save_path_rosenbrock_40d = f"{save_path_rosenbrock}/40xdim"
# No noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)
# rosenbrock_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=40, save_path=save_path_rosenbrock_40d, noise_train=0.05, noise_test=0.05)

# %% Keane Bump ------------------------------------------------------------------------------------------------

# # Keane Bump 5xdim
# save_path_keane_bump_5d = f"{save_path_keane_bump}/5xdim"
# # No noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.0, noise_test=0.0)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.0, noise_test=0.0)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.0, noise_test=0.0)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.005, noise_test=0.005)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.005, noise_test=0.005)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.005, noise_test=0.005)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.05, noise_test=0.05)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.05, noise_test=0.05)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.05, noise_test=0.05)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_keane_bump_5d, noise_train=0.05, noise_test=0.05)

# # Keane Bump 30xdim
# save_path_keane_bump_30d = f"{save_path_keane_bump}/30xdim"
# # No noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.0, noise_test=0.0)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.0, noise_test=0.0)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.0, noise_test=0.0)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.0, noise_test=0.0)
# # 0.005 noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.005, noise_test=0.005)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.005, noise_test=0.005)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.005, noise_test=0.005)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.005, noise_test=0.005)
# # 0.05 noise
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.05, noise_test=0.05)
# keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.05, noise_test=0.05)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.05, noise_test=0.05)
# # keane_bump_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=30, save_path=save_path_keane_bump_30d, noise_train=0.05, noise_test=0.05)


# Ackley 5xdim
save_path_ackley_5d = f"{save_path_ackley}/5xdim"
save_path_ackley_5d_V2 = f"{save_path_ackley_V2}/5xdim"
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_5d, noise_train=0.05, noise_test=0.05)
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.0, noise_test=0.0)
    # 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.005, noise_test=0.005)
# 0.05 noisev2=True, 
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=5, save_path=save_path_ackley_5d_V2, noise_train=0.05, noise_test=0.05)
# Ackley 10xdim
save_path_ackley_10d = f"{save_path_ackley}/10xdim"
save_path_ackley_10d_V2 = f"{save_path_ackley_V2}/10xdim"
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_10d, noise_train=0.05, noise_test=0.05)
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.0, noise_test=0.0)
# 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.005, noise_test=0.005)
# 0.05 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=10, save_path=save_path_ackley_10d_V2, noise_train=0.05, noise_test=0.05)

# Ackley 20xdim
save_path_ackley_20d = f"{save_path_ackley}/20xdim"
save_path_ackley_20d_V2 = f"{save_path_ackley_V2}/20xdim"
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.0, noise_test=0.0)
# 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.005, noise_test=0.005)
# 0.05 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_20d, noise_train=0.05, noise_test=0.05)
# No noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.0, noise_test=0.0)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.0, noise_test=0.0)
# 0.005 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.005, noise_test=0.005)
# 0.05 noise
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=10, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=20, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=40, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, v2=True, train_size=80, dimensions=20, save_path=save_path_ackley_20d_V2, noise_train=0.05, noise_test=0.05)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")
