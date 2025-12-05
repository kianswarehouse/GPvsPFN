from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN

num_folds = 20
num_runs = 16
title = None

folder = "resultsHPC3"
date = "12_04"
save_path_wing = f"./{folder}/wing/{date}"
save_path_borehole = f"./{folder}/borehole/{date}"
save_path_buckling = f"./{folder}/buckling/{date}"

# # %% Wing ------------------------------------------------------------------------------------------------
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# %% Borehole ------------------------------------------------------------------------------------------------
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(title=title, num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)
