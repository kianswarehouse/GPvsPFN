from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

num_folds = 20
num_runs = 16
folder = "resultsHPC3"
date = "12_04"

# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./{folder}/wing/{date}"
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing)

wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)

wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)

# # %% Buckling ------------------------------------------------------------------------------------------------
save_path_buckling = f"./{folder}/buckling/{date}"
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling)

buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)

buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = f"./{folder}/borehole/{date}"
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole)

borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)

borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_SF_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)

# # %% Ackley ------------------------------------------------------------------------------------------------

save_path_ackley = f"./{folder}/ackley/{date}/5D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley)

save_path_ackley = f"./{folder}/ackley/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley)

save_path_ackley = f"./{folder}/ackley/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True)


# # %% Ackley with noise ------------------------------------------------------------------------------------------------
save_path_ackley = f"./{folder}/ackley/{date}/5D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

save_path_ackley = f"./{folder}/ackley/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

save_path_ackley = f"./{folder}/ackley/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley, noise_train=0.005, noise_test=0.005)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/5D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=5, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/10D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=10, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

save_path_ackley_V2 = f"./{folder}/ackleyV2/{date}/20D"
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.05, noise_test=0.05)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)
ackley_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, dimensions=20, save_path=save_path_ackley_V2, V2=True, noise_train=0.005, noise_test=0.005)

# # %% Wing ------------------------------------------------------------------------------------------------
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 5, 3, 2], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10, 6, 4], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 25, 30, 35], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 20], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 40], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[40, 80], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling)

# %% Borehole ------------------------------------------------------------------------------------------------
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 15, 20, 25, 30], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 8, 6, 4, 2], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[10, 4, 3, 2, 1], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=[20, 30, 40, 50, 60], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole)
