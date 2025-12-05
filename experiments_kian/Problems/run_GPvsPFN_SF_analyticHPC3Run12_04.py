from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN as wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN as buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN as borehole_GPvsPFN
from A4_Ackley_GPvsPFN import ackley_GPvsPFN

num_folds = 20
num_runs = 16
folder = "resultsHPC3"
date = "12_04"

# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./{folder}/wing/{date}/default"
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)

# # %% Buckling ------------------------------------------------------------------------------------------------
save_path_buckling = f"./{folder}/buckling/{date}"
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05)

buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)
buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005)

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = f"./{folder}/borehole/{date}"
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05)

borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)
borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005)

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