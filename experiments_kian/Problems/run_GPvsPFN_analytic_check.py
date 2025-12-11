import time
from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A1_wing_MF_GPvsPFN import wing_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN

# %% Wing ------------------------------------------------------------------------------------------------
date = "temp0"
save_path_wing = f"./results/wing/{date}"

epochs = 1
num_folds = 1
runs = 2
start_time = time.time()
wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

wing_GPvsPFN(train_size=[10, 10, 10, 10], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_GPvsPFN(train_size=[10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_GPvsPFN(train_size=[10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

wing_GPvsPFN(train_size=[20, 20, 20, 20], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_GPvsPFN(train_size=[20, 20, 20, 20], noise_train=[0.005, 0.01, 0.025, 0.05], noise_test=[0.005, 0.01, 0.025, 0.05], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
wing_GPvsPFN(train_size=[20, 20, 20, 20], noise_train=[0.05, 0.1, 0.15, 0.25], noise_test=[0.05, 0.1, 0.15, 0.25], save_path=save_path_wing, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

# %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = f"./results/buckling/{date}"
buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_GPvsPFN(train_size=[10, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_GPvsPFN(train_size=[10, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

buckling_GPvsPFN(train_size=[10, 10], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_GPvsPFN(train_size=[10, 10], noise_train=[0.005, 0.01], noise_test=[0.005, 0.01], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
buckling_GPvsPFN(train_size=[10, 10], noise_train=[0.05, 0.1], noise_test=[0.05, 0.1], save_path=save_path_buckling, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = f"./results/borehole/{date}"
borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], save_path=save_path_borehole, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], noise_train=[0.005, 0.01, 0.025, 0.05, 0.1], noise_test=[0.005, 0.01, 0.025, 0.05, 0.1], save_path=save_path_borehole, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
borehole_GPvsPFN(train_size=[10, 10, 10, 10, 10], noise_train=[0.05, 0.1, 0.15, 0.25, 0.35], noise_test=[0.05, 0.1, 0.15, 0.25, 0.35], save_path=save_path_borehole, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

# %% Ackley ------------------------------------------------------------------------------------------------
save_path_ackley = f"./results/ackley/{date}"
ackley_GPvsPFN(dimensions=5, train_size=10, save_path=save_path_ackley, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
ackley_GPvsPFN(dimensions=5, train_size=10, save_path=save_path_ackley, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

ackley_GPvsPFN(dimensions=20, train_size=10, save_path=save_path_ackley, num_epochs=epochs, num_folds=num_folds, num_runs=runs)
ackley_GPvsPFN(dimensions=20, train_size=10, save_path=save_path_ackley, num_epochs=epochs, num_folds=num_folds, num_runs=runs)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")