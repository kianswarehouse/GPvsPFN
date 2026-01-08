# %% Import SEEK Kernel Problems ------------------------------------------------------------------------------------------------
import time

from A1_wing_SF_GPvsPFN_SEEK import wing_SF_GPvsPFN as wing_GPvsPFN
# TODO: Add other SEEK versions when available:
# from A2_buckling_SF_GPvsPFN_seek import buckling_SF_GPvsPFN as buckling_GPvsPFN
# from A3_borehole_SF_GPvsPFN_seek import borehole_SF_GPvsPFN as borehole_GPvsPFN
# etc.

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics

date = "12_18"
# Note: Uses GPPlus defaults from defaults.py (LBFGS optimizer, etc.)

num_folds = 20
num_runs = 16
folder = "seek_results"
start_time = time.time()

# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = f"./{folder}/wing/{date}"
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.0, noise_test=0.0)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.0, noise_test=0.0)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.0, noise_test=0.0)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.0, noise_test=0.0)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.005, noise_test=0.005)

wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)
wing_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=80, save_path=save_path_wing, noise_train=0.05, noise_test=0.05)


# %% Buckling ------------------------------------------------------------------------------------------------
# TODO: Add buckling SEEK version
# save_path_buckling = f"./{folder}/buckling/{date}"
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_buckling)
# buckling_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_buckling)

# %% Borehole ------------------------------------------------------------------------------------------------
# TODO: Add borehole SEEK version
# save_path_borehole = f"./{folder}/borehole/{date}"
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=10, save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=20, save_path=save_path_borehole)
# borehole_GPvsPFN(num_folds=num_folds, num_runs=num_runs, train_size=40, save_path=save_path_borehole)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")
# %%
