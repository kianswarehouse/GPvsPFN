from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A8_griewankCV_GPvsPFN import griewankCV_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN

folder = "results_final"
date = "griewank_fix"
# date = "test"

save_path_wing = f"./{folder}/{date}/wing/"
save_path_buckling = f"./{folder}/{date}/buckling/"
save_path_borehole = f"./{folder}/{date}/borehole/"
save_path_ackley = f"./{folder}/{date}/ackley/"
save_path_ackley_V2 = f"./{folder}/{date}/ackleyV2/"
save_path_rosenbrock = f"./{folder}/{date}/rosenbrock/"
save_path_rastrigin = f"./{folder}/{date}/rastrigin/"
save_path_zakharov = f"./{folder}/{date}/zakharov/"
save_path_dixon_price = f"./{folder}/{date}/dixon_price/"



save_path_griewank = f"./{folder}/{date}/griewank/"
# save_path_griewank = None

num_test = 5000
# run_models = None
run_models = 'gp'
# run_models = 'pfn'
# title = "no_retrain3"
# title = None


v = 1
# title = f"check{v}_1e-12j"
num_folds = 1
wing_SF_GPvsPFN(title="warmup", num_folds=2, num_epochs=2, train_size=10, num_runs=16, save_path=None, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# wing_SF_GPvsPFN(title="test", num_folds=2, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title="test", num_folds=2, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# borehole_SF_GPvsPFN(title="test", num_folds=2, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title="3", num_folds=num_folds, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title="3", num_folds=num_folds, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)


title = f"check{v}_1e-6j"
griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, num_runs=1, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, num_runs=16, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
title = f"check{v}_1e-12j"
griewank_GPvsPFN(title=title, cholesky_jitter=1e-12, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, cholesky_jitter=1e-12, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=f"check{v}_2_1e-12j", cholesky_jitter=1e-12, num_folds=num_folds, num_runs=1, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=f"check{v}_2_1e-12j", cholesky_jitter=1e-12, num_folds=num_folds, num_runs=16, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)


num_folds = 2
title = f"check{v}_1e-6j"
griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
title = f"check{v}_1e-12j"
griewank_GPvsPFN(title=title, cholesky_jitter=1e-12, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
griewank_GPvsPFN(title=title, cholesky_jitter=1e-12, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# %% Griewank -------------------------------------------------------------
# griewankCV_GPvsPFN(title=title, num_folds=num_folds, CV_folds=4, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewankCV_GPvsPFN(title="3", num_folds=num_folds, CV_folds=4, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, phase2_retrain=True, run_models=run_models)
# griewankCV_GPvsPFN(title=title, num_folds=num_folds, CV_folds=4, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewankCV_GPvsPFN(title="3", num_folds=num_folds, CV_folds=4, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, phase2_retrain=True, run_models=run_models)
# griewankCV_GPvsPFN(title=title, num_folds=num_folds, CV_folds=1, train_size=40, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewankCV_GPvsPFN(title=title, num_folds=num_folds, CV_folds=2, train_size=40, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)


# # %% Griewank ------------------------------------------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=5, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=5, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=5, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=5, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=5, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=5, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=5, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=5, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title='16inits', num_folds=1, train_size=40, dimensions=5, save_path=None, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title='2inits', num_folds=1, num_runs=2, train_size=40, dimensions=5, save_path=None, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=10, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=10, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=10, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=10, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=10, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=10, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=10, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=10, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models) # Where to restart 1_11 results


# # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=80, dimensions=20, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)



# # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, cholesky_jitter=1e-12, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, cholesky_jitter=1e-6, num_folds=num_folds, train_size=40, dimensions=40, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

# # # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=80, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=10, dimensions=80, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# griewank_GPvsPFN(title=title, num_folds=num_folds, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)

