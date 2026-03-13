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
date = "custom_GPvsPFN"
# date = "test"

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

# save_path_buckling = save_path_buckling + "/buckling"

num_test = 5000
run_models = None
run_models = 'gp'
# run_models = 'pfn'
# title = "2.5"
# title = None
title = "grouped_Am_U[-3,-2]noise"
# title = "grouped_NN_U[-4,-3]noise"
# title = "grouped_NN"

# from gpplus.training.parameter_initializer import DefaultParameterInitializer
# parameter_init = gpplus.ParameterInit(scale_noise=-3.5, scale_lengthscale=1)

num_runs = 20
num_runs = 16
wing_SF_GPvsPFN(title="warmup", num_runs=1, train_size=10, save_path=None, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# wing_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_wing, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# borehole_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_borehole, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="2", train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="GP_only", train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, run_pfn=False, num_test=num_test, run_models=run_models)

buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="GP_only", train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, run_pfn=False, num_test=num_test, run_models=run_models)


# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, MF_kernel=False, train_size=10, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=20, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, run_pfn=False, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=40, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, run_pfn=False, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=80, save_path=save_path_buckling, noise_train=0.005, noise_test=0.005, run_pfn=False, num_test=num_test, run_models=run_models)

# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, MF_kernel=False, train_size=10, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=20, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, run_pfn=False, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=40, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, run_pfn=False, num_test=num_test, run_models=run_models)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, title="SF", MF_kernel=False, train_size=80, save_path=save_path_buckling, noise_train=0.05, noise_test=0.05, run_pfn=False, num_test=num_test, run_models=run_models)
