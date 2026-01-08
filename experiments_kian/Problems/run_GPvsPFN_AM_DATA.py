# from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
# from A1_wing_MF_GPvsPFN import wing_GPvsPFN
# from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
# from A2_buckling_MF_GPvsPFN import buckling_GPvsPFN
# from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
# from A3_borehole_MF_GPvsPFN import borehole_GPvsPFN
# from A4_ackley_GPvsPFN import ackley_GPvsPFN
# from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
# from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
# from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
# from A8_griewank_GPvsPFN import griewank_GPvsPFN
# from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN
# from A10_styblinski_tang_GPvsPFN import styblinski_tang_GPvsPFN

from D1_1_am_data_porosity_GPvsPFN import am_data_porosity_GPvsPFN
from D1_2_am_data_hardness_GPvsPFN import am_data_hardness_GPvsPFN


num_folds = 20
num_runs = 16
folder = "results"
date = "1_07"

# save_path_rosenbrock = f"./{folder}/rosenbrock/{date}"
# save_path_rastrigin = f"./{folder}/rastrigin/{date}"
# save_path_zakharov = f"./{folder}/zakharov/{date}"
# save_path_griewank = f"./{folder}/griewank/{date}"
# save_path_dixon_price = f"./{folder}/dixon_price/{date}"
# save_path_styblinski = f"./{folder}/styblinski/{date}"
save_path_am_porosity = f"./{folder}/am_data_porosity/{date}"
save_path_am_hardness = f"./{folder}/am_data_hardness/{date}"


# %% AM Data Porosity (D1.1) ------------------------------------------------------------------------------------------------
am_data_porosity_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_porosity)

am_data_porosity_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_porosity)

# %% AM Data Hardness (D1.2) ------------------------------------------------------------------------------------------------
am_data_hardness_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0", num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_hardness)

am_data_hardness_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd0_SFkernel", encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_hardness)

# %% AM Data Porosity (D1.1) ------------------------------------------------------------------------------------------------
am_data_porosity_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_porosity)

am_data_porosity_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_porosity)
am_data_porosity_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_porosity)

# # %% AM Data Hardness (D1.2) ------------------------------------------------------------------------------------------------
am_data_hardness_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2", x_standardize_method=2, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_hardness)

am_data_hardness_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.8, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.6, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.4, save_path=save_path_am_hardness)
am_data_hardness_GPvsPFN(title="mfxstd2_SFkernel", x_standardize_method=2, encode_data=False, num_folds=num_folds, num_runs=num_runs, test_size=0.2, save_path=save_path_am_hardness)
