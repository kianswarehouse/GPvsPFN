from A1_wing_SF_GPvsPFN import wing_GPvsPFN as wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_GPvsPFN as buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_GPvsPFN as borehole_SF_GPvsPFN

from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
# %% Wing ------------------------------------------------------------------------------------------------
save_path_wing = "./results/wing/10_28/default"

epochs = 5
seeds = 3
runs = 4

gp_metrics_10, tabpfn_metrics_10 = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20, tabpfn_metrics_20 = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40, tabpfn_metrics_40 = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80, tabpfn_metrics_80 = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="wing_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="wing_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="wing_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="wing_20D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="wing_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="wing_40D")
analyze_metrics(gp_metrics_80, print_summary=True, label="GP", title="wing_80D")
analyze_metrics(tabpfn_metrics_80, print_summary=True, label="TabPFN", title="wing_80D")

save_path_wing_noise = "./results/wing/10_28/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs  )
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)


analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)  
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

gp_metrics_10_noise, tabpfn_metrics_10_noise = wing_SF_GPvsPFN(train_size=10, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = wing_SF_GPvsPFN(train_size=20, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = wing_SF_GPvsPFN(train_size=40, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = wing_SF_GPvsPFN(train_size=80, save_path=save_path_wing_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)


analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="wing_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="wing_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="wing_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="wing_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="wing_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="wing_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="wing_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="wing_80D_noise") 

# %% Buckling ------------------------------------------------------------------------------------------------

save_path_buckling = "./results/buckling/10_28/default"
gp_metrics_10, tabpfn_metrics_10 = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20, tabpfn_metrics_20 = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40, tabpfn_metrics_40 = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80, tabpfn_metrics_80 = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="buckling_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="buckling_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="buckling_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="buckling_20D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="buckling_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="buckling_40D")
analyze_metrics(gp_metrics_80, print_summary=True, label="GP", title="buckling_80D")
analyze_metrics(tabpfn_metrics_80, print_summary=True, label="TabPFN", title="buckling_80D")

save_path_buckling_noise = "./results/buckling/10_28/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = buckling_SF_GPvsPFN(train_size=10, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = buckling_SF_GPvsPFN(train_size=20, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = buckling_SF_GPvsPFN(train_size=40, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = buckling_SF_GPvsPFN(train_size=80, save_path=save_path_buckling_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="buckling_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="buckling_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="buckling_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="buckling_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="buckling_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="buckling_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="buckling_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="buckling_80D_noise")

# %% Borehole ------------------------------------------------------------------------------------------------

save_path_borehole = "./results/borehole/10_28/default"
gp_metrics_10, tabpfn_metrics_10 = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20, tabpfn_metrics_20 = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40, tabpfn_metrics_40 = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80, tabpfn_metrics_80 = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10, print_summary=True, label="GP", title="borehole_10D")
analyze_metrics(tabpfn_metrics_10, print_summary=True, label="TabPFN", title="borehole_10D")
analyze_metrics(gp_metrics_20, print_summary=True, label="GP", title="borehole_20D")
analyze_metrics(tabpfn_metrics_20, print_summary=True, label="TabPFN", title="borehole_20D")
analyze_metrics(gp_metrics_40, print_summary=True, label="GP", title="borehole_40D")
analyze_metrics(tabpfn_metrics_40, print_summary=True, label="TabPFN", title="borehole_40D")
analyze_metrics(gp_metrics_80, print_summary=True, label="GP", title="borehole_80D")
analyze_metrics(tabpfn_metrics_80, print_summary=True, label="TabPFN", title="borehole_80D")

save_path_borehole_noise = "./results/borehole/10_28/noise"
gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs) 
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.05, noise_test=0.05, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs) 
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.0, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.0, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")

gp_metrics_10_noise, tabpfn_metrics_10_noise = borehole_SF_GPvsPFN(train_size=10, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_20_noise, tabpfn_metrics_20_noise = borehole_SF_GPvsPFN(train_size=20, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_40_noise, tabpfn_metrics_40_noise = borehole_SF_GPvsPFN(train_size=40, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)
gp_metrics_80_noise, tabpfn_metrics_80_noise = borehole_SF_GPvsPFN(train_size=80, save_path=save_path_borehole_noise, noise_train=0.005, noise_test=0.005, num_epochs=epochs, num_seeds=seeds, num_runs=runs)

analyze_metrics(gp_metrics_10_noise, print_summary=True, label="GP", title="borehole_10D_noise")
analyze_metrics(tabpfn_metrics_10_noise, print_summary=True, label="TabPFN", title="borehole_10D_noise")
analyze_metrics(gp_metrics_20_noise, print_summary=True, label="GP", title="borehole_20D_noise")
analyze_metrics(tabpfn_metrics_20_noise, print_summary=True, label="TabPFN", title="borehole_20D_noise")
analyze_metrics(gp_metrics_40_noise, print_summary=True, label="GP", title="borehole_40D_noise")
analyze_metrics(tabpfn_metrics_40_noise, print_summary=True, label="TabPFN", title="borehole_40D_noise")
analyze_metrics(gp_metrics_80_noise, print_summary=True, label="GP", title="borehole_80D_noise")
analyze_metrics(tabpfn_metrics_80_noise, print_summary=True, label="TabPFN", title="borehole_80D_noise")