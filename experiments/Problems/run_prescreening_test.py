# %% Import Problems ------------------------------------------------------------------------------------------------
import time
import json
from pathlib import Path

from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.training.parameter_initializer import DefaultParameterInitializer
from gpplus.training.optimizers import LBFGSScipy

initializer_class = DefaultParameterInitializer
num_epochs = 100
num_runs = 8
lr = 0.1
folder = "prescreening_test2"
start_time = time.time()
num_folds = 10

# Storage for metrics - organized by train_size
wing_metrics_with = {}  # {train_size: [metrics]}
wing_metrics_without = {}
wing_metrics_tabpfn = {}  # TabPFN metrics (same for with/without prescreening)
borehole_metrics_with = {}
borehole_metrics_without = {}
borehole_metrics_tabpfn = {}
rosenbrock_metrics_with = {}
rosenbrock_metrics_without = {}
rosenbrock_metrics_tabpfn = {}

# %% A1 Wing - WITH Prescreening ------------------------------------------------------------------------------------------------
save_path_wing_with = f"./{folder}/wing/with_prescreening"
for train_size in [3, 5, 10, 20, 30]:
    gp_metrics, tabpfn_metrics = wing_SF_GPvsPFN(num_folds=num_folds, train_size=train_size, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_wing_with, initializer_class=initializer_class, standardize_y=True, use_recorder=True)
    if train_size not in wing_metrics_with:
        wing_metrics_with[train_size] = []
    wing_metrics_with[train_size].extend(gp_metrics)
    # Collect TabPFN metrics (same for with/without prescreening)
    if train_size not in wing_metrics_tabpfn:
        wing_metrics_tabpfn[train_size] = []
    wing_metrics_tabpfn[train_size].extend(tabpfn_metrics)

# %% A1 Wing - WITHOUT Prescreening ------------------------------------------------------------------------------------------------
save_path_wing_without = f"./{folder}/wing/without_prescreening"
for train_size in [3, 5, 10, 20, 30]:
    gp_metrics, _ = wing_SF_GPvsPFN(num_folds=num_folds, train_size=train_size, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_wing_without, initializer_class=initializer_class, standardize_y=True, use_recorder=False)
    if train_size not in wing_metrics_without:
        wing_metrics_without[train_size] = []
    wing_metrics_without[train_size].extend(gp_metrics)

# # %% A3 Borehole - WITH Prescreening ------------------------------------------------------------------------------------------------
# save_path_borehole_with = f"./{folder}/borehole/with_prescreening"
# for train_size in [3, 5, 10, 20, 30]:
#     gp_metrics, tabpfn_metrics = borehole_SF_GPvsPFN(num_folds=num_folds, train_size=train_size, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_borehole_with, initializer_class=initializer_class, standardize_y=True, use_recorder=True)
#     if train_size not in borehole_metrics_with:
#         borehole_metrics_with[train_size] = []
#     borehole_metrics_with[train_size].extend(gp_metrics)
#     # Collect TabPFN metrics (same for with/without prescreening)
#     if train_size not in borehole_metrics_tabpfn:
#         borehole_metrics_tabpfn[train_size] = []
#     borehole_metrics_tabpfn[train_size].extend(tabpfn_metrics)

# # %% A3 Borehole - WITHOUT Prescreening ------------------------------------------------------------------------------------------------
# save_path_borehole_without = f"./{folder}/borehole/without_prescreening"
# for train_size in [3, 5, 10, 20, 30]:
#     gp_metrics, _ = borehole_SF_GPvsPFN(num_folds=num_folds, train_size=train_size, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_borehole_without, initializer_class=initializer_class, standardize_y=True, use_recorder=False)
#     if train_size not in borehole_metrics_without:
#         borehole_metrics_without[train_size] = []
#     borehole_metrics_without[train_size].extend(gp_metrics)

# # %% A6 Rosenbrock - WITH Prescreening ------------------------------------------------------------------------------------------------
# save_path_rosenbrock_with = f"./{folder}/rosenbrock/with_prescreening"
# for train_size in [10, 20]:
#     gp_metrics, tabpfn_metrics = rosenbrock_GPvsPFN(num_folds=num_folds, train_size=train_size, dimensions=40, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_rosenbrock_with, initializer_class=initializer_class, standardize_y=True, use_recorder=True)
#     if train_size not in rosenbrock_metrics_with:
#         rosenbrock_metrics_with[train_size] = []
#     rosenbrock_metrics_with[train_size].extend(gp_metrics)
#     # Collect TabPFN metrics (same for with/without prescreening)
#     if train_size not in rosenbrock_metrics_tabpfn:
#         rosenbrock_metrics_tabpfn[train_size] = []
#     rosenbrock_metrics_tabpfn[train_size].extend(tabpfn_metrics)

# # %% A6 Rosenbrock - WITHOUT Prescreening ------------------------------------------------------------------------------------------------
# save_path_rosenbrock_without = f"./{folder}/rosenbrock/without_prescreening"
# for train_size in [10, 20]:
#     gp_metrics, _ = rosenbrock_GPvsPFN(num_folds=num_folds, train_size=train_size, dimensions=40, num_test=5000, num_epochs=num_epochs, num_runs=num_runs, lr=lr, optimizer_class=LBFGSScipy, save_path=save_path_rosenbrock_without, initializer_class=initializer_class, standardize_y=True, use_recorder=False)
#     if train_size not in rosenbrock_metrics_without:
#         rosenbrock_metrics_without[train_size] = []
#     rosenbrock_metrics_without[train_size].extend(gp_metrics)

# %% Comparison Plots ------------------------------------------------------------------------------------------------
comparison_save_path = f"./{folder}/comparison_plots"
Path(comparison_save_path).mkdir(parents=True, exist_ok=True)

# A1 Wing Comparison - separate plot for each train_size
print("\n" + "="*80)
print("Creating A1 Wing Comparison Plots")
print("="*80)
for train_size in sorted(set(list(wing_metrics_with.keys()) + list(wing_metrics_without.keys()))):
    if train_size in wing_metrics_with and train_size in wing_metrics_without:
        metrics_with = wing_metrics_with[train_size]
        metrics_without = wing_metrics_without[train_size]
        metrics_tabpfn = wing_metrics_tabpfn.get(train_size, [])
        if metrics_with and metrics_without:
            print(f"\n  Train size {train_size}: {len(metrics_with)} with, {len(metrics_without)} without, {len(metrics_tabpfn)} TabPFN")
            try:
                plot_metrics(
                    metrics_with,
                    metrics_without,
                    metrics_tabpfn,
                    labels=["With Prescreening", "Without Prescreening", "TabPFN"],
                    title=f"A1_Wing_train_size_{train_size}",
                    save_path=comparison_save_path
                )
                print(f"    Plot created successfully")
            except Exception as e:
                print(f"    ERROR creating plot: {e}")
                import traceback
                traceback.print_exc()

# # A3 Borehole Comparison - separate plot for each train_size
# print("\n" + "="*80)
# print("Creating A3 Borehole Comparison Plots")
# print("="*80)
# for train_size in sorted(set(list(borehole_metrics_with.keys()) + list(borehole_metrics_without.keys()))):
#     if train_size in borehole_metrics_with and train_size in borehole_metrics_without:
#         metrics_with = borehole_metrics_with[train_size]
#         metrics_without = borehole_metrics_without[train_size]
#         metrics_tabpfn = borehole_metrics_tabpfn.get(train_size, [])
#         if metrics_with and metrics_without:
#             print(f"\n  Train size {train_size}: {len(metrics_with)} with, {len(metrics_without)} without, {len(metrics_tabpfn)} TabPFN")
#             try:
#                 plot_metrics(
#                     metrics_with,
#                     metrics_without,
#                     metrics_tabpfn,
#                     labels=["With Prescreening", "Without Prescreening", "TabPFN"],
#                     title=f"A3_Borehole_train_size_{train_size}",
#                     save_path=comparison_save_path
#                 )
#                 print(f"    Plot created successfully")
#             except Exception as e:
#                 print(f"    ERROR creating plot: {e}")
#                 import traceback
#                 traceback.print_exc()

# # A6 Rosenbrock Comparison - separate plot for each train_size
# print("\n" + "="*80)
# print("Creating A6 Rosenbrock Comparison Plots")
# print("="*80)
# for train_size in sorted(set(list(rosenbrock_metrics_with.keys()) + list(rosenbrock_metrics_without.keys()))):
#     if train_size in rosenbrock_metrics_with and train_size in rosenbrock_metrics_without:
#         metrics_with = rosenbrock_metrics_with[train_size]
#         metrics_without = rosenbrock_metrics_without[train_size]
#         metrics_tabpfn = rosenbrock_metrics_tabpfn.get(train_size, [])
#         if metrics_with and metrics_without:
#             print(f"\n  Train size {train_size}: {len(metrics_with)} with, {len(metrics_without)} without, {len(metrics_tabpfn)} TabPFN")
#             try:
#                 plot_metrics(
#                     metrics_with,
#                     metrics_without,
#                     metrics_tabpfn,
#                     labels=["With Prescreening", "Without Prescreening", "TabPFN"],
#                     title=f"A6_Rosenbrock_train_size_{train_size}",
#                     save_path=comparison_save_path
#                 )
#                 print(f"    Plot created successfully")
#             except Exception as e:
#                 print(f"    ERROR creating plot: {e}")
#                 import traceback
#                 traceback.print_exc()

end_time = time.time()
print(f"\nTime taken: {(end_time - start_time)/3600:0.2f} hours [{(end_time - start_time)/60:0.1f} minutes]")
print(f"Comparison plots saved to: {comparison_save_path}")
# %%

