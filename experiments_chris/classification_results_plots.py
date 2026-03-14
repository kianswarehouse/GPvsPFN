import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

summary_path = "results/run_20260310_143608/grid_rbf_matern05_powerexp_tabpfn_multistart_stable_summary.csv"  # update if needed
df = pd.read_csv(summary_path)

df = df.sort_values(["model", "train_size"])


# ------------------------------------------------------------
# MODEL NAME MAPPING
# ------------------------------------------------------------

pretty_name = {
    "GP+ GPClassifier (RBF, ARD, LBFGS-defaults)": "GP+",
    "GP+ GPClassifier (PowerExp, p=learned, LBFGS-defaults)": "GP+ (PE)",
    "GP+ GPClassifier (Matern, nu=0.5, ARD, LBFGS-defaults)": "GP+ (Matern)",
    "TabPFN Classifier (v2.5 default)": "PFN 2.5",
    "TabPFN Classifier (v2 compatibility)": "PFN 2.0",
}

df["pretty"] = df["model"].map(pretty_name)

# models to show
gp_models = ["GP+", "GP+ (PE)", "GP+ (Matern)"]
pfn_models = ["PFN 2.5", "PFN 2.0"]

all_models = gp_models + pfn_models


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

markers = {
    "GP+": "o",
    "GP+ (PE)": "s",
    "GP+ (Matern)": "^",
    "PFN 2.5": "D",
    "PFN 2.0": "v",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ("final_train_nll_mean", "Train Negative Log Likelihood"),
    ("accuracy_mean", "Test Accuracy"),
    ("brier_mean", "Test Brier Score"),
]

for ax, (metric, title) in zip(axes, metrics):

    if metric == "final_train_nll_mean":
        models = gp_models
    else:
        models = all_models

    for model in models:

        sub = df[df["pretty"] == model].sort_values("train_size")

        ax.plot(
            sub["train_size"],
            sub[metric],
            linestyle=":",
            marker=markers[model],
            markersize=8,
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Train Size")
    ax.set_title(title)
    
    train_sizes = sorted(df["train_size"].unique())
    ax.set_xticks(train_sizes)
    ax.set
    
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Train NLL")
axes[1].set_ylabel("Accuracy")
axes[2].set_ylabel("Brier Score")


# ------------------------------------------------------------
# LEGEND (BOTTOM RIGHT OF MIDDLE PLOT)
# ------------------------------------------------------------

axes[1].legend(
    loc="lower right",
    frameon=True,
)


plt.tight_layout()

plt.savefig(
    "results/paper_performance_figure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()