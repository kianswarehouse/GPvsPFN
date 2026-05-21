# *** SCRIPTS STILL BEING FINALIZED ***

# Gaussian Processes vs Tabular Prior Fitted Network
Public repository containing scripts for validating experiments and additional results for the paper "On the Uncertainty Quantification Ability of Tabular Foundation Models" - part of the IEEE Computing in Science and Engineering Special Issue on "Controversies on the Usage of AI/ML for Science and Engineering".

# What is currently in this repo: 
- Current snapshot of our GP+ library (5-17-2026).
- Scripts to generate data and run comparison studies between GP+ and TabPFN v2.5.
- A requirements.txt file containing a list of necessary package and their versions to run these experiments.
- Detailed results from comparison studies (including additonal experiments conducted on real-world datasets and their results).

# Setup:
1. Clone repository to a new environment: https://github.com/kianswarehouse/GPvsPFN.git
2. Install compatible version of pytorch: https://pytorch.org/
3. Navigate to the repo's foler: cd GPvsPFN
4. Install all required packages: pip install -r requirements.txt
5. Install GP+ package: pip install .


# Running experiments:
- Run comparisons on specific problems individually in the 'A#' scripts within the 'experiments' folder.
- Run all comparison problems using 'B#' scripts.
   - The setups in the problems can be easily changed in the function calls for each problem (train_size, num_test, dimensions, etc.).

- Defaults for all problems are in the 'defaults.py' file.
   - Detailed trainer analysis for GP+ can be enabled by setting 'TRAINER_INFO = TRUE'
   - GPvsPFN RRMSE and NIS result plotting can be enabled with 'PLOT_METRICS = TRUE'


# TabPFN usage:
Use of PriorLab's TabPFN v2.5 model requires token authentification through HuggingFace website. Instructions will automatically be provided in the console's output. The instructions will advise you to sign in and create a token (https://huggingface.co/settings/tokens). In the token's permissions, under 'Repositories', check the box that says "Read access to contents of all public gated repos you can access". Then save the token, then copy and paste it into the console when prompted.
