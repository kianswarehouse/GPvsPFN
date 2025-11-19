# Comparison Summary: 11_17 vs 11_172

## Key Differences Found

### 1. **Number of Runs/Seeds**
- **11_17**: Uses **5 seeds/runs** (count: 5)
- **11_172**: Uses **20 seeds/runs** (count: 20)

This is the most significant difference - 11_172 has 4x more runs, providing more statistical power.

---

## WING Results

### Files Present in Both Folders:
1. `gpVpfn_wing_SF_10D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
2. `gpVpfn_wing_SF_10D_10000epochs_4runs_0.1_noiseTest0.05_noiseTrain0.05.json`
3. `gpVpfn_wing_SF_80D_10000epochs_4runs_0.1_noiseTest0.05_noiseTrain0.05.json`
4. Multi-fidelity configurations (with different naming)

### Files Only in 11_17:
- `gpVpfn_wingMF_[5, 5, 5, 5]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
- `gpVpfn_wingMF_[5, 10, 10, 10]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`

### Files Only in 11_172:
- None (but note the "Test" prefix in wingMF filenames)

### Naming Difference:
- **11_17**: `gpVpfn_wingMF_[...]`
- **11_172**: `gpVpfn_TestwingMF_[...]` (has "Test" prefix)

### Example Metric Comparison (wing_SF_10D, no noise):
- **11_17** (5 runs):
  - GP RRMSE mean: 0.017357, count: 5
  - GP NIS mean: 0.128890, count: 5
  
- **11_172** (20 runs):
  - GP RRMSE mean: 0.016007, count: 20
  - GP NIS mean: 0.097806, count: 20

---

## BUCKLING Results

### Files Present in Both Folders (Default):
1. `gpVpfn_buckling_[5, 5]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
2. `gpVpfn_buckling_[10, 10]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`

### Files Present in Both Folders (Noise):
1. `gpVpfn_buckling_[5, 5]D_10000epochs_4runs_0.1_noiseTest0.005_noiseTrain0.005.json`
2. `gpVpfn_buckling_[5, 5]D_10000epochs_4runs_0.1_noiseTest0.05_noiseTrain0.05.json`
3. `gpVpfn_buckling_[10, 10]D_10000epochs_4runs_0.1_noiseTest0.005_noiseTrain0.005.json`
4. `gpVpfn_buckling_[10, 10]D_10000epochs_4runs_0.1_noiseTest0.05_noiseTrain0.05.json`

### Files Only in 11_17:
- None (all files match)

### Files Only in 11_172:
- None (all files match)

### Example Metric Comparison (buckling_[5,5], no noise):
- **11_17** (5 runs):
  - GP RRMSE mean: 0.313350, count: 5
  - GP NIS mean: 3.166558, count: 5
  
- **11_172** (20 runs):
  - GP RRMSE mean: 0.330555, count: 20
  - GP NIS mean: 3.914982, count: 20

### Example Metric Comparison (buckling_[5,5], noise 0.05):
- **11_17** (5 runs):
  - GP RRMSE mean: 0.366974, count: 5
  - GP NIS mean: 4.269845, count: 5
  
- **11_172** (20 runs):
  - GP RRMSE mean: 0.459070, count: 20
  - GP NIS mean: 6.343215, count: 20

---

## BOREHOLE Results

### Files Present in Both Folders:
1. `gpVpfn_borehole_[1, 2, 3, 4, 5]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
2. `gpVpfn_borehole_[10, 1, 1, 1, 1]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
3. `gpVpfn_borehole_[10, 10, 10, 10, 10]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
4. `gpVpfn_borehole_[5, 5, 5, 5, 5]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`

### Files Only in 11_17:
- `gpVpfn_borehole_[2, 2, 2, 2, 2]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`
- `gpVpfn_borehole_[5, 10, 15, 20, 25]D_10000epochs_4runs_0.1_noiseTest0.0_noiseTrain0.0.json`

### Files Only in 11_172:
- None

### Example Metric Comparison (borehole_[5,5,5,5,5], no noise):
- **11_17** (5 runs):
  - GP RRMSE mean: 0.017014, count: 5
  - GP NIS mean: 0.089192, count: 5
  
- **11_172** (20 runs):
  - GP RRMSE mean: 0.019722, count: 20
  - GP NIS mean: 0.106907, count: 20

---

## Summary of Changes

### Major Changes:
1. **Increased number of seeds**: Changed from 5 to 20 runs (4x increase)
2. **Missing experiments in 11_172**:
   - Wing: Missing `[5, 5, 5, 5]` and `[5, 10, 10, 10]` configurations
   - Borehole: Missing `[2, 2, 2, 2, 2]` and `[5, 10, 15, 20, 25]` configurations
3. **Naming change**: Wing multi-fidelity files have "Test" prefix in 11_172

### Statistical Impact:
- **11_172** has more statistical power with 20 runs vs 5 runs
- Metrics show some variation between the two sets, which is expected with different sample sizes
- Standard deviations are generally lower in 11_172 due to larger sample size

### Recommendations:
- Use **11_172** for more reliable statistical results (20 runs)
- Note that some experimental configurations from 11_17 are not present in 11_172
- The "Test" prefix in wing files suggests 11_172 might be a test run before finalizing

