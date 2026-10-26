# Scientific Image Forgery Detection: a learning report

## Chapter 0: Foundation

**Goal:** build a trustworthy way to measure any forgery detector before building one. That
means a fixed validation set, a faithful copy of the competition metric, and reference points
that every later score can be placed against.

### The data

From [`notebooks/00_explore_data.ipynb`](../notebooks/00_explore_data.ipynb):

- **5,176 images:** 2,377 authentic training images, 2,751 forged training images, and 48
  supplemental forged images.
- **Each forged image has a mask:** one 0/1 layer per forged region. Most forged images have 1
  region (1,902), many have 2 (710), a few have 3 or more, and one has 13.
- **Forgeries are small.** Measured over all 2,799 forged images (training plus supplemental),
  the median is 2.8% of the image (mean 5.6%); 90% of forged images are no more than 14.4%
  forged.
- **Twins:** every authentic image has a forged twin with the same case ID. They differ in only
  a small fraction of pixels (0.43%–3.27% in the five pairs shown).
- **The mask covers more than the edit.** Only the pasted copy's pixels change, but the mask
  marks both the copy and the source region it came from. The source carries no trace of
  editing; it can only be found by matching it to its copy.

### The metric

Kaggle's official metric (`metric/recodai-f1`) is ported to
[`src/forgery/metric.py`](../src/forgery/metric.py) and checked against Kaggle's own examples.
Each image scores between 0 and 1, and the final score is the average:

1. **Authentic or forged?** If the truth or the prediction is "authentic", the image scores 1
   when both agree and 0 otherwise. No pixels are counted.
2. **Both say forged:** every predicted mask is scored against every true mask with pixel
   F1 = 2·TP / (2·TP + FP + FN). The best one-to-one pairing is chosen. The paired F1s are
   averaged, then multiplied by n_true ÷ max(n_predicted, n_true), which penalises extra masks.

Scores for different predictions on image 568, which has two true regions of 6,243 and 185
pixels:

| Prediction | Score |
|---|---|
| Both regions exactly right | 1.000 |
| Big region drawn too large, small region right | 0.905 |
| Both right, plus one extra junk mask | 0.667 |
| Only the big region (the small one missed) | 0.500 |
| Both regions merged into one mask | 0.493 |

Every region counts equally whatever its size, extra masks cost points, and a wrong
authentic/forged call scores 0 however good the mask is.

### The validation split

20% of the training cases, chosen with a fixed seed and stored in `splits/val_case_ids.txt`:
550 cases (475 twin pairs and 75 forged-only), which is 1,025 images (475 authentic, 550 forged).

The split is by case, not by file. If an authentic image sat in training while its
near-identical forged twin sat in validation, a model could score well just by memorising the
originals. That kind of leakage makes validation scores look better than real performance. The
split is also stratified, so twin pairs and forged-only cases appear in the same proportions on
both sides.

### Baselines

#### Predictions (recorded 2026-10-26, before any baseline was run)

Validation set: 1,025 images (475 authentic, 550 forged). Estimates were worked out by reasoning
about the metric, then accepted by the user as their predictions.

| Baseline | Predicted val score | Reasoning |
|---|---|---|
| All authentic | 0.46 | 475 authentic images score 1, 550 forged score 0 → 475 / 1025 |
| Perfect detection, whole-image mask | ~0.50 | Authentic images score 1; a forged region covering ~3% of the image gives F1 ≈ 2 × 0.03 ≈ 0.06 → (475 + 550 × 0.06) / 1025 |
| Perfect detection, ground-truth mask | 1.0 | Every image is perfect |

#### Results (commit `1caa122`, from `experiments/results.csv`)

| Baseline | Predicted (val) | Actual (val) | Actual (supplemental) |
|---|---|---|---|
| All authentic | 0.46 | **0.4634** | 0.0000 |
| Perfect detection, whole-image mask | ~0.50 | **0.5032** | 0.1305 |
| Perfect detection, ground-truth mask | 1.0 | **1.0000** | 1.0000 |

All three predictions were right to within 0.004. The one estimate inside them was off: we
guessed a whole-image mask would earn about 0.06 per forged image, and it earned **0.0742**
(`mean_of1_detected`). Forged area is skewed: the median forged image in val is only 2.2% forged,
but a minority of large forgeries pull the val mean up to 5.1%, and those earn much more from a
whole-image mask.

On the supplemental set the whole-image mask earns 0.13 per forged image. Supplemental
forgeries are about three times larger (median area 6.7% vs 2.2% in val), so painting the
whole image overlaps them more. That's despite them having more regions on average (1.79 vs
1.38), which the metric penalises. The supplemental set only has forged images, so calling
everything authentic scores 0.

**Takeaways**
- **Floor: 0.46.** Any model has to beat calling everything authentic.
- **Perfect detection alone is worth only +0.04** (0.46 → 0.50). Almost all the remaining
  room, 0.50 → 1.0, comes from drawing accurate masks.
- **The ground-truth oracle scores exactly 1.0 on both sets**, so data loading, mask handling
  and scoring work end to end.

### Limitation

Our training and validation forgeries appear to be synthetic: made by copying regions within
an image. Kaggle's hidden test set consisted of real cases from published papers. A validation
score here measures progress on this synthetic distribution, not real-world accuracy, so later
chapters should treat big jumps with some suspicion and check them on the supplemental set too.
