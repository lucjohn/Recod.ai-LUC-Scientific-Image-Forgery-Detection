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

## Chapter 1: Classical baseline (SIFT + RANSAC)

**Goal:** the first real detector, with no learning. It finds copy-moved regions by matching an
image's SIFT keypoints against themselves, then grouping the matches by a shared movement with
RANSAC. Tuned on a slice of the training data, scored once on validation.

### Predictions (recorded 2026-10-27, before the validation run)

At prediction time, the user had seen the tuning results (score 0.574 on the 1,027-image tuning
slice: 98.9% of authentic images left alone, 27.8% of forged images detected, mask score 0.777
on those) and the Chapter 0 baselines.

| Run | Predicted val score |
|---|---|
| Tuned detector (`01_sift`) | 0.70 |
| Single RANSAC (`01_sift_single_ransac`) | not predicted |
| No RANSAC (`01_sift_no_ransac`) | not predicted |

### How the detector works

From [`notebooks/01_classical.ipynb`](../notebooks/01_classical.ipynb), section 1, and
[`src/forgery/classical.py`](../src/forgery/classical.py):

1. **Prepare:** grayscale, downscaled to at most 1,024 px on the longer side, optional contrast
   boost (CLAHE).
2. **Keypoints:** SIFT finds distinctive spots and describes the texture around each one.
3. **Self-match:** each keypoint is compared with every other keypoint in the same image. A
   match is kept only if it is clearly closer than the next-best candidate (the ratio test),
   or an exact duplicate. Matches between nearby spots are dropped.
4. **Group:** RANSAC looks for one movement (shift, rotation, scale) that explains many matches
   at once, removes them, and looks again, up to 4 times. A group needs at least
   `min_inliers` matches.
5. **Masks:** for each group, the outline (convex hull) around the source points and around
   the copy points, grown by `pad` pixels, in one mask layer. No group means authentic.

### Tuning

All 160 combinations of four settings were scored on a **tuning slice**: 25% of the training
cases (551 cases, 1,027 images), chosen with a fixed seed and stored in
`splits/tune_case_ids.txt`. Validation was not used. The log is
`experiments/tuning/01_sift.csv`.

| Setting | Values tried | Chosen |
|---|---|---|
| `clahe` | off, on | on |
| `ratio` | 0.4, 0.5, 0.6, 0.7 | 0.4 |
| `min_inliers` | 4, 6, 10, 15, 25 | 6 |
| `pad` (px) | 0, 5, 10, 20 | 10 |

The chosen setting scored **0.574** on the tuning slice. CLAHE was worth about +0.01 and
padding the masks about +0.017 (the best score with `pad` 0 was 0.558). The strictest `ratio`
won, and 0.4 is the edge of the grid, so a stricter value might do better still. The grid was
not widened after seeing this, since that would mean tuning further on the same images.

### Results (commit `930b13b`, from `experiments/results.csv`)

| Run | Predicted (val) | Actual (val) | Authentic left alone | Forged flagged | Mask score when flagged |
|---|---|---|---|---|---|
| Tuned detector (`01_sift`) | 0.70 | **0.5704** | 100% | 24.5% | 0.812 |
| Single RANSAC (at most 1 group) | — | **0.5596** | 100% | 24.5% | 0.730 |
| No RANSAC (no geometric check) | — | **0.5105** | 98.9% | 27.8% | 0.348 |

On the supplemental set the tuned detector scores **0.2826**. It flags 93.8% of the images,
but its mask score on them is only 0.301.

The single-RANSAC, no-RANSAC and supplemental rows are labelled `930b13b-dirty`. They ran on
exactly the same code as the first row. The only change was `results.csv` itself, which the
first run had appended to. The dirty check has since been fixed to ignore the experiment logs.

**On the Chapter 0 scale:**

| Floor (all authentic) | Whole-image masks | **SIFT + RANSAC** | Ceiling |
|---|---|---|---|
| 0.463 | 0.503 | **0.570** | 1.0 |

The detector beats perfect detection with whole-image masks, and covers about 20% of the gap
from the floor to the ceiling.

**Why the prediction was high:**
- The score breaks down as (475 × 1.00 + 550 × 0.245 × 0.812) ÷ 1025 = 0.570.
- Reaching 0.70 would need forged images to contribute more than twice as much: for example,
  flagging about 54% of them at the same mask quality, instead of 24.5%.
- The tuning score (0.574) turned out to be a close stand-in for validation (0.570), which
  is what a well-matched tuning slice should give.

### What each piece buys (validation)

- **The geometric check (RANSAC): +0.049** (0.5105 → 0.5596). Mostly through the masks.
  - Without it, one outline has to cover the source, the copy and everything in between, so
    the mask score falls from 0.730 to 0.348.
  - It also removes the 5 false alarms on authentic images that unchecked matches cause.
- **Finding several groups: +0.011** (0.5596 → 0.5704). Each duplicated region gets its own
  mask instead of only the strongest one, which raises the mask score from 0.730 to 0.812.
- **Tuning** (on the tuning slice): CLAHE about +0.01, mask padding about +0.017.

### Where it fails (validation, `01_sift`)

From the notebook's error analysis, section 4.

**Matching is the bottleneck, not grouping.** Each forged image is counted at the first step
it fails:

| Step reached | Forged images | Share |
|---|---|---|
| Fewer than 6 matches survive the ratio test | 397 | 72.2% |
| Enough matches, but RANSAC finds no group | 18 | 3.3% |
| Detected | 135 | 24.5% |

The median forged image yields just **1** match.
- **Missed images have almost no keypoints in the forged region:** a median of 12, against
  1,007 for detected ones.
- **SIFT needs texture, and many copied regions don't have any.** The largest misses in the
  gallery are smooth western-blot bands and a plain background; one has no keypoints at all
  inside the forged region.

**By kind of forged image:**

| Forged images | Count | Flagged | Mean score |
|---|---|---|---|
| All | 550 | 24.5% | 0.199 |
| Dark (mean gray < 40) | 192 | 2.1% | 0.009 |
| Few keypoints (< 50) | 100 | 0% | 0.000 |
| Tiny forgery (< 1% of image) | 187 | 2.1% | 0.015 |
| Several regions (2+) | 166 | 21.7% | 0.166 |

**Size decides almost everything.** By quarter of forged area, from the smallest quarter to
the largest, the share flagged rises from **2.2% → 10.2% → 35.0% → 50.7%**.

**When it fires, it is usually right.**
- It found the correct number of regions on 121 of the 135 forged images it flagged.
- **No authentic image was flagged (0 of 475).** The most matches on any authentic image
  was 8, and a group needs 6 consistent ones.
- **The detector is precise but blind:** its errors are almost all misses.

**Supplemental shows the opposite failure.** There the detector flags nearly everything, but
predicts too many masks: 24 of the 48 images get the maximum of 4. 14 of those have only 1
true region, and every extra mask cuts the score.

### What Chapter 2 needs to fix

- **Find copies without keypoints.** Most misses are smooth, dark or small regions where SIFT
  sees nothing. A model that predicts every pixel from learned features (segmentation) does
  not depend on keypoints.
- **Keep what works here:**
  - zero false alarms on validation
  - a mask score of 0.81 when it fires

  Later chapters can combine the two approaches, using this detector where it is confident.
- **Judge which matches are real.** The supplemental results show that repeated structure can
  produce extra groups, and the metric charges for every one.
