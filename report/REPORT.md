# Scientific Image Forgery Detection: a learning report

## Chapter 0: Foundation

### Baselines

#### Predictions (recorded 2026-10-26, before any baseline was run)

Validation set: 1,025 images (475 authentic, 550 forged). Estimates were worked out by reasoning
about the metric, then accepted by the user as their predictions.

| Baseline | Predicted val score | Reasoning |
|---|---|---|
| All authentic | 0.46 | 475 authentic images score 1, 550 forged score 0 → 475 / 1025 |
| Perfect detection, whole-image mask | ~0.50 | Authentic images score 1; a forged region covering ~3% of the image gives F1 ≈ 2 × 0.03 ≈ 0.06 → (475 + 550 × 0.06) / 1025 |
| Perfect detection, ground-truth mask | 1.0 | Every image is perfect |

#### Results (commit `4662281`, from `experiments/results.csv`)

| Baseline | Predicted (val) | Actual (val) | Actual (supplemental) |
|---|---|---|---|
| All authentic | 0.46 | **0.4634** | 0.0000 |
| Perfect detection, whole-image mask | ~0.50 | **0.5032** | 0.1305 |
| Perfect detection, ground-truth mask | 1.0 | **1.0000** | 1.0000 |

All three predictions were right to within 0.004. The one estimate inside them was off: we
guessed a whole-image mask would earn about 0.06 per forged image, and it earned **0.0742**
(`mean_of1_detected`). Forged area is skewed: the median forged image is only 2.2% forged, but
a minority of large forgeries pull the mean up to 5.1%, and those earn much more from a
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
