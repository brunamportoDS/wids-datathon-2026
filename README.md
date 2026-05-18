# wids-datathon-2026
# Predicting Wildfire Threat to Evacuation Zones Using Survival Analysis

**WiDS Global Datathon 2026 | Kaggle Competition**

Best Leaderboard Score: **0.9663** | Lightning Talk at WiDS Puget Sound Conference 2026

## Overview

When a wildfire ignites, emergency managers need to know: which fires will reach communities, and how soon? This project predicts the probability that a wildfire will come within 5km of an evacuation zone at four time horizons (12h, 24h, 48h, 72h) using only data from the first 5 hours after ignition.

The problem is framed as **right-censored survival analysis** with a hybrid evaluation metric: 0.3 C-index + 0.7 Weighted Brier Score, heavily weighting probability calibration over ranking accuracy.

## Key Findings

**The 5km perfect threshold:** Every fire within 5km hit its evacuation zone (100%, n=69). Every fire beyond 5km did not (0%, n=152). Distance is a perfect binary separator ‚Äî the real challenge is predicting *when*, not *whether*.

**Risk Score + Platt Calibration architecture:** Separating ranking (GBS risk scores) from calibration (Platt scaling with 2 parameters) outperformed direct survival curve predictions. Non-parametric isotonic calibration overfit on 221 rows and *hurt* leaderboard performance by 0.01.

**The ensemble paradox:** Adding Cox PH (weakest individual model, CV 0.948) to Gradient Boosting Survival (strongest, CV 0.974) consistently improved test performance. Model diversity beats single-model optimization on small datasets.

## Winning Approach

Three-component architecture:

1. **Gradient Boosting Survival** produces risk scores that rank fires by urgency
2. **Platt scaling** (logistic regression, 2 parameters per horizon) converts risk scores to calibrated probabilities
3. **Cox PH** (18% blend weight) adds model diversity through structurally different errors

Final blend: **82% Risk+Platt + 18% Cox PH ‚Üí LB 0.9663**

## Dataset

| Property | Value |
|----------|-------|
| Training observations | 221 |
| Test observations | 95 |
| Features | 34 (all numeric) |
| Event rate | 31.2% (69 hits, 152 censored) |
| Censoring rate | 68.8% |


## Feature Engineering

16 features engineered from 34 raw inputs:

- **Binary indicators (5):** has_growth, has_movement, has_closing_speed, has_alignment, has_distance_change
- **Distance transforms (2):** log(distance), distance in km
- **Interaction features (3):** alignment/distance, closing_speed/distance, growth_rate/distance
- **Temporal flags (2):** is_afternoon, is_summer
- **Original features retained (4):** dist_min_ci, alignment_abs, num_perimeters, dt_first_last

5 features achieve 97% of full performance, confirming signal concentration.

**Note:** scikit-survival requires specific version pinning to avoid compatibility issues:
```bash
pip install scikit-survival==0.23.1 scikit-learn==1.5.2 numpy==1.26.4 --force-reinstall
```

## Cross-Validation

| Method | Hybrid | C-index | Brier |
|--------|--------|---------|-------|
| 5-Fold Stratified | 0.972 | 0.948 | 0.015 |
| Leave-One-Out (221 folds) | 0.971 | 0.938 | 0.015 |

Close agreement confirms model robustness. Adversarial validation AUC = 0.424, confirming no train-test distribution shift.

## Lessons Learned

1. **On small datasets, every parameter is a liability.** Shallow trees (max_depth=2), few features (5 capture 97%), and parametric calibration (Platt) all outperformed complex alternatives.
2. **Model diversity beats single-model optimization.** A weak model can improve a strong one if it makes structurally different errors.
3. **Calibration method matters more than model choice** when the metric weights calibration at 70%. Platt scaling (2 params) vs isotonic regression (n‚àí1 params) was the single highest-impact decision.
4. **CV-LB divergence is real on small data.** Best CV model ‚â† best LB model. Validate with LOO CV and adversarial validation.

## Acknowledgments

- **WiDS (Women in Data Science)** for organizing the global datathon
- **Watch Duty** (nonprofit wildfire alert service) for providing the dataset
- **WiDS Puget Sound** for the local datathon event and conference opportunity

## Competition

- [WiDS Global Datathon 2026 on Kaggle](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- [WiDS Puget Sound](https://www.widspugetsound.org/)


