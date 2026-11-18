# Air-Quality

# Predicting PM2.5 from Multi-Sensor Air-Quality Readings

An end-to-end, leakage-safe machine-learning project that builds a **"virtual" (soft) sensor** to estimate fine particulate matter (**PM2.5**) from other low-cost sensor channels: PM10, NOx, NO₂, SO₂, VOCs, CO, CO₂, CH₄ and temperature.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-End--to--End-brightgreen)

---

##  Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Key Insights](#-key-insights)
- [Recommendations](#-recommendations)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Author](#-author)

---

##  Problem Statement

PM2.5 is the pollutant most strongly linked to cardiovascular and respiratory disease, but reference-grade PM2.5 monitors are expensive, need frequent calibration and are sparse. Cheaper gas sensors and temperature probes are widely deployed.

**Can PM2.5 be estimated accurately from the other sensor channels?** A working soft sensor would:
- back-fill readings when a PM2.5 monitor fails or is missing
- give exposure estimates and health alerts at locations that only have gas sensors

**Formal task:** supervised regression, `PM2.5 = f(PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4, Temperature)`

**Success criteria (set before modelling)**

| Criterion | Target |
|---|---|
| Generalisation | Test RMSE clearly below the "predict the mean" baseline |
| Stability | Small CV standard deviation and train–CV gap |
| Honesty | No leakage; fit and over-fit diagnostics documented |
| Practical value | Correct health band (AQI-style) for most observations |

---

## Dataset

- **File:** `air_quality_dataset.csv`
- **Size:** 500 observations × 10 numeric columns
- **Assumptions:** rows are i.i.d. (no timestamp or location column). Units are not given; magnitudes suggest µg/m³ for particulates and ppm for gases, but this is an assumption.

**Data-quality findings**
- Four negative readings (physically impossible)
- 7% of rows have PM10 < PM2.5 (PM10 contains PM2.5 by definition)
- 4% of rows have NO₂ > NOx (NOx = NO + NO₂)

Rules that learn nothing from the data are applied up front. Anything that learns a statistic (imputation, winsorising, scaling) lives inside the pipeline.

---

## Approach

```
raw sensors → negative→NaN → median impute → winsorise (1–99%) → feature engineering
            → GMM regime probabilities → scaling → model (standardised target)
```

1. **Audit & clean** with physical-validity rules
2. **Split first** (80/20, stratified on target quintiles). The test set is touched **once**, at the end.
3. **EDA on the training set only:** distributions, normality, univariate and multivariate outliers, VIF, PCA, hidden pollution regimes
4. **Physically motivated feature engineering:** log transforms, chemistry ratios, an Arrhenius temperature factor, Mahalanobis "unusualness" of the sensor vector, and **unsupervised Gaussian-mixture regime probabilities**
5. **Feature reduction:** VIF, clustering, LASSO, RFECV, PCA, with CV tests of whether reduction helps
6. **Scaler comparison**, then a model zoo (linear, regularised, kernel, KNN, bagging, boosting) against a dummy baseline
7. **Nested cross-validation** for an unbiased estimate of the tuned model, with the one-standard-error rule for the final choice
8. **Diagnostics:** learning curves, validation curves, bias–variance decomposition
9. **Leakage audit:** permutation test, a deliberately leaky feature, duplicates, autocorrelation, PM10 ablation

**Final model:** Ridge regression (α ≈ 5, standardised target) on engineered features plus regime probabilities.

---

## Key Results

| Metric | Value |
|---|---|
| Mean-predictor baseline, test RMSE | 24.05 µg/m³ |
| Cross-validated RMSE (training, 15 folds) | 12.67 |
| **Nested-CV RMSE** (tuning included) | 12.78 ± 0.83 |
| **Test RMSE** | **14.22** (95% bootstrap CI 11.9 to 16.5) |
| **Test R²** | **0.650** (95% CI 0.53 to 0.73) |
| Test MAE / MAPE | 11.3 µg/m³ / 24.9% |
| **Health-band accuracy** | 65% exact, **100% within one band** |

Test RMSE is higher than CV RMSE, but the test set is also noisier (baseline RMSE 24.05 vs 22.5 in training). Relative to the baseline the two agree (0.59 vs 0.57), and the CV figure lies inside the test confidence interval. With only 100 test rows, the test estimate is uncertain by roughly ±2.

---

## Key Insights

1. **The problem is "which regime?" more than "how much?"** The sensors split into a cleaner-air regime (≈ 64% of rows, PM2.5 ≈ 41) and a polluted-episode regime (≈ 36%, PM2.5 ≈ 79). Regime membership alone explains ≈ 67% of PM2.5 variance; within a regime, the sensors add almost nothing.
2. **Simple beats flexible.** Every model family lands at R² ≈ 0.60 to 0.68, and flexible models only over-fit. Regime-aware features let a plain linear model match or beat random forests and boosting.
3. **PM10 is unnecessary.** Feature selection and polynomial expansion don't help, engineering helps a little, and regime features help most.
4. **Residuals are heteroscedastic** (Breusch–Pagan p = 0.005): the polluted regime is noisier (σ ≈ 16 vs 11 µg/m³).
5. **Permutation importance** ranks CO₂ ≫ NOx > NO₂ > SO₂ > CH₄. The sensors are collinear and share credit, so these are predictive, **not causal**, effects.

---

## Recommendations

**Deployment**
- **Deploy the gas-only variant** (no PM10 input): same accuracy (test R² 0.660, RMSE 14.01) with one fewer sensor to maintain
- **Use it for screening and alerting, not as a precise instrument.** Individual estimates carry about ±12 to 14 µg/m³ of error.
- **Report regime-specific prediction intervals** (split-conformal or per-regime quantile regression)
- **Monitor drift** in regime shares and GMM component means; retrain before using the model at a new site

**Data and sensors**
- Fix data quality at the source and flag physically impossible records (the three test rows with PM10 < PM2.5 had RMSE 33 vs 13 for the rest)
- Add meteorology (humidity, wind, boundary-layer height), time and location, the most likely sources of the unexplained variance
- Collect more labelled episodes, especially in the transition zone between regimes

**Before relying on it**
- Validate on temporally and spatially held-out data
- Don't interpret coefficients or partial-dependence curves causally

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<my-username>/<my-repo-name>.git
cd <my-repo-name>
```

### 2. Install dependencies
```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn joblib jupyter
```

### 3. Add the data
Place `air_quality_dataset.csv` next to the notebook.

### 4. Run the notebook
```bash
jupyter notebook air_quality_pm25_modeling1.ipynb
```
Choose **Kernel → Restart & Run All**. Expect roughly 10 to 15 minutes on 2 CPU cores; the nested CV in §11 is the slow part.

> The notebook was developed with scikit-learn 1.9. The end of §14 saves the fitted pipeline with `joblib`; loading it later requires the custom classes (`NegativeToNaN`, `Winsorizer`, `FeatureEngineer`, `RegimeFeatures`) to be defined first.

---

## Repository Structure

```
├── air_quality_pm25_modeling.ipynb   # Full end-to-end notebook
├── air_quality_dataset.csv           # Sensor data (add locally)
└── README.md
```

---

## Limitations

- Only 500 rows and a 100-row test set, so confidence intervals are wide
- No time or location columns; units are assumed
- A single site's regime structure
- The number of regimes (k = 3) was chosen on training CV and is not distinguishable from k = 2 at this sample size
- Random i.i.d. splits cannot detect seasonal or site shift

---

## Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
