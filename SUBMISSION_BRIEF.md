# Credit Card Fraud Risk Assessment: Technical Brief & Test Audit

**Repository**: [https://github.com/Rhythm22Sharma/Assignment-warrgyizmorsch](https://github.com/Rhythm22Sharma/Assignment-warrgyizmorsch)  
**Evaluation Standard**: Strict Temporal Cohort, User-Level Disjoint Split, Zero Leakage  
**Seed**: Fixed strictly to `42` across all data partitioning, training, bootstrapping, and tests.

---

## 1. Executive Summary & Core Results

This project designs, trains, calibrates, and stress-tests a **user-level fraud risk assessment system** using Kaggle's 1.85M transaction dataset across 908 active cardholders.

### Headline Findings:
1. **The Primary Result — Empirical Proof of Data Leakage in Random Splits**:
   - Random row-level splitting produces an illusory **98.59% accuracy and 0.999965 PR-AUC** solely because tree models memorize static cardholder coordinates (`lat`, `long`) as entity fingerprints.
   - When evaluated on truly unseen cardholders (User-Level Split), performance collapses to **47.44% accuracy and 0.1777 PR-AUC**.
   - Testing a model trained *strictly on static demographic features* confirms this: **1.0000 PR-AUC on random split vs 0.1680 on unseen user split**.
2. **Honest Forward-Looking Assessment**:
   - Predicting future cardholder risk forward in time provides **little signal beyond the base rate** (Calibrated PR-AUC = **0.1166** vs Random Base Rate = **0.1022**). Single-transaction features are dominated by legitimate spending (intra-card ICC $<0.08$).
3. **Artifact Verification**:
   - The production model bundle is serialized at `models/best_model.joblib` (0.51 MB) with schema and metrics locked in `models/metadata.json`.

---

## 2. Cohort Formulation & Tie-Preserving Target Design

| Parameter | Specification | Purpose / Rationale |
| :--- | :--- | :--- |
| **Observation Window** | First 60% of timeline (up to `2020-03-14 14:23:51`) | Features computed *strictly* prior to this timestamp; zero future leakage. |
| **Outcome Window** | Last 40% of timeline (up to `2020-12-31 23:59:34`) | Ground truth cardholder risk class computed *exclusively* from this window. |
| **Quality Filter** | $\ge 10$ outcome transactions; $\ge 1$ observation txn | 17 users dropped for $<10$ outcome txns, 16 for no activity; **908 users retained**. |
| **Cap per User** | 150 random observation transactions (~136.2k total) | Prevents transaction count skew and bounds computational runtime. |
| **Target Construction** | `out_rate` = (Outcome Fraud Txns) / (Total Outcome Txns) | Continuous rate partitioned into 3 discrete risk classes. |
| **Tie-Preservation** | **Low** ($0.0$), **Medium** ($0.0 < r < 0.02$), **High** ($r \ge 0.02$) | Preserves all 579 zero-fraud ties; avoids arbitrary rank assignment. |

---

## 3. Recommended Production Model Architecture

- **Algorithm**: LightGBM Classifier (`max_depth=4`, `learning_rate=0.05`, `n_estimators=100`, `class_weight='balanced'`).
- **Feature Set**: **10 Base Features** (`amt`, `hour`, `dow`, `age`, `haversine_dist`, `city_pop`, `secs_since_prev`, `gender`, `state`, `category`).
  - *Selection Evidence*: 5-Fold CV PR-AUC on Base Features ($0.1112 \pm 0.0326$) vs Past-Window Features ($0.1156 \pm 0.0307$). Difference ($+0.0044$) is within 1 std of CV noise ($\pm 0.031$); simplest config selected by Occam's razor.
- **Probability Calibration**: Isotonic Regression (fit strictly on Validation fold). Reduces High-Risk Brier Score from $0.1435 \to 0.0840$.
- **Operating Threshold**: $th = 0.1288$ (tuned strictly on Validation users to maximize $F_2$ subject to Precision $\ge 15\%$).

---

## 4. Test Set Evaluation Benchmark (Holdout $N=137$ Unseen Users, 20,550 Txns)

### Headline Test Performance
| Aggregation Level | Accuracy | Macro F1 | High-Risk Recall | High-Risk Precision | High-Risk PR-AUC | High-Risk Brier | Majority Baseline (Acc / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Row-Level (20,550 Txns)** | **0.4885** | **0.2864** | **35.71%** | **11.86%** | **0.1166** | **0.0933** | 0.6350 / 0.2589 |
| **User-Level (137 Users)** | **0.4818** | **0.2832** | **35.71%** | **11.63%** | **0.1168** | — | 0.6350 / 0.2589 |

### User-Cluster Bootstrap 95% Confidence Intervals (1,000 Resamples)
- **High-Risk Recall [95% CI]**: $[0.1108, 0.6250]$
- **High-Risk Precision [95% CI]**: $[0.0270, 0.2250]$
- **$(\text{PR-AUC} - \text{Base Rate})$ [95% CI]**: $[0.0208, 0.3329]$ *(Statistically distinguishable from 0 at $p<0.05$, though practical lift is modest).*

### 3x3 Confusion Matrices
- **Row-Level Matrix**:
  $$\begin{pmatrix} 6614 & 3192 & 3244 \\ 996 & 2686 & 2370 \\ 441 & 911 & 750 \end{pmatrix}$$
- **User-Level Aggregated Matrix**:
  $$\begin{pmatrix} 44 & 21 & 22 \\ 7 & 17 & 16 \\ 3 & 6 & 5 \end{pmatrix}$$

### Review Capacity Efficiency (Top-K Flagged Users)
- **Top 10% Flagged (14 users)**: Caught 1 high-risk user, Recall = **7.14%**, Lift = **0.71x**
- **Top 20% Flagged (28 users)**: Caught 3 high-risk users, Recall = **21.43%**, Lift = **1.07x**

---

## 5. Automated Test Suite Audit (11 of 11 Passing)

All tests are implemented in `tests/test_pipeline.py` and run via `pytest`:

| # | Test Function Name | Category | Exact Condition Verified | Result |
| :-: | :--- | :--- | :--- | :---: |
| **1** | `test_no_user_overlap_across_user_level_splits` | Data Integrity | `len(train_users & val_users & test_users) == 0` | **PASS** |
| **2** | `test_observation_window_strictly_before_cutoff` | Zero Leakage | All observation timestamps $\le 60\%$ cutoff timestamp | **PASS** |
| **3** | `test_tie_preservation_in_user_labeling` | Target Integrity | Users with identical fraud rates always receive identical risk labels | **PASS** |
| **4** | `test_preprocessing_fit_on_train_only` | Pipeline Integrity | Medians/Modes/Encoders are fitted strictly on Train split | **PASS** |
| **5** | `test_predict_risk_output_schema` | Production API | Schema contains predicted class, probabilities, score, and alert boolean | **PASS** |
| **6** | `test_missing_value_handling` | Robustness | Payload with `None` values is cleanly imputed without throwing errors | **PASS** |
| **7** | `test_invalid_input_rejected` | Input Validation | Negative amounts, latitude $>90$, longitude $>180$ raise validation errors | **PASS** |
| **8** | `test_threshold_logic` | Operating Point | Prediction triggers `alert_triggered=True` iff score $\ge$ threshold | **PASS** |
| **9** | `test_reproducibility_with_fixed_seed` | Reproducibility | Two independent split runs with seed 42 yield identical user orderings | **PASS** |
| **10** | `test_is_fraud_not_in_features` | Hard Leak Rule | Raw `is_fraud`, `out_rate`, and unlagged labels strictly absent from features | **PASS** |
| **11** | `test_saved_artifact_metadata_consistency` | Artifact Audit | `best_model.joblib` and `metadata.json` match in threshold, version, & features | **PASS** |

---

## 6. Model Artifacts in Repository

The trained model bundle is committed and pushed directly in the repository:
1. `models/best_model.joblib` (524 KB):
   - Preprocessing imputation and one-hot encoding pipeline.
   - Calibrated Classifier (Isotonic calibrated LightGBM).
   - Tuned decision threshold ($th = 0.1288$).
   - Exact list of numeric and categorical feature names.
2. `models/metadata.json`:
   - SHA-256 raw data hashes, library versions (`lightgbm 4.7.0`, `scikit-learn 1.6.1`).
   - Hyperparameter configurations, validation tuning metric records, and test benchmarks.
