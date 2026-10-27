"""
Experiments module for Credit Card Fraud Risk Assessment.
Executes 3 rigorous experiments under the exact same user-level split protocol:
(1) Past-only user-history features vs Baseline row-level features [Test]
(2) Handling class imbalance: Class Weights vs SMOTE vs Threshold Tuning [Val / Test]
(3) Aggregation: Row-level classification vs User-level probability aggregation (Separate tables) [Test]
Includes high-risk base rate comparison.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.features import (
    BASE_FEATURES, BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES,
    ALL_EXPERIMENT_FEATURES, EXPERIMENT_NUMERIC_FEATURES
)
from src.split import create_user_split, build_preprocessor
from src.evaluate import (
    compute_comprehensive_metrics,
    tune_high_risk_threshold,
    compute_user_level_metrics,
    analyze_high_risk_false_negatives
)

RANDOM_SEED = 42


def run_experiment_features(
    df_base: pd.DataFrame,
    df_past: pd.DataFrame
) -> pd.DataFrame:
    """
    Experiment 1: Baseline row-level features vs Past-only user-history features.
    Evaluated on Holdout Test Users [Test, Row-Level].
    """
    print("\n--- Running Experiment 1: Past-Only User History Features [Test Set, Row-Level] ---")
    results = []

    # 1. Baseline
    df_tr_b, df_val_b, df_te_b = create_user_split(df_base, random_state=RANDOM_SEED)
    prep_b = build_preprocessor(BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES)
    X_tr_b = prep_b.fit_transform(df_tr_b)
    X_te_b = prep_b.transform(df_te_b)
    y_tr_b = df_tr_b['risk_class'].values
    y_te_b = df_te_b['risk_class'].values
    high_base_rate = float((y_te_b == 2).mean())

    clf_b = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    clf_b.fit(X_tr_b, y_tr_b)
    proba_b = clf_b.predict_proba(X_te_b)
    m_b = compute_comprehensive_metrics(y_te_b, proba_b)
    m_b['split'] = '[Test]'
    m_b['model'] = 'LightGBM (balanced, lr=0.05, depth=4)'
    m_b['feature_set'] = 'Baseline Row Features'
    m_b['calibration'] = 'None (Uncalibrated)'
    m_b['threshold'] = 'Argmax (th=0.33)'
    m_b['random_chance_pr_auc'] = high_base_rate
    results.append(m_b)

    # 2. Base + Past-only features
    df_tr_p, df_val_p, df_te_p = create_user_split(df_past, random_state=RANDOM_SEED)
    num_cols_p = BASE_NUMERIC_FEATURES + EXPERIMENT_NUMERIC_FEATURES
    prep_p = build_preprocessor(num_cols_p, BASE_CATEGORICAL_FEATURES)
    X_tr_p = prep_p.fit_transform(df_tr_p)
    X_te_p = prep_p.transform(df_te_p)
    y_tr_p = df_tr_p['risk_class'].values
    y_te_p = df_te_p['risk_class'].values

    clf_p = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    clf_p.fit(X_tr_p, y_tr_p)
    proba_p = clf_p.predict_proba(X_te_p)
    m_p = compute_comprehensive_metrics(y_te_p, proba_p)
    m_p['split'] = '[Test]'
    m_p['model'] = 'LightGBM (balanced, lr=0.05, depth=4)'
    m_p['feature_set'] = 'Baseline + Past-Only Features'
    m_p['calibration'] = 'None (Uncalibrated)'
    m_p['threshold'] = 'Argmax (th=0.33)'
    m_p['random_chance_pr_auc'] = high_base_rate
    results.append(m_p)

    cols = ['split', 'model', 'feature_set', 'calibration', 'threshold', 'accuracy', 'macro_f1', 'high_recall', 'high_precision', 'pr_auc_high', 'random_chance_pr_auc']
    return pd.DataFrame(results)[cols]


def run_experiment_imbalance(
    df_past: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment 2: Class Weights vs SMOTE vs Threshold Tuning.
    Returns:
    - val_df: Validation tuning metrics [Val]
    - test_df: Test set metrics [Test, Row-Level]
    """
    print("\n--- Running Experiment 2: Class Imbalance Strategies [Val & Test, Row-Level] ---")
    df_tr, df_val, df_te = create_user_split(df_past, random_state=RANDOM_SEED)
    num_cols = BASE_NUMERIC_FEATURES + EXPERIMENT_NUMERIC_FEATURES
    prep = build_preprocessor(num_cols, BASE_CATEGORICAL_FEATURES)

    X_tr = prep.fit_transform(df_tr)
    X_val = prep.transform(df_val)
    X_te = prep.transform(df_te)
    y_tr = df_tr['risk_class'].values
    y_val = df_val['risk_class'].values
    y_te = df_te['risk_class'].values

    high_base_rate_val = float((y_val == 2).mean())
    high_base_rate_test = float((y_te == 2).mean())

    val_results = []
    test_results = []

    # Strategy A: Unweighted Baseline (Argmax)
    clf_raw = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_estimators=100)
    clf_raw.fit(X_tr, y_tr)
    # Val
    p_val_raw = clf_raw.predict_proba(X_val)
    m_v_raw = compute_comprehensive_metrics(y_val, p_val_raw)
    m_v_raw['split'] = '[Val]'
    m_v_raw['model'] = 'LightGBM (unweighted)'
    m_v_raw['feature_set'] = 'Baseline + Past Features'
    m_v_raw['calibration'] = 'None'
    m_v_raw['threshold'] = 'Argmax (th=0.33)'
    m_v_raw['strategy'] = 'Standard (No weights / Argmax)'
    m_v_raw['random_chance_pr_auc'] = high_base_rate_val
    val_results.append(m_v_raw)
    # Test
    p_te_raw = clf_raw.predict_proba(X_te)
    m_t_raw = compute_comprehensive_metrics(y_te, p_te_raw)
    m_t_raw['split'] = '[Test]'
    m_t_raw['model'] = 'LightGBM (unweighted)'
    m_t_raw['feature_set'] = 'Baseline + Past Features'
    m_t_raw['calibration'] = 'None'
    m_t_raw['threshold'] = 'Argmax (th=0.33)'
    m_t_raw['strategy'] = 'Standard (No weights / Argmax)'
    m_t_raw['random_chance_pr_auc'] = high_base_rate_test
    test_results.append(m_t_raw)

    # Strategy B: Class Weights ('balanced')
    clf_cw = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    clf_cw.fit(X_tr, y_tr)
    # Val
    p_val_cw = clf_cw.predict_proba(X_val)
    m_v_cw = compute_comprehensive_metrics(y_val, p_val_cw)
    m_v_cw['split'] = '[Val]'
    m_v_cw['model'] = 'LightGBM (balanced)'
    m_v_cw['feature_set'] = 'Baseline + Past Features'
    m_v_cw['calibration'] = 'None'
    m_v_cw['threshold'] = 'Argmax (th=0.33)'
    m_v_cw['strategy'] = 'Class Weights (Balanced)'
    m_v_cw['random_chance_pr_auc'] = high_base_rate_val
    val_results.append(m_v_cw)
    # Test
    p_te_cw = clf_cw.predict_proba(X_te)
    m_t_cw = compute_comprehensive_metrics(y_te, p_te_cw)
    m_t_cw['split'] = '[Test]'
    m_t_cw['model'] = 'LightGBM (balanced)'
    m_t_cw['feature_set'] = 'Baseline + Past Features'
    m_t_cw['calibration'] = 'None'
    m_t_cw['threshold'] = 'Argmax (th=0.33)'
    m_t_cw['strategy'] = 'Class Weights (Balanced)'
    m_t_cw['random_chance_pr_auc'] = high_base_rate_test
    test_results.append(m_t_cw)

    # Strategy C: SMOTE (Oversampling inside pipeline)
    smote_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_SEED)),
        ('clf', lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_estimators=100))
    ])
    smote_pipe.fit(X_tr, y_tr)
    # Val
    p_val_smote = smote_pipe.predict_proba(X_val)
    m_v_smote = compute_comprehensive_metrics(y_val, p_val_smote)
    m_v_smote['split'] = '[Val]'
    m_v_smote['model'] = 'LightGBM + SMOTE'
    m_v_smote['feature_set'] = 'Baseline + Past Features'
    m_v_smote['calibration'] = 'None'
    m_v_smote['threshold'] = 'Argmax (th=0.33)'
    m_v_smote['strategy'] = 'SMOTE (Inside Pipeline)'
    m_v_smote['random_chance_pr_auc'] = high_base_rate_val
    val_results.append(m_v_smote)
    # Test
    p_te_smote = smote_pipe.predict_proba(X_te)
    m_t_smote = compute_comprehensive_metrics(y_te, p_te_smote)
    m_t_smote['split'] = '[Test]'
    m_t_smote['model'] = 'LightGBM + SMOTE'
    m_t_smote['feature_set'] = 'Baseline + Past Features'
    m_t_smote['calibration'] = 'None'
    m_t_smote['threshold'] = 'Argmax (th=0.33)'
    m_t_smote['strategy'] = 'SMOTE (Inside Pipeline)'
    m_t_smote['random_chance_pr_auc'] = high_base_rate_test
    test_results.append(m_t_smote)

    # Strategy D: Validation Threshold Tuning on Balanced Model (Tuned strictly on Val)
    best_th, th_metrics = tune_high_risk_threshold(y_val, p_val_cw, min_precision=0.15)
    # Val
    m_v_tuned = compute_comprehensive_metrics(y_val, p_val_cw, threshold=best_th)
    m_v_tuned['split'] = '[Val]'
    m_v_tuned['model'] = 'LightGBM (balanced)'
    m_v_tuned['feature_set'] = 'Baseline + Past Features'
    m_v_tuned['calibration'] = 'None'
    m_v_tuned['threshold'] = f'th={best_th:.2f} (tuned on Val)'
    m_v_tuned['strategy'] = f'Tuned Threshold (th={best_th:.2f} on Val)'
    m_v_tuned['random_chance_pr_auc'] = high_base_rate_val
    val_results.append(m_v_tuned)
    # Test
    m_t_tuned = compute_comprehensive_metrics(y_te, p_te_cw, threshold=best_th)
    m_t_tuned['split'] = '[Test]'
    m_t_tuned['model'] = 'LightGBM (balanced)'
    m_t_tuned['feature_set'] = 'Baseline + Past Features'
    m_t_tuned['calibration'] = 'None'
    m_t_tuned['threshold'] = f'th={best_th:.2f} (tuned on Val)'
    m_t_tuned['strategy'] = f'Tuned Threshold (th={best_th:.2f} on Val)'
    m_t_tuned['random_chance_pr_auc'] = high_base_rate_test
    test_results.append(m_t_tuned)

    cols = ['split', 'model', 'strategy', 'calibration', 'threshold', 'accuracy', 'macro_f1', 'high_recall', 'high_precision', 'pr_auc_high', 'random_chance_pr_auc']
    return pd.DataFrame(val_results)[cols], pd.DataFrame(test_results)[cols]


def run_experiment_aggregation(
    df_past: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment 3: Row-level Classification vs User-level Probability Aggregation.
    Keeps row-level and user-level in strictly separate tables.
    """
    print("\n--- Running Experiment 3: Row-Level vs User-Level Aggregation [Test Set] ---")
    df_tr, df_val, df_te = create_user_split(df_past, random_state=RANDOM_SEED)
    num_cols = BASE_NUMERIC_FEATURES + EXPERIMENT_NUMERIC_FEATURES
    prep = build_preprocessor(num_cols, BASE_CATEGORICAL_FEATURES)

    X_tr = prep.fit_transform(df_tr)
    X_val = prep.transform(df_val)
    X_te = prep.transform(df_te)
    y_tr = df_tr['risk_class'].values
    y_val = df_val['risk_class'].values
    y_te = df_te['risk_class'].values

    clf = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    clf.fit(X_tr, y_tr)

    # Strictly tune threshold on validation users only
    proba_val = clf.predict_proba(X_val)
    best_th, _ = tune_high_risk_threshold(y_val, proba_val, min_precision=0.15)

    proba_te = clf.predict_proba(X_te)

    # Separate Table A: Row-Level
    row_base_rate = float((y_te == 2).mean())
    row_m = compute_comprehensive_metrics(y_te, proba_te, threshold=best_th)
    row_m['split'] = '[Test]'
    row_m['model'] = 'LightGBM (balanced, lr=0.05, depth=4)'
    row_m['feature_set'] = 'Baseline + Past Features'
    row_m['calibration'] = 'None'
    row_m['threshold'] = f'th={best_th:.2f} (tuned on Val)'
    row_m['aggregation_level'] = 'Row-Level (N=20,550 Transactions)'
    row_m['random_chance_pr_auc'] = row_base_rate
    cols = ['split', 'model', 'feature_set', 'calibration', 'threshold', 'accuracy', 'macro_f1', 'high_recall', 'high_precision', 'pr_auc_high', 'random_chance_pr_auc']
    row_df = pd.DataFrame([row_m])[cols]

    # Separate Table B: User-Level
    user_m = compute_user_level_metrics(df_te, proba_te, threshold=best_th)
    user_preds_test = df_te.groupby('user_id')['risk_class'].first().values
    user_base_rate = float((user_preds_test == 2).mean())
    user_m['split'] = '[Test]'
    user_m['model'] = 'LightGBM (balanced, lr=0.05, depth=4)'
    user_m['feature_set'] = 'Baseline + Past Features'
    user_m['calibration'] = 'None'
    user_m['threshold'] = f'th={best_th:.2f} (tuned on Val)'
    user_m['aggregation_level'] = f'User-Level Aggregated (N={len(user_preds_test)} Users)'
    user_m['random_chance_pr_auc'] = user_base_rate
    user_df = pd.DataFrame([user_m])[cols]

    return row_df, user_df
