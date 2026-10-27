"""
Evaluation and High-Risk Detection Module.
Features:
- Comprehensive metrics: Accuracy, Macro F1, Per-class P/R/F1, Confusion Matrix, PR-AUC (High)
- Probability Calibration (Platt / Sigmoid vs Isotonic) & Brier Score (fit on Val only)
- Validation threshold tuning (maximizing F2 / recall at min precision constraint)
- User-level probability aggregation (average transaction probabilities per user)
- High-risk False Negative error analysis across transaction attributes
- Business trade-off explanation (Missed fraud cost vs Analyst workload).
"""

from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc, brier_score_loss, classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def compute_cluster_bootstrap_ci(
    df_eval: pd.DataFrame,
    y_true_col: str = 'y_true',
    y_pred_col: str = 'y_pred',
    user_col: str = 'user_id',
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% cluster (user-level) bootstrap confidence intervals for recall and precision
    by resampling whole users (with replacement), preserving intra-user transaction correlation.
    """
    rng = np.random.RandomState(random_state)
    unique_users = df_eval[user_col].unique()
    n_users = len(unique_users)

    # Pre-aggregate user level ground truth and prediction
    user_agg = df_eval.groupby(user_col).agg({y_true_col: 'first', y_pred_col: 'mean'})
    user_agg['user_pred_binary'] = (user_agg[y_pred_col] >= 0.5).astype(int)

    # Map user to transaction row arrays
    user_to_rows = df_eval.groupby(user_col)[[y_true_col, y_pred_col]].apply(
        lambda g: (g[y_true_col].values, g[y_pred_col].values)
    ).to_dict()

    row_recalls, row_precs = [], []
    user_recalls, user_precs = [], []
    row_pr_auc_diffs = []
    user_pr_auc_diffs = []

    for _ in range(n_bootstraps):
        sampled_users = rng.choice(unique_users, size=n_users, replace=True)

        # 1. User level
        u_sample = user_agg.loc[sampled_users]
        if u_sample[y_true_col].nunique() > 1:
            user_recalls.append(recall_score(u_sample[y_true_col], u_sample['user_pred_binary'], zero_division=0))
            user_precs.append(precision_score(u_sample[y_true_col], u_sample['user_pred_binary'], zero_division=0))
            
            p_prec_u, p_rec_u, _ = precision_recall_curve(u_sample[y_true_col], u_sample[y_pred_col])
            pr_auc_u = auc(p_rec_u, p_prec_u)
            base_rate_u = float(u_sample[y_true_col].mean())
            user_pr_auc_diffs.append(pr_auc_u - base_rate_u)

        # 2. Row level
        sample_y_true = np.concatenate([user_to_rows[u][0] for u in sampled_users])
        sample_y_pred = np.concatenate([user_to_rows[u][1] for u in sampled_users])
        if len(np.unique(sample_y_true)) > 1:
            row_recalls.append(recall_score(sample_y_true, sample_y_pred, zero_division=0))
            row_precs.append(precision_score(sample_y_true, sample_y_pred, zero_division=0))
            
            p_prec_r, p_rec_r, _ = precision_recall_curve(sample_y_true, sample_y_pred)
            pr_auc_r = auc(p_rec_r, p_prec_r)
            base_rate_r = float(sample_y_true.mean())
            row_pr_auc_diffs.append(pr_auc_r - base_rate_r)

    user_diff_ci = (float(np.percentile(user_pr_auc_diffs, 100 * (alpha / 2))), float(np.percentile(user_pr_auc_diffs, 100 * (1 - alpha / 2))))
    row_diff_ci = (float(np.percentile(row_pr_auc_diffs, 100 * (alpha / 2))), float(np.percentile(row_pr_auc_diffs, 100 * (1 - alpha / 2))))

    return {
        'row_recall_ci': (float(np.percentile(row_recalls, 100 * (alpha / 2))), float(np.percentile(row_recalls, 100 * (1 - alpha / 2)))),
        'row_precision_ci': (float(np.percentile(row_precs, 100 * (alpha / 2))), float(np.percentile(row_precs, 100 * (1 - alpha / 2)))),
        'user_recall_ci': (float(np.percentile(user_recalls, 100 * (alpha / 2))), float(np.percentile(user_recalls, 100 * (1 - alpha / 2)))),
        'user_precision_ci': (float(np.percentile(user_precs, 100 * (alpha / 2))), float(np.percentile(user_precs, 100 * (1 - alpha / 2)))),
        'user_pr_auc_diff_ci': user_diff_ci,
        'row_pr_auc_diff_ci': row_diff_ci,
        'user_lift_distinguishable_from_zero': bool(user_diff_ci[0] > 0),
        'row_lift_distinguishable_from_zero': bool(row_diff_ci[0] > 0)
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: Optional[float] = None,
    high_class_idx: int = 2
) -> Dict[str, Any]:
    """
    Computes row-level classification metrics given probabilities and an optional custom threshold for high-risk.
    """
    if threshold is None:
        y_pred = np.argmax(y_proba, axis=1)
    else:
        # If high-risk probability exceeds threshold, classify as high-risk (class 2)
        # otherwise argmax between class 0 and 1
        is_high = y_proba[:, high_class_idx] >= threshold
        argmax_low_med = np.argmax(y_proba[:, :2], axis=1)
        y_pred = np.where(is_high, high_class_idx, argmax_low_med)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class P/R/F1
    labels = [0, 1, 2]
    class_names = ['low', 'medium', 'high']
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {
        class_names[i]: {'precision': float(precisions[i]), 'recall': float(recalls[i]), 'f1': float(f1s[i])}
        for i in range(len(class_names))
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # PR-AUC for high-risk class
    p_prec, p_rec, _ = precision_recall_curve((y_true == high_class_idx).astype(int), y_proba[:, high_class_idx])
    pr_auc_high = auc(p_rec, p_prec)

    # Brier score for high-risk class
    brier = brier_score_loss((y_true == high_class_idx).astype(int), y_proba[:, high_class_idx])

    # Majority baseline (predict all Low / class 0)
    majority_acc = float((y_true == 0).mean())
    majority_macro_f1 = float(f1_score(y_true, np.zeros_like(y_true), average='macro', zero_division=0))

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'majority_baseline_accuracy': majority_acc,
        'majority_baseline_macro_f1': majority_macro_f1,
        'high_recall': float(recalls[high_class_idx]),
        'high_precision': float(precisions[high_class_idx]),
        'high_f1': float(f1s[high_class_idx]),
        'pr_auc_high': float(pr_auc_high),
        'brier_score_high': float(brier),
        'per_class': per_class,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


def tune_high_risk_threshold(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    high_class_idx: int = 2,
    min_precision: float = 0.20,
    beta: float = 2.0
) -> Tuple[float, Dict[str, float]]:
    """
    Tunes threshold on VALIDATION set only to maximize F-beta (F2, prioritizing recall)
    while respecting a minimum precision constraint.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.33
    best_fbeta = -1.0
    best_metrics = {}

    y_val_binary = (y_val == high_class_idx).astype(int)
    p_high = proba_val[:, high_class_idx]

    for th in thresholds:
        pred_binary = (p_high >= th).astype(int)
        rec = recall_score(y_val_binary, pred_binary, zero_division=0)
        prec = precision_score(y_val_binary, pred_binary, zero_division=0)

        # F-beta calculation: (1 + beta^2) * prec * rec / (beta^2 * prec + rec)
        if (prec + rec) > 0 and prec >= min_precision:
            fbeta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_thresh = float(th)
                best_metrics = {'threshold': float(th), 'precision': float(prec), 'recall': float(rec), 'fbeta': float(fbeta)}

    # Fallback if constraint too strict
    if best_fbeta < 0:
        best_thresh = 0.20
        pred_binary = (p_high >= best_thresh).astype(int)
        best_metrics = {
            'threshold': best_thresh,
            'precision': float(precision_score(y_val_binary, pred_binary, zero_division=0)),
            'recall': float(recall_score(y_val_binary, pred_binary, zero_division=0)),
            'fbeta': 0.0
        }

    return best_thresh, best_metrics


def calibrate_probabilities(
    base_model,
    X_val,
    y_val,
    method: str = 'sigmoid'
) -> CalibratedClassifierCV:
    """
    Fits probability calibration on VALIDATION data only (never on test).
    Method can be 'sigmoid' (Platt scaling) or 'isotonic'.
    """
    calibrator = CalibratedClassifierCV(estimator=base_model, method=method, cv='prefit')
    calibrator.fit(X_val, y_val)
    return calibrator


def compute_user_level_metrics(
    df_eval: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: Optional[float] = None,
    high_class_idx: int = 2
) -> Dict[str, Any]:
    """
    Aggregates row-level probabilities per user:
    - User probability = mean of row probabilities for that user.
    - User predicted class = thresholded or argmax on average probabilities.
    - Evaluated against true user risk_class.
    """
    df_agg = df_eval[['user_id', 'risk_class']].copy()
    for c in range(y_proba.shape[1]):
        df_agg[f'proba_{c}'] = y_proba[:, c]

    # Aggregate by user
    user_preds = df_agg.groupby('user_id').agg({
        'risk_class': 'first',
        'proba_0': 'mean',
        'proba_1': 'mean',
        'proba_2': 'mean'
    }).reset_index()

    user_y_true = user_preds['risk_class'].values
    user_proba = user_preds[['proba_0', 'proba_1', 'proba_2']].values

    return compute_comprehensive_metrics(
        y_true=user_y_true,
        y_proba=user_proba,
        threshold=threshold,
        high_class_idx=high_class_idx
    )


def compute_threshold_sweep_table(
    y_val: np.ndarray,
    proba_val_high: np.ndarray,
    high_class_idx: int = 2,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Computes validation threshold sweep table reporting threshold, recall, precision, flag rate, and F2.
    """
    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.1288, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    y_bin = (y_val == high_class_idx).astype(int)
    records = []
    for th in thresholds:
        pred_bin = (proba_val_high >= th).astype(int)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        flag_rate = float(pred_bin.mean())
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        records.append({
            'Threshold': float(th),
            'Recall (High)': float(rec),
            'Precision (High)': float(prec),
            'Flag Rate': float(flag_rate),
            'F2 Score': float(f2)
        })
    return pd.DataFrame(records)


def compute_recall_at_top_k_users(
    df_eval: pd.DataFrame,
    y_proba: np.ndarray,
    high_class_idx: int = 2,
    k_list: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Ranks users by their mean high-risk probability.
    Computes recall at top k% users flagged and lift over random baseline.
    """
    if k_list is None:
        k_list = [0.10, 0.20]

    df_copy = df_eval[['user_id', 'risk_class']].copy()
    df_copy['proba_high'] = y_proba[:, high_class_idx]
    
    user_agg = df_copy.groupby('user_id').agg({
        'risk_class': 'first',
        'proba_high': 'mean'
    }).reset_index()
    
    user_agg['is_high_true'] = (user_agg['risk_class'] == high_class_idx).astype(int)
    total_high_users = int(user_agg['is_high_true'].sum())
    total_users = int(len(user_agg))
    
    user_agg = user_agg.sort_values('proba_high', ascending=False).reset_index(drop=True)
    
    results = []
    for k in k_list:
        n_flagged = int(np.ceil(total_users * k))
        top_k_slice = user_agg.iloc[:n_flagged]
        caught_high = int(top_k_slice['is_high_true'].sum())
        recall = caught_high / max(total_high_users, 1)
        precision = caught_high / max(n_flagged, 1)
        random_recall = k
        lift_vs_random = recall / random_recall if random_recall > 0 else 1.0
        
        results.append({
            'top_k_pct': float(k),
            'n_flagged': n_flagged,
            'total_users': total_users,
            'caught_high': caught_high,
            'total_high': total_high_users,
            'recall': float(recall),
            'precision': float(precision),
            'random_recall': float(random_recall),
            'lift_vs_random': float(lift_vs_random)
        })
    return results


def plot_pr_curve_with_operating_point(
    y_test: np.ndarray,
    proba_test_high: np.ndarray,
    chosen_threshold: float,
    high_class_idx: int = 2,
    save_path: str = 'reports/figures/pr_curve_operating_point.png'
):
    """
    Plots the continuous Precision-Recall curve on the test set,
    marks the chosen operating point, and compares to the random chance base rate.
    """
    y_bin = (y_test == high_class_idx).astype(int)
    base_rate = float(y_bin.mean())
    precisions, recalls, thresholds = precision_recall_curve(y_bin, proba_test_high)
    pr_auc = auc(recalls, precisions)
    
    # Calculate precision and recall at chosen operating point
    pred_bin = (proba_test_high >= chosen_threshold).astype(int)
    chosen_rec = recall_score(y_bin, pred_bin, zero_division=0)
    chosen_prec = precision_score(y_bin, pred_bin, zero_division=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color='#2980b9', lw=2.5, label=f'Model PR Curve (PR-AUC = {pr_auc:.4f})')
    ax.axhline(y=base_rate, color='#7f8c8d', linestyle='--', lw=1.5, label=f'Random Chance Base Rate ({base_rate*100:.2f}%)')
    
    ax.plot(chosen_rec, chosen_prec, marker='*', markersize=14, color='#e74c3c', label=f'Chosen Cut (th={chosen_threshold:.4f})\nRecall={chosen_rec*100:.1f}%, Prec={chosen_prec*100:.1f}%')
    
    ax.set_xlabel('Recall (High-Risk)', fontsize=12)
    ax.set_ylabel('Precision (High-Risk)', fontsize=12)
    ax.set_title('Precision-Recall Curve with Chosen Operating Threshold', fontweight='bold', fontsize=13)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PR curve operating point plot saved: {save_path}")


def analyze_high_risk_false_negatives(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    high_class_idx: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive false negative analysis:
    - By category
    - By amount bucket (<$25, $25-$100, $100-$500, >$500)
    - By hour (Night 0-5, Morning 6-11, Afternoon 12-17, Evening 18-23)
    - Profile comparison of caught vs missed high-risk test users
    """
    df_analysis = df_test.copy()
    df_analysis['y_pred'] = y_pred
    df_analysis['is_high_true'] = df_analysis['risk_class'] == high_class_idx
    df_analysis['is_high_pred'] = df_analysis['y_pred'] == high_class_idx

    df_analysis['is_fn'] = df_analysis['is_high_true'] & (~df_analysis['is_high_pred'])
    df_analysis['is_tp'] = df_analysis['is_high_true'] & df_analysis['is_high_pred']

    high_txns = df_analysis[df_analysis['is_high_true']].copy()
    if len(high_txns) == 0:
        return {}

    # 1. By category
    cat_fn = high_txns.groupby('category', observed=False).agg(
        total_high=('user_id', 'count'),
        false_negatives=('is_fn', 'sum'),
        true_positives=('is_tp', 'sum')
    )
    cat_fn['fn_rate'] = cat_fn['false_negatives'] / np.maximum(cat_fn['total_high'], 1)
    cat_fn = cat_fn.sort_values('fn_rate', ascending=False)

    # 2. By amount bucket
    amt_bins = [-1, 25, 100, 500, 100000]
    amt_labels = ['<$25', '$25-$100', '$100-$500', '>$500']
    high_txns['amt_bucket'] = pd.cut(high_txns['amt'], bins=amt_bins, labels=amt_labels)
    amt_fn = high_txns.groupby('amt_bucket', observed=False).agg(
        total_high=('user_id', 'count'),
        false_negatives=('is_fn', 'sum'),
        true_positives=('is_tp', 'sum')
    )
    amt_fn['fn_rate'] = amt_fn['false_negatives'] / np.maximum(amt_fn['total_high'], 1)

    # 3. By hour bucket
    hour_bins = [-1, 5, 11, 17, 24]
    hour_labels = ['Night (00-05)', 'Morning (06-11)', 'Afternoon (12-17)', 'Evening (18-23)']
    high_txns['hour_bucket'] = pd.cut(high_txns['hour'], bins=hour_bins, labels=hour_labels)
    hour_fn = high_txns.groupby('hour_bucket', observed=False).agg(
        total_high=('user_id', 'count'),
        false_negatives=('is_fn', 'sum'),
        true_positives=('is_tp', 'sum')
    )
    hour_fn['fn_rate'] = hour_fn['false_negatives'] / np.maximum(hour_fn['total_high'], 1)

    # 4. User profile comparison: Caught vs Missed High-Risk Users
    user_flagged = df_analysis.groupby('user_id')['is_high_pred'].any()
    high_users_profile = df_analysis[df_analysis['is_high_true']].groupby('user_id').agg(
        mean_amt=('amt', 'mean'),
        total_txns=('amt', 'count'),
        mean_dist=('haversine_dist', 'mean')
    )
    high_users_profile['is_caught'] = user_flagged.loc[high_users_profile.index]
    
    caught_users = high_users_profile[high_users_profile['is_caught']]
    missed_users = high_users_profile[~high_users_profile['is_caught']]
    
    user_compare = pd.DataFrame([
        {
            'High-Risk Cohort': f'Caught High-Risk Users (N={len(caught_users)})',
            'Mean Transaction Amount': f"${caught_users['mean_amt'].mean():.2f}" if len(caught_users) > 0 else 'N/A',
            'Mean Transactions per User': f"{caught_users['total_txns'].mean():.1f}" if len(caught_users) > 0 else 'N/A',
            'Mean Merchant Distance (km)': f"{caught_users['mean_dist'].mean():.2f} km" if len(caught_users) > 0 else 'N/A'
        },
        {
            'High-Risk Cohort': f'Missed High-Risk Users (N={len(missed_users)})',
            'Mean Transaction Amount': f"${missed_users['mean_amt'].mean():.2f}" if len(missed_users) > 0 else 'N/A',
            'Mean Transactions per User': f"{missed_users['total_txns'].mean():.1f}" if len(missed_users) > 0 else 'N/A',
            'Mean Merchant Distance (km)': f"{missed_users['mean_dist'].mean():.2f} km" if len(missed_users) > 0 else 'N/A'
        }
    ])

    return {
        'category_fn': cat_fn,
        'amt_fn': amt_fn,
        'hour_fn': hour_fn,
        'user_compare': user_compare
    }


def plot_calibration_curves(
    y_test: np.ndarray,
    proba_uncal: np.ndarray,
    proba_platt: np.ndarray,
    proba_iso: np.ndarray,
    save_path: str = 'reports/figures/calibration_curves.png'
):
    """Plots reliability calibration curves comparing uncalibrated, Platt, and Isotonic methods."""
    y_bin = (y_test == 2).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Uncalibrated
    fop_u, mpv_u = calibration_curve(y_bin, proba_uncal[:, 2], n_bins=10)
    brier_u = brier_score_loss(y_bin, proba_uncal[:, 2])
    ax.plot(mpv_u, fop_u, marker='o', label=f'Uncalibrated (Brier={brier_u:.4f})', color='#e74c3c')

    # Platt
    fop_p, mpv_p = calibration_curve(y_bin, proba_platt[:, 2], n_bins=10)
    brier_p = brier_score_loss(y_bin, proba_platt[:, 2])
    ax.plot(mpv_p, fop_p, marker='s', label=f'Platt / Sigmoid (Brier={brier_p:.4f})', color='#3498db')

    # Isotonic
    fop_i, mpv_i = calibration_curve(y_bin, proba_iso[:, 2], n_bins=10)
    brier_i = brier_score_loss(y_bin, proba_iso[:, 2])
    ax.plot(mpv_i, fop_i, marker='^', label=f'Isotonic (Brier={brier_i:.4f})', color='#2ecc71')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Empirical Fraction of High-Risk Positives')
    ax.set_title('High-Risk Probability Calibration Curves', fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Calibration plot saved: {save_path}")
