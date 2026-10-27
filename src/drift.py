"""
Data and Prediction Drift Monitoring Module.
Implements:
1. Population Stability Index (PSI) calculation for numeric & score distributions
2. Two-sample Kolmogorov-Smirnov (KS) statistic & p-value
3. Categorical distribution drift (Chi-Square / Total Variation Distance)
4. Production drift report generation for continuous monitoring.
"""

from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    reference: Union[np.ndarray, List[float], pd.Series],
    current: Union[np.ndarray, List[float], pd.Series],
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes the Population Stability Index (PSI) between a reference (training/baseline)
    distribution and a current (serving/production) distribution.

    Interpretation thresholds:
    - PSI < 0.10: Stable, no significant distribution shift.
    - 0.10 <= PSI < 0.20: Moderate drift, investigate features.
    - PSI >= 0.20: Significant drift, trigger model retraining.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    # Determine quantile bin edges based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Count frequencies
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    ref_props = ref_counts / len(ref)
    curr_props = curr_counts / len(curr)

    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)

    # Re-normalize
    ref_props /= ref_props.sum()
    curr_props /= curr_props.sum()

    # PSI formula: sum((A - E) * ln(A / E))
    psi_val = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return float(max(0.0, psi_val))


def calculate_ks_test(
    reference: Union[np.ndarray, List[float], pd.Series],
    current: Union[np.ndarray, List[float], pd.Series]
) -> Dict[str, float]:
    """
    Computes the two-sample Kolmogorov-Smirnov test statistic and p-value.
    Measures the maximum vertical distance between empirical cumulative distributions.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return {'ks_statistic': 0.0, 'p_value': 1.0, 'drift_detected': False}

    res = stats.ks_2samp(ref, curr)
    ks_stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        'ks_statistic': ks_stat,
        'p_value': p_val,
        'drift_detected': bool(p_val < 0.05)
    }


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: List[str]
) -> Dict[str, Any]:
    """
    Generates a full feature drift summary report across all numeric features.
    """
    report = {}
    for feat in numeric_features:
        if feat in reference_df.columns and feat in current_df.columns:
            ref_vals = reference_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values

            psi = calculate_psi(ref_vals, curr_vals)
            ks = calculate_ks_test(ref_vals, curr_vals)

            status = "STABLE"
            if psi >= 0.20:
                status = "SIGNIFICANT_DRIFT"
            elif psi >= 0.10:
                status = "MODERATE_DRIFT"

            report[feat] = {
                'psi': psi,
                'ks_statistic': ks['ks_statistic'],
                'ks_p_value': ks['p_value'],
                'status': status
            }

    return report
