"""
Exploratory Data Analysis (EDA) module.
Generates:
1. Class distribution analysis (user-level & transaction-level)
2. Data quality analysis (missingness, duplicates, outliers in amt/distances)
3. Intra-user correlation (ICC and within vs between-user variance)
4. Leak demonstration visualizations saved to reports/figures/
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data import load_and_prepare_data
from src.features import compute_base_features, STATIC_USER_FEATURES
from src.split import run_split_comparison_and_leak_demo

sns.set_theme(style='whitegrid', font_scale=1.1)
FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, thresh: dict):
    """Plot 1: User-level and Transaction-level Risk Class Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # User level counts
    user_counts = thresh.get('class_counts', thresh.get('counts', {}))
    total_users = thresh.get('total_retained_users', thresh.get('total_users', sum(user_counts.values())))
    classes = ['low', 'medium', 'high']
    u_vals = [user_counts.get(c, 0) for c in classes]
    colors = ['#2b5c8f', '#e67e22', '#c0392b']

    ax1.bar(classes, u_vals, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title(f'User-Level Risk Distribution (N={total_users})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Users')
    ax1.set_xlabel('Assigned Risk Class')
    for i, v in enumerate(u_vals):
        pct = (v / total_users) * 100
        ax1.text(i, v + 15, f"{v}\n({pct:.1f}%)", ha='center', fontweight='bold')
    ax1.set_ylim(0, max(u_vals) * 1.2)

    # Transaction level counts
    tx_counts = df['risk_level'].value_counts()[classes]
    t_vals = tx_counts.values
    ax2.bar(classes, t_vals, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title(f'Transaction-Level Distribution (N={len(df):,})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_xlabel('Assigned Risk Class')
    for i, v in enumerate(t_vals):
        pct = (v / len(df)) * 100
        ax2.text(i, v + 2500, f"{v:,}\n({pct:.1f}%)", ha='center', fontweight='bold')
    ax2.set_ylim(0, max(t_vals) * 1.2)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'class_distribution.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def analyze_missing_duplicates_outliers(df: pd.DataFrame):
    """Plot 2: Data Quality & Outlier Analysis for Amount and Distance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Missing values
    missing = df.isnull().sum()
    axes[0, 0].bar(missing.index.astype(str), missing.values, color='#34495e')
    axes[0, 0].set_title('Missing Values per Feature', fontweight='bold')
    axes[0, 0].set_xticklabels(missing.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Missing Count')

    # Duplicates check
    dups = df.duplicated(subset=['user_id', 'trans_date_trans_time']).sum()

    # Amount Distribution (Log scale)
    sns.histplot(df['amt'], bins=60, log_scale=True, ax=axes[0, 1], color='#2980b9', kde=True)
    axes[0, 1].set_title('Transaction Amount ($) Log Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Amount (USD, Log Scale)')

    # Haversine distance boxplot by risk class
    sns.boxplot(data=df, x='risk_level', y='haversine_dist', order=['low', 'medium', 'high'],
                palette=['#2b5c8f', '#e67e22', '#c0392b'], ax=axes[1, 0])
    axes[1, 0].set_title('Merchant-to-Home Haversine Distance by Risk Class', fontweight='bold')
    axes[1, 0].set_ylabel('Distance (km)')

    # Amount by risk class
    sns.boxplot(data=df, x='risk_level', y='amt', order=['low', 'medium', 'high'],
                palette=['#2b5c8f', '#e67e22', '#c0392b'], ax=axes[1, 1])
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Transaction Amount by Risk Class (Log Scale)', fontweight='bold')
    axes[1, 1].set_ylabel('Amount ($)')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'missing_outliers.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    return {
        'total_missing': int(missing.sum()),
        'duplicate_transactions': int(dups),
        'amt_quantiles': df['amt'].quantile([0.5, 0.95, 0.99, 0.999]).to_dict(),
        'amt_max': float(df['amt'].max()),
        'distance_max': float(df['haversine_dist'].max())
    }


def compute_icc(df: pd.DataFrame, group_col: str, val_col: str) -> float:
    """
    Computes Intraclass Correlation Coefficient (ICC(1)) for intra-user correlation:
    ICC(1) = (MSB - MSW) / (MSB + (k - 1) * MSW)
    where MSB is Mean Square Between groups, MSW is Mean Square Within groups,
    and k is the average group size.
    """
    groups = df.groupby(group_col)[val_col].apply(list).values
    # Filter groups with length > 1
    groups = [np.array(g) for g in groups if len(g) > 1]
    k_vals = np.array([len(g) for g in groups])
    N = sum(k_vals)
    K = len(groups)

    # Average group size
    k_avg = (N - np.sum(k_vals ** 2) / N) / (K - 1)

    grand_mean = np.mean(np.concatenate(groups))
    group_means = np.array([np.mean(g) for g in groups])

    # Sum of squares between
    SSB = np.sum(k_vals * (group_means - grand_mean) ** 2)
    DFB = K - 1
    MSB = SSB / DFB

    # Sum of squares within
    SSW = np.sum([np.sum((g - np.mean(g)) ** 2) for g in groups])
    DFW = N - K
    MSW = SSW / DFW

    icc = (MSB - MSW) / (MSB + (k_avg - 1) * MSW)
    return float(icc)


def plot_intra_user_variance(df: pd.DataFrame):
    """Plot 3: Intra-User Correlation & Within vs Between User Variance."""
    test_cols = ['amt', 'hour', 'haversine_dist', 'secs_since_prev', 'city_pop', 'age']
    icc_results = {}
    for col in test_cols:
        icc_results[col] = compute_icc(df, 'user_id', col)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#16a085', '#16a085', '#16a085', '#16a085', '#e67e22', '#e67e22']
    bars = ax.bar(list(icc_results.keys()), list(icc_results.values()), color=colors, edgecolor='black', alpha=0.85)
    ax.set_title('Intra-User Correlation (ICC(1)): Behavioral (<0.08) vs Static (~1.00)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ICC (Intraclass Correlation)')
    ax.set_ylim(0, 1.2)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'intra_user_variance.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return icc_results


def plot_leak_demonstration(split_results_df: pd.DataFrame):
    """Plot 4: Performance Drop from Random Split to User Split and Static Leak."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Metric comparison across experiments
    experiments = split_results_df['experiment'].tolist()
    # Shorten labels for clean plot
    short_labels = [
        '(a) Random Row Split',
        '(b) User Split',
        'Leak: Static on Random',
        'Leak: Static on User'
    ]

    pr_auc = split_results_df['pr_auc_high'].values
    recall = split_results_df['high_recall'].values

    x = np.arange(len(short_labels))
    width = 0.35

    ax1.bar(x - width/2, pr_auc, width, label='PR-AUC (High Risk)', color='#3498db', edgecolor='black')
    ax1.bar(x + width/2, recall, width, label='High-Risk Recall', color='#e74c3c', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=25, ha='right', fontsize=10)
    ax1.set_ylabel('Score')
    ax1.set_title('Data Leakage Impact on High-Risk Detection', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')

    # F1 vs Accuracy
    acc = split_results_df['accuracy'].values
    f1 = split_results_df['macro_f1'].values

    ax2.bar(x - width/2, acc, width, label='Overall Accuracy', color='#2ecc71', edgecolor='black')
    ax2.bar(x + width/2, f1, width, label='Macro F1', color='#9b59b6', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=25, ha='right', fontsize=10)
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy vs Macro F1 Under Distribution Shifts', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'leak_demonstration.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def run_full_eda():
    """Executes full exploratory data analysis and generates all artifacts."""
    print("=== Running Exploratory Data Analysis (EDA) ===")
    df, thresh = load_and_prepare_data(data_dir='data', cap_rows=True)
    df = compute_base_features(df)

    print("\n1. Generating Class Distribution Figures...")
    plot_class_distribution(df, thresh)

    print("\n2. Checking Missingness, Duplicates, and Outliers...")
    dq = analyze_missing_duplicates_outliers(df)
    print("Data Quality Summary:", dq)

    print("\n3. Computing Intra-User Correlation (ICC)...")
    icc_stats = plot_intra_user_variance(df)
    print("Intra-User Correlation Stats:", icc_stats)

    print("\n4. Running Split Comparison & Leak Demonstration Plot...")
    split_df = run_split_comparison_and_leak_demo(df)
    plot_leak_demonstration(split_df)

    print("\n=== EDA Pipeline Successfully Completed ===")
    return dq, icc_stats, split_df


if __name__ == '__main__':
    run_full_eda()
