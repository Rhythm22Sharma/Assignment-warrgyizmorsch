"""
Data loading, Observation/Outcome Window Cohort Design, and Labeling Module.

Temporal Framing (Observation / Outcome Design):
- Timeline: 2019-01-01 to 2020-12-31 (~730 days)
- Observation Window (first 60% of timeline): rows strictly before 2020-03-14 14:23:51.
  Features are computed ONLY from observation-window rows.
- Outcome Window (last 40% of timeline): rows at or after 2020-03-14 14:23:51.
  User risk labels are computed ONLY from fraud rates in the outcome window.
- Quality Filter: Users with < 10 outcome-window transactions are dropped.
- Tie-Preserving Thresholds:
  Users with identical rates always receive the identical risk class (never rank-based).
  * Low Risk: out_rate == 0.0 (~63.8%, 579 users)
  * Medium Risk: 0.0 < out_rate < 0.020 (~26.2%, 238 users)
  * High Risk: out_rate >= 0.020 (~10.0%, 91 users)
- Speed Optimization: Observation rows capped at 150 random rows per user (~136k rows).
- HARD RULE: is_fraud and any outcome-window data are NEVER features.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

RANDOM_SEED = 42

NEEDED_COLUMNS = [
    'trans_date_trans_time',
    'cc_num',
    'amt',
    'category',
    'gender',
    'state',
    'city_pop',
    'dob',
    'lat',
    'long',
    'merch_lat',
    'merch_long',
    'is_fraud'
]

COMPACT_DTYPES = {
    'cc_num': 'int64',
    'amt': 'float32',
    'category': 'category',
    'gender': 'category',
    'state': 'category',
    'city_pop': 'int32',
    'dob': 'string',
    'lat': 'float32',
    'long': 'float32',
    'merch_lat': 'float32',
    'merch_long': 'float32',
    'is_fraud': 'int8'
}


def load_raw_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    Loads raw CSVs with compact dtypes, concatenates, sorts chronologically,
    and renames cc_num -> user_id.
    """
    data_path = Path(data_dir)
    train_file = data_path / 'fraudTrain.csv'
    test_file = data_path / 'fraudTest.csv'

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Expected fraudTrain.csv and fraudTest.csv in {data_dir}")

    df_train = pd.read_csv(train_file, usecols=NEEDED_COLUMNS, dtype=COMPACT_DTYPES)
    df_test = pd.read_csv(test_file, usecols=NEEDED_COLUMNS, dtype=COMPACT_DTYPES)

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.rename(columns={'cc_num': 'user_id'})
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    return df


def compute_observation_outcome_labels(
    df: pd.DataFrame,
    observation_ratio: float = 0.60,
    min_outcome_txns: int = 10,
    high_risk_threshold: float = 0.020
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Splits timeline into observation window (first 60%) and outcome window (last 40%).
    Computes per-user risk labels strictly from the outcome window.
    Filters users with < min_outcome_txns.
    Applies tie-preserving thresholds:
    - low: rate == 0.0
    - medium: 0.0 < rate < high_risk_threshold
    - high: rate >= high_risk_threshold
    """
    t_min = df['trans_date_trans_time'].min()
    t_max = df['trans_date_trans_time'].max()
    total_seconds = (t_max - t_min).total_seconds()
    cutoff_date = t_min + pd.Timedelta(seconds=total_seconds * observation_ratio)

    obs_df_full = df[df['trans_date_trans_time'] < cutoff_date]
    out_df_full = df[df['trans_date_trans_time'] >= cutoff_date]

    # Outcome window metrics per user
    out_user_stats = out_df_full.groupby('user_id')['is_fraud'].agg(
        out_txns='count',
        out_fraud='sum',
        out_rate='mean'
    ).reset_index()

    # Drop users with < min_outcome_txns
    dropped_low_txns = int((out_user_stats['out_txns'] < min_outcome_txns).sum())
    valid_outcome_users = out_user_stats[out_user_stats['out_txns'] >= min_outcome_txns].copy()

    # Users present in observation window
    obs_user_ids = set(obs_df_full['user_id'].unique())
    dropped_no_obs = int((~valid_outcome_users['user_id'].isin(obs_user_ids)).sum())

    # Retained users present in both windows
    retained_users = valid_outcome_users[valid_outcome_users['user_id'].isin(obs_user_ids)].copy()

    # Statistical distribution of outcome fraud rates
    rates = retained_users['out_rate']
    rate_stats = {
        'count': int(len(rates)),
        'min': float(rates.min()),
        'max': float(rates.max()),
        'mean': float(rates.mean()),
        'median': float(rates.median()),
        'q25': float(rates.quantile(0.25)),
        'q50': float(rates.quantile(0.50)),
        'q65': float(rates.quantile(0.65)),
        'q75': float(rates.quantile(0.75)),
        'q90': float(rates.quantile(0.90)),
        'pct_zero': float((rates == 0.0).mean() * 100.0),
        'unique_rates': int(rates.nunique())
    }

    # Boundary ties inspection
    zero_ties = int((rates == 0.0).sum())
    high_boundary_ties = int((rates == high_risk_threshold).sum())

    # Assign risk classes preserving exact ties
    conditions = [
        rates == 0.0,
        (rates > 0.0) & (rates < high_risk_threshold),
        rates >= high_risk_threshold
    ]
    choices_label = ['low', 'medium', 'high']
    choices_class = [0, 1, 2]

    retained_users['risk_level'] = np.select(conditions, choices_label, default='low')
    retained_users['risk_class'] = np.select(conditions, choices_class, default=0).astype(int)

    counts = retained_users['risk_level'].value_counts().to_dict()
    proportions = {k: v / len(retained_users) for k, v in counts.items()}

    # Check minimum high-risk users constraint
    high_user_count = counts.get('high', 0)
    if high_user_count < 30:
        raise ValueError(f"High-risk user count ({high_user_count}) is below minimum threshold of 30!")

    metadata = {
        'timeline_min': str(t_min),
        'timeline_max': str(t_max),
        'cutoff_date': str(cutoff_date),
        'observation_ratio': observation_ratio,
        'min_outcome_txns': min_outcome_txns,
        'dropped_users_low_outcome_txns': dropped_low_txns,
        'dropped_users_no_obs_txns': dropped_no_obs,
        'total_retained_users': len(retained_users),
        'rate_stats': rate_stats,
        'boundary_ties': {
            'zero_fraud_rate_ties': zero_ties,
            'high_threshold_ties': high_boundary_ties
        },
        'class_counts': counts,
        'class_proportions': proportions,
        'high_risk_threshold': high_risk_threshold
    }

    return retained_users, metadata


def sample_capped_data(
    df: pd.DataFrame,
    max_rows_per_user: int = 150,
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Caps dataset at max_rows_per_user random rows per user within the observation window.
    """
    sampled_indices = df.groupby('user_id', group_keys=False).apply(
        lambda g: g.index.to_series().sample(n=min(len(g), max_rows_per_user), random_state=random_state),
        include_groups=False
    ).values
    capped_df = df.loc[sampled_indices].sort_values('trans_date_trans_time').reset_index(drop=True)
    return capped_df


def load_and_prepare_data(
    data_dir: str = 'data',
    cap_rows: bool = True,
    max_rows_per_user: int = 150,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for data ingestion:
    1. Loads raw dataset.
    2. Derives user-level risk labels strictly from the outcome window (last 40%).
    3. Retains only observation-window transactions for valid users.
    4. Merges risk labels into observation transactions.
    5. Capped sampling of observation rows.
    """
    df = load_raw_data(data_dir=data_dir)
    user_labels_df, meta = compute_observation_outcome_labels(df)

    cutoff_date = pd.to_datetime(meta['cutoff_date'])

    # Retain ONLY observation window rows for retained users
    retained_user_ids = set(user_labels_df['user_id'])
    obs_df = df[(df['trans_date_trans_time'] < cutoff_date) & (df['user_id'].isin(retained_user_ids))].copy()

    # Merge labels
    obs_df = obs_df.merge(
        user_labels_df[['user_id', 'risk_level', 'risk_class', 'out_rate']],
        on='user_id',
        how='left'
    )

    if cap_rows:
        obs_df = sample_capped_data(obs_df, max_rows_per_user=max_rows_per_user, random_state=random_state)

    return obs_df, meta


if __name__ == '__main__':
    print("=== Loading Data with Observation / Outcome Window Design ===")
    df, meta = load_and_prepare_data(data_dir='data', cap_rows=True, random_state=RANDOM_SEED)

    print(f"\nTimeline Min: {meta['timeline_min']}")
    print(f"Timeline Max: {meta['timeline_max']}")
    print(f"60% Cutoff Date (Obs/Outcome Boundary): {meta['cutoff_date']}")
    print(f"Dropped Users (<10 Outcome Txns): {meta['dropped_users_low_outcome_txns']}")
    print(f"Dropped Users (No Obs Txns): {meta['dropped_users_no_obs_txns']}")
    print(f"Total Retained Users: {meta['total_retained_users']}")

    print("\n--- Per-User Outcome Fraud Rate Statistics ---")
    st = meta['rate_stats']
    print(f"  Count:        {st['count']}")
    print(f"  Min:          {st['min']:.6f}")
    print(f"  Max:          {st['max']:.6f}")
    print(f"  Mean:         {st['mean']:.6f}")
    print(f"  Median:       {st['median']:.6f}")
    print(f"  25th Pct:     {st['q25']:.6f}")
    print(f"  50th Pct:     {st['q50']:.6f}")
    print(f"  65th Pct:     {st['q65']:.6f}")
    print(f"  75th Pct:     {st['q75']:.6f}")
    print(f"  90th Pct:     {st['q90']:.6f}")
    print(f"  Percent Zero: {st['pct_zero']:.2f}%")
    print(f"  Unique Rates: {st['unique_rates']}")

    print("\n--- Boundary Ties Analysis (Tie Preservation) ---")
    print(f"  Zero Rate Ties (Assigned to 'low'): {meta['boundary_ties']['zero_fraud_rate_ties']}")
    print(f"  High Boundary Ties at 0.020:        {meta['boundary_ties']['high_threshold_ties']}")

    print("\n--- Final User Risk Class Counts ---")
    for k, v in meta['class_counts'].items():
        pct = meta['class_proportions'][k] * 100
        print(f"  Class '{k}': {v} users ({pct:.1f}%)")

    print(f"\nObservation Window Capped Rows: {len(df):,}")
    print("Verification passed successfully.")
