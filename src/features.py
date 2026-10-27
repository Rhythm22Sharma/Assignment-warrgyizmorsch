"""
Feature engineering module.
Contains:
1. Base row-level features:
   - amt, hour, dow, age, haversine_dist, city_pop, gender, state, category, secs_since_prev
2. Past-only experiment features (strictly computed with shift/expanding, no future info):
   - past_amt_mean, past_amt_std, amt_zscore_past, past_txn_count,
     txns_last_24h, past_night_share, past_cat_entropy
3. Utility for single-record feature extraction for production prediction.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


# Static user-level fingerprint features (used to demonstrate leakage)
STATIC_USER_FEATURES = ['gender', 'state', 'city_pop', 'age', 'lat', 'long']

# Base row-level features
BASE_NUMERIC_FEATURES = ['amt', 'hour', 'dow', 'age', 'haversine_dist', 'city_pop', 'secs_since_prev']
BASE_CATEGORICAL_FEATURES = ['gender', 'state', 'category']
BASE_FEATURES = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES

# Past-only experimental features
EXPERIMENT_NUMERIC_FEATURES = [
    'past_txn_count',
    'past_amt_mean',
    'past_amt_std',
    'amt_zscore_past',
    'txns_last_24h',
    'past_night_share',
    'past_cat_entropy'
]
ALL_EXPERIMENT_FEATURES = BASE_FEATURES + EXPERIMENT_NUMERIC_FEATURES


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Computes Haversine great-circle distance in kilometers between two sets of coordinates.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (R * c).astype(np.float32)


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts row-level features. Expects trans_date_trans_time, dob, lat, long, merch_lat, merch_long, amt.
    """
    out = df.copy()

    # Ensure datetime format
    if not np.issubdtype(out['trans_date_trans_time'].dtype, np.datetime64):
        out['trans_date_trans_time'] = pd.to_datetime(out['trans_date_trans_time'])
    if not np.issubdtype(out['dob'].dtype, np.datetime64):
        out['dob'] = pd.to_datetime(out['dob'])

    out['hour'] = out['trans_date_trans_time'].dt.hour.astype(int)
    out['dow'] = out['trans_date_trans_time'].dt.dayofweek.astype(int)
    out['age'] = ((out['trans_date_trans_time'] - out['dob']).dt.days / 365.25).astype(np.float32)

    # Home-to-merchant Haversine distance
    out['haversine_dist'] = haversine_np(
        out['lat'].values,
        out['long'].values,
        out['merch_lat'].values,
        out['merch_long'].values
    )

    # Time since previous transaction per user
    if 'user_id' in out.columns:
        out = out.sort_values(['user_id', 'trans_date_trans_time']).reset_index(drop=True)
        time_diff = out.groupby('user_id')['trans_date_trans_time'].diff().dt.total_seconds()
        out['secs_since_prev'] = time_diff.fillna(0.0).astype(np.float32)
    else:
        out['secs_since_prev'] = 0.0

    return out


def compute_past_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes past-only historical features strictly using shift/expanding windows:
    - past_txn_count
    - past_amt_mean & past_amt_std
    - amt_zscore_past
    - txns_last_24h
    - past_night_share
    - past_cat_entropy
    """
    out = compute_base_features(df)
    out = out.sort_values(['user_id', 'trans_date_trans_time']).reset_index(drop=True)

    # 1. Past txn count so far
    out['past_txn_count'] = out.groupby('user_id').cumcount().astype(np.float32)

    # 2. Expanding mean & std of amt (shifted by 1 so strictly past only)
    exp_mean = out.groupby('user_id')['amt'].transform(lambda s: s.shift(1).expanding().mean())
    exp_std = out.groupby('user_id')['amt'].transform(lambda s: s.shift(1).expanding().std()).fillna(0.0)

    out['past_amt_mean'] = exp_mean.fillna(out['amt']).astype(np.float32)
    out['past_amt_std'] = exp_std.astype(np.float32)
    out['amt_zscore_past'] = ((out['amt'] - out['past_amt_mean']) / (out['past_amt_std'] + 1e-5)).clip(-10.0, 10.0).astype(np.float32)

    # 3. Share of night transactions so far (22h - 6h)
    is_night = ((out['hour'] < 6) | (out['hour'] >= 22)).astype(np.float32)
    out['past_night_share'] = is_night.groupby(out['user_id']).transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(0.0).astype(np.float32)

    # 4. Txns in last 24h (closed='left' guarantees past-only, excludes current txn)
    out['txns_last_24h'] = out.groupby('user_id', group_keys=False).apply(
        lambda g: pd.Series(
            g.set_index('trans_date_trans_time').rolling('24h', closed='left')['amt'].count().values,
            index=g.index
        ),
        include_groups=False
    ).fillna(0.0).astype(np.float32)

    # 5. Category entropy so far
    # Vectorized category dummy cumulative frequencies
    cat_dummies = pd.get_dummies(out['category']).astype(np.float32)
    shifted_dummies = cat_dummies.groupby(out['user_id']).shift(1).fillna(0.0)
    cum_counts = shifted_dummies.groupby(out['user_id']).cumsum()
    total_past = cum_counts.sum(axis=1).values

    probs = cum_counts.values / np.maximum(total_past[:, None], 1e-9)
    log_p = np.where(probs > 0, np.log(probs + 1e-12), 0.0)
    ent = -np.sum(probs * log_p, axis=1)
    ent = np.where(total_past == 0, 0.0, ent)
    out['past_cat_entropy'] = ent.astype(np.float32)

    return out


def prepare_feature_record(record: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms a single incoming production dictionary record into the model's feature space.
    Provides robust fallbacks for missing historical or contextual fields.
    """
    rec = dict(record)

    # Parse datetimes with None-safety
    raw_time = rec.get('trans_date_trans_time')
    if raw_time is None or pd.isna(raw_time):
        txn_time = pd.Timestamp.now()
    else:
        txn_time = pd.to_datetime(raw_time)

    raw_dob = rec.get('dob')
    if raw_dob is None or pd.isna(raw_dob):
        dob = txn_time - pd.Timedelta(days=365.25 * 35)
    else:
        dob = pd.to_datetime(raw_dob)

    amt = float(rec.get('amt') if rec.get('amt') is not None else 0.0)
    lat = float(rec.get('lat') if rec.get('lat') is not None else 35.0)
    long = float(rec.get('long') if rec.get('long') is not None else -95.0)
    merch_lat = float(rec.get('merch_lat') if rec.get('merch_lat') is not None else lat)
    merch_long = float(rec.get('merch_long') if rec.get('merch_long') is not None else long)

    haversine_dist = float(haversine_np(np.array([lat]), np.array([long]), np.array([merch_lat]), np.array([merch_long]))[0])
    age = float((txn_time - dob).days / 365.25)
    hour = int(txn_time.hour)
    dow = int(txn_time.dayofweek)

    past_mean = rec.get('past_amt_mean')
    past_mean = float(past_mean) if past_mean is not None else amt

    past_std = rec.get('past_amt_std')
    past_std = float(past_std) if past_std is not None else 0.0

    zscore = rec.get('amt_zscore_past')
    zscore = float(zscore) if zscore is not None else 0.0

    txns_24h = rec.get('txns_last_24h')
    txns_24h = float(txns_24h) if txns_24h is not None else 0.0

    past_txns = rec.get('past_txn_count')
    past_txns = float(past_txns) if past_txns is not None else 0.0

    past_night = rec.get('past_night_share')
    past_night = float(past_night) if past_night is not None else (1.0 if (hour < 6 or hour >= 22) else 0.0)

    past_entropy = rec.get('past_cat_entropy')
    past_entropy = float(past_entropy) if past_entropy is not None else 0.0

    row = {
        'amt': amt,
        'hour': hour,
        'dow': dow,
        'age': age,
        'haversine_dist': haversine_dist,
        'city_pop': int(rec.get('city_pop', 10000) if rec.get('city_pop') is not None else 10000),
        'gender': str(rec.get('gender', 'M') if rec.get('gender') is not None else 'M'),
        'state': str(rec.get('state', 'TX') if rec.get('state') is not None else 'TX'),
        'category': str(rec.get('category', 'misc_net') if rec.get('category') is not None else 'misc_net'),
        'secs_since_prev': float(rec.get('secs_since_prev', 0.0) if rec.get('secs_since_prev') is not None else 0.0),
        'lat': lat,
        'long': long,
        # Experimental features if present or defaulted
        'past_txn_count': past_txns,
        'past_amt_mean': past_mean,
        'past_amt_std': past_std,
        'amt_zscore_past': zscore,
        'txns_last_24h': txns_24h,
        'past_night_share': past_night,
        'past_cat_entropy': past_entropy
    }
    return pd.DataFrame([row])


if __name__ == '__main__':
    print("=== Testing Feature Engineering on Sample Data ===")
    from src.data import load_and_prepare_data
    df, _ = load_and_prepare_data(data_dir='data', cap_rows=True)

    print("Computing base row-level features...")
    df_base = compute_base_features(df)
    print("Base features sample:")
    print(df_base[BASE_FEATURES].head(3))

    print("\nComputing past-only experimental features...")
    df_past = compute_past_features(df)
    print("Experimental features sample:")
    print(df_past[ALL_EXPERIMENT_FEATURES].head(3))

    # Verify HARD RULE: is_fraud is NEVER in features
    assert 'is_fraud' not in BASE_FEATURES, "VIOLATION: is_fraud in BASE_FEATURES"
    assert 'is_fraud' not in ALL_EXPERIMENT_FEATURES, "VIOLATION: is_fraud in ALL_EXPERIMENT_FEATURES"
    assert 'fraud_rate' not in BASE_FEATURES, "VIOLATION: fraud_rate in BASE_FEATURES"
    assert 'fraud_rate' not in ALL_EXPERIMENT_FEATURES, "VIOLATION: fraud_rate in ALL_EXPERIMENT_FEATURES"
    print("\nHARD RULE VERIFIED: Neither 'is_fraud' nor 'fraud_rate' is present in feature sets.")
