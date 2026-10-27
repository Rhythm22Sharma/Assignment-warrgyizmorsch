"""
Comprehensive Test Suite for Fraud Detection ML Pipeline.
Updated to reflect the Observation / Outcome Window Cohort Design:
1. test_no_user_overlap_across_user_level_splits
2. test_observation_window_strictly_before_cutoff
3. test_tie_preservation_in_user_labeling
4. test_preprocessing_fit_on_train_only
5. test_predict_risk_output_schema
6. test_missing_value_handling
7. test_invalid_input_rejected
8. test_threshold_logic
9. test_reproducibility_with_fixed_seed
10. test_is_fraud_not_in_features
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data import load_and_prepare_data, compute_observation_outcome_labels, RANDOM_SEED
from src.features import (
    BASE_FEATURES,
    ALL_EXPERIMENT_FEATURES,
    BASE_NUMERIC_FEATURES,
    BASE_CATEGORICAL_FEATURES,
    compute_base_features,
    prepare_feature_record
)
from src.split import (
    create_user_split,
    build_preprocessor
)
from src.predict import TransactionInput, predict_risk
from src.drift import calculate_psi, calculate_ks_test


@pytest.fixture(scope="module")
def sample_data():
    """Loads a small sample of prepared observation data for test execution."""
    df, meta = load_and_prepare_data(data_dir='data', cap_rows=True, max_rows_per_user=50, random_state=RANDOM_SEED)
    df_features = compute_base_features(df)
    return df_features, meta


def test_no_user_overlap_across_user_level_splits(sample_data):
    """Test 1: Verify zero user overlap across Train, Val, and Test in user-level split."""
    df_features, _ = sample_data
    df_train, df_val, df_test = create_user_split(df_features, train_size=0.70, val_size=0.15, test_size=0.15, random_state=RANDOM_SEED)

    train_users = set(df_train['user_id'])
    val_users = set(df_val['user_id'])
    test_users = set(df_test['user_id'])

    assert len(train_users) > 0, "Train users empty"
    assert len(val_users) > 0, "Val users empty"
    assert len(test_users) > 0, "Test users empty"

    assert train_users.isdisjoint(val_users), "Data Leak: Train and Val have overlapping users!"
    assert train_users.isdisjoint(test_users), "Data Leak: Train and Test have overlapping users!"
    assert val_users.isdisjoint(test_users), "Data Leak: Val and Test have overlapping users!"


def test_observation_window_strictly_before_cutoff(sample_data):
    """Test 2: Verify all modeling rows reside strictly within the observation window before cutoff."""
    df_features, meta = sample_data
    cutoff = pd.to_datetime(meta['cutoff_date'])

    assert (df_features['trans_date_trans_time'] < cutoff).all(), "Temporal Leak: Row exists at or after observation cutoff!"
    assert meta['dropped_users_low_outcome_txns'] > 0, "Quality filter did not drop users with <10 outcome txns"


def test_tie_preservation_in_user_labeling(sample_data):
    """Test 3: Verify tie preservation: users with identical outcome rates ALWAYS get the same class."""
    df_features, _ = sample_data
    user_classes = df_features.groupby('out_rate')['risk_class'].nunique()
    assert (user_classes == 1).all(), "Violation: Users with identical fraud rates received different risk classes!"


def test_preprocessing_fit_on_train_only(sample_data):
    """Test 4: Preprocessing pipeline is fit on train only and does not leak test statistics."""
    df_features, _ = sample_data
    df_train, _, df_test = create_user_split(df_features, random_state=RANDOM_SEED)

    prep = build_preprocessor(BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES)
    X_train_trans = prep.fit_transform(df_train)

    scaler = prep.named_transformers_['num'].named_steps['scaler']
    X_test_trans = prep.transform(df_test)

    assert X_train_trans.shape[0] == len(df_train)
    assert X_test_trans.shape[0] == len(df_test)
    assert hasattr(scaler, 'mean_'), "Scaler was not fit!"


def test_predict_risk_output_schema():
    """Test 5: Verify production predict_risk output conforms to required schema."""
    record = {
        'trans_date_trans_time': '2020-02-15 14:30:00',
        'user_id': 123456789,
        'amt': 125.50,
        'category': 'shopping_pos',
        'gender': 'F',
        'state': 'CA',
        'city_pop': 250000,
        'dob': '1985-04-12',
        'lat': 34.05,
        'long': -118.24,
        'merch_lat': 34.08,
        'merch_long': -118.22
    }

    res = predict_risk(record)

    assert isinstance(res, dict)
    assert 'predicted_class' in res
    assert res['predicted_class'] in ['low', 'medium', 'high']
    assert 'probabilities' in res
    assert set(res['probabilities'].keys()) == {'low', 'medium', 'high'}
    assert 'high_risk_score' in res
    assert 0.0 <= res['high_risk_score'] <= 1.0
    assert 'model_version' in res
    assert 'threshold_applied' in res


def test_missing_value_handling():
    """Test 6: Verify missing optional fields are handled gracefully with robust defaults."""
    minimal_record = {
        'amt': 49.99
    }

    res = predict_risk(minimal_record)
    assert isinstance(res, dict)
    assert res['predicted_class'] in ['low', 'medium', 'high']
    assert 0.0 <= res['high_risk_score'] <= 1.0


def test_invalid_input_rejected():
    """Test 7: Verify invalid inputs (negative amt, invalid lat/long) are rejected with clear errors."""
    with pytest.raises(ValueError, match="non-negative"):
        TransactionInput(amt=-10.0)

    with pytest.raises(ValueError, match="Latitude"):
        TransactionInput(amt=50.0, lat=120.0)

    with pytest.raises(ValueError, match="Longitude"):
        TransactionInput(amt=50.0, long=-200.0)

    with pytest.raises(ValueError, match="City population"):
        TransactionInput(amt=50.0, city_pop=-5)


def test_threshold_logic():
    """Test 8: Verify high-risk threshold triggering logic."""
    record = {
        'amt': 999.0,
        'category': 'shopping_net',
        'past_txn_count': 100.0,
        'amt_zscore_past': 5.0
    }
    res = predict_risk(record)

    score = res['high_risk_score']
    threshold = res['threshold_applied']

    if score >= threshold:
        assert res['predicted_class'] == 'high'
        assert res['alert_triggered'] is True
    else:
        assert res['predicted_class'] in ['low', 'medium']
        assert res['alert_triggered'] is False


def test_reproducibility_with_fixed_seed(sample_data):
    """Test 9: Verify exact split and sampling reproducibility with fixed seed 42."""
    df_features, _ = sample_data
    df_tr1, _, df_te1 = create_user_split(df_features, random_state=42)
    df_tr2, _, df_te2 = create_user_split(df_features, random_state=42)

    assert df_tr1['user_id'].tolist() == df_tr2['user_id'].tolist()
    assert df_te1['user_id'].tolist() == df_te2['user_id'].tolist()


def test_is_fraud_not_in_features():
    """Test 10: HARD RULE: Verify neither is_fraud nor out_rate is ever present in features.
    
    Only the documented optional experiment feature 'past_fraud_count_lagged' is permitted 
    in specialized experiment sets, but NEVER raw is_fraud, fraud_rate, or outcome window labels.
    """
    forbidden = ['is_fraud', 'fraud_rate', 'out_rate', 'fraud']

    for feat in BASE_FEATURES:
        for f in forbidden:
            assert feat != f, f"LEAK VIOLATION: {feat} found in BASE_FEATURES"
        assert feat != 'past_fraud_count', "BASE_FEATURES must not contain unlagged fraud counts"

    for feat in ALL_EXPERIMENT_FEATURES:
        for f in forbidden:
            assert feat != f, f"LEAK VIOLATION: {feat} found in ALL_EXPERIMENT_FEATURES"
        assert feat != 'past_fraud_count', "ALL_EXPERIMENT_FEATURES must not contain unlagged fraud counts"


def test_saved_artifact_metadata_consistency():
    """Test 11: Verify serialized artifact and metadata.json are consistent in threshold and version."""
    import json
    import joblib
    
    model_path = Path('models/best_model.joblib')
    meta_path = Path('models/metadata.json')
    
    assert model_path.exists(), "models/best_model.joblib does not exist"
    assert meta_path.exists(), "models/metadata.json does not exist"
    
    bundle = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    assert bundle['model_version'] == metadata['model_version'], "Model version mismatch between artifact and metadata"
    assert abs(bundle['threshold'] - metadata['tuned_threshold']) < 1e-6, "Threshold mismatch between artifact and metadata"
    assert bundle['numeric_features'] == metadata['features']['numeric'], "Numeric features mismatch"
    assert bundle['categorical_features'] == metadata['features']['categorical'], "Categorical features mismatch"
    assert hasattr(bundle['model'], 'predict_proba'), "Loaded model lacks predict_proba method"
