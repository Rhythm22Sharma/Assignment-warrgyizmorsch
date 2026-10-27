"""
Production Prediction API Module.
Implements:
1. Pydantic input schema validation with clear error messages.
2. Missing-value handling and feature extraction.
3. Model pipeline loading and caching.
4. predict_risk(record: dict) -> {predicted_class, probabilities, high_risk_score, model_version}.
5. Batch prediction interface.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError

from src.features import prepare_feature_record

MODEL_PATH = Path('models/best_model.joblib')
METADATA_PATH = Path('models/metadata.json')

_CACHED_PIPELINE = None
_CACHED_METADATA = None


class TransactionInput(BaseModel):
    """
    Validation schema for incoming transaction scoring requests.
    Validates types, ranges, and formats.
    """
    trans_date_trans_time: Optional[Union[datetime, str]] = Field(default=None, description="Transaction timestamp (ISO format or datetime)")
    user_id: Optional[Union[int, str]] = Field(default="UNKNOWN_USER", description="Unique card/user identifier")
    amt: float = Field(..., description="Transaction amount in USD (must be strictly >= 0)")
    category: Optional[str] = Field(default="misc_net", description="Merchant category")
    gender: Optional[str] = Field(default="M", description="Cardholder gender ('M' or 'F')")
    state: Optional[str] = Field(default="TX", description="2-letter US state code")
    city_pop: Optional[int] = Field(default=10000, description="City population (must be >= 0)")
    dob: Optional[Union[datetime, str, date]] = Field(default=None, description="Cardholder date of birth")
    lat: Optional[float] = Field(default=35.0, description="Home latitude (-90 to 90)")
    long: Optional[float] = Field(default=-95.0, description="Home longitude (-180 to 180)")
    merch_lat: Optional[float] = Field(default=35.0, description="Merchant latitude (-90 to 90)")
    merch_long: Optional[float] = Field(default=-95.0, description="Merchant longitude (-180 to 180)")

    # Optional past-only features if passed by upstream feature store
    secs_since_prev: Optional[float] = Field(default=0.0, description="Seconds since last transaction")
    past_txn_count: Optional[float] = Field(default=0.0, description="Past transaction count for user")
    past_amt_mean: Optional[float] = Field(default=None, description="Past amount mean for user")
    past_amt_std: Optional[float] = Field(default=0.0, description="Past amount std for user")
    amt_zscore_past: Optional[float] = Field(default=0.0, description="Amt z-score vs user past")
    txns_last_24h: Optional[float] = Field(default=0.0, description="Txns in last 24h")
    past_night_share: Optional[float] = Field(default=0.0, description="Past night txns share")
    past_cat_entropy: Optional[float] = Field(default=0.0, description="Past category entropy")

    @field_validator('amt')
    def validate_amt(cls, v):
        if v < 0:
            raise ValueError(f"Transaction amount must be non-negative. Received: {v}")
        if v > 1_000_000:
            raise ValueError(f"Transaction amount exceeds maximum allowable ceiling of 1,000,000. Received: {v}")
        return v

    @field_validator('lat', 'merch_lat')
    def validate_latitude(cls, v):
        if v is not None and not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90. Received: {v}")
        return v

    @field_validator('long', 'merch_long')
    def validate_longitude(cls, v):
        if v is not None and not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180. Received: {v}")
        return v

    @field_validator('city_pop')
    def validate_city_pop(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"City population must be non-negative. Received: {v}")
        return v


def get_model_pipeline():
    """Loads and caches the model pipeline bundle and metadata."""
    global _CACHED_PIPELINE, _CACHED_METADATA

    if _CACHED_PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run `python -m src.train` first.")
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)

    if _CACHED_METADATA is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                _CACHED_METADATA = json.load(f)
        else:
            _CACHED_METADATA = {'model_version': '1.0.0', 'tuned_threshold': 0.20}

    return _CACHED_PIPELINE, _CACHED_METADATA


def predict_risk(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production entry point for risk prediction on a single record.
    Returns:
    {
        'predicted_class': str ('low', 'medium', 'high'),
        'probabilities': {'low': float, 'medium': float, 'high': float},
        'high_risk_score': float,
        'model_version': str,
        'threshold_applied': float,
        'alert_triggered': bool
    }
    """
    # 1. Pydantic validation & sanitization
    try:
        validated = TransactionInput(**record)
    except ValidationError as e:
        raise ValueError(f"Input validation error: {e.errors()}") from e

    rec_dict = validated.model_dump()

    # 2. Load model
    bundle, metadata = get_model_pipeline()
    preprocessor = bundle['preprocessor']
    model = bundle['model']
    threshold = float(bundle.get('threshold', metadata.get('tuned_threshold', 0.20)))
    version = metadata.get('model_version', '1.0.0')

    # 3. Extract and align features
    df_feat = prepare_feature_record(rec_dict)
    X_trans = preprocessor.transform(df_feat)

    # 4. Predict calibrated probabilities
    probas = model.predict_proba(X_trans)[0]
    p_low = float(probas[0])
    p_med = float(probas[1])
    p_high = float(probas[2])

    # 5. Apply threshold logic for high-risk classification
    if p_high >= threshold:
        predicted_class = 'high'
        alert_triggered = True
    else:
        alert_triggered = False
        predicted_class = 'medium' if p_med >= p_low else 'low'

    return {
        'predicted_class': predicted_class,
        'probabilities': {
            'low': round(p_low, 4),
            'medium': round(p_med, 4),
            'high': round(p_high, 4)
        },
        'high_risk_score': round(p_high, 4),
        'model_version': version,
        'threshold_applied': round(threshold, 4),
        'alert_triggered': alert_triggered
    }


def predict_risk_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scores a batch of incoming transactions."""
    return [predict_risk(r) for r in records]
