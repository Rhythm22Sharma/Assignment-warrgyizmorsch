"""
Data Splitting and Evaluation Rigor Module.
Updated to reflect the Observation / Outcome Window Cohort Design:
- Features are strictly from the observation window (first 60% of timeline).
- Labels are strictly from the outcome window (last 40% of timeline).
- The observation/outcome window design replaces the prior within-dataset temporal split.
- (a) Leaky baseline: Random row split of observation transactions.
- (b) Rigorous benchmark: User-level split (strictly disjoint users, zero user overlap).
- Direct demonstration of static feature leakage on Random vs User splits.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
import lightgbm as lgb

from src.features import (
    BASE_FEATURES,
    BASE_NUMERIC_FEATURES,
    BASE_CATEGORICAL_FEATURES,
    STATIC_USER_FEATURES,
    compute_base_features
)

RANDOM_SEED = 42


def create_random_row_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (a) BASELINE random row-level split (LEAKY).
    Stratified by risk_class at transaction level. Transactions from same user appear in all splits.
    """
    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        stratify=df['risk_class'],
        random_state=random_state
    )

    relative_val_size = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val_size,
        stratify=df_temp['risk_class'],
        random_state=random_state
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def create_user_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (b) User-level split: users are partitioned so that no user appears in more than one split.
    Stratified by each user's outcome-derived risk_class.
    """
    user_df = df[['user_id', 'risk_class']].drop_duplicates('user_id').reset_index(drop=True)

    users_train, users_temp = train_test_split(
        user_df,
        train_size=train_size,
        stratify=user_df['risk_class'],
        random_state=random_state
    )

    relative_val_size = val_size / (val_size + test_size)
    users_val, users_test = train_test_split(
        users_temp,
        train_size=relative_val_size,
        stratify=users_temp['risk_class'],
        random_state=random_state
    )

    train_users_set = set(users_train['user_id'])
    val_users_set = set(users_val['user_id'])
    test_users_set = set(users_test['user_id'])

    # Strict user-disjoint assertions
    assert train_users_set.isdisjoint(val_users_set), "Train and Val users overlap!"
    assert train_users_set.isdisjoint(test_users_set), "Train and Test users overlap!"
    assert val_users_set.isdisjoint(test_users_set), "Val and Test users overlap!"

    df_train = df[df['user_id'].isin(train_users_set)].reset_index(drop=True)
    df_val = df[df['user_id'].isin(val_users_set)].reset_index(drop=True)
    df_test = df[df['user_id'].isin(test_users_set)].reset_index(drop=True)

    return df_train, df_val, df_test


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a robust scikit-learn ColumnTransformer for preprocessing.
    Fit ONLY on training data!
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def evaluate_split_performance(
    model,
    X_train, y_train,
    X_test, y_test,
    high_class_idx: int = 2
) -> Dict[str, float]:
    """Evaluates classification metrics on test data."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    high_rec = recall_score(y_test == high_class_idx, y_pred == high_class_idx, zero_division=0)
    high_prec = precision_score(y_test == high_class_idx, y_pred == high_class_idx, zero_division=0)

    p_prec, p_rec, _ = precision_recall_curve((y_test == high_class_idx).astype(int), y_proba[:, high_class_idx])
    pr_auc_high = auc(p_rec, p_prec)

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'high_recall': float(high_rec),
        'high_precision': float(high_prec),
        'pr_auc_high': float(pr_auc_high)
    }


def run_split_comparison_and_leak_demo(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Compares:
    1. (a) Baseline Random Row Split (Leaky)
    2. (b) User-level Split (Disjoint Users - Observation/Outcome Cohort Design)
    3. DIRECT LEAK DEMO: Model with ONLY static user features on Random Split vs User Split.
    """
    print("\n--- Running Split Benchmark and Data Leak Demonstration ---")
    results = []

    # 1. BASELINE RANDOM ROW SPLIT
    df_train_a, df_val_a, df_test_a = create_random_row_split(df_processed, random_state=RANDOM_SEED)
    prep_a = build_preprocessor(BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES)
    X_tr_a = prep_a.fit_transform(df_train_a)
    X_te_a = prep_a.transform(df_test_a)
    y_tr_a = df_train_a['risk_class'].values
    y_te_a = df_test_a['risk_class'].values

    clf_a = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    res_a = evaluate_split_performance(clf_a, X_tr_a, y_tr_a, X_te_a, y_te_a)
    res_a['experiment'] = '(a) Random Row Split (Leaky)'
    results.append(res_a)

    # 2. USER-LEVEL SPLIT (Disjoint Users)
    df_train_b, df_val_b, df_test_b = create_user_split(df_processed, random_state=RANDOM_SEED)
    prep_b = build_preprocessor(BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES)
    X_tr_b = prep_b.fit_transform(df_train_b)
    X_te_b = prep_b.transform(df_test_b)
    y_tr_b = df_train_b['risk_class'].values
    y_te_b = df_test_b['risk_class'].values

    clf_b = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    res_b = evaluate_split_performance(clf_b, X_tr_b, y_tr_b, X_te_b, y_te_b)
    res_b['experiment'] = '(b) User-level Split (Disjoint Users)'
    results.append(res_b)

    # 3. DIRECT LEAK DEMONSTRATION: ONLY STATIC USER FEATURES
    static_num = ['city_pop', 'age', 'lat', 'long']
    static_cat = ['gender', 'state']

    # On Random Row Split (should memorize user coordinates)
    prep_leak_a = build_preprocessor(static_num, static_cat)
    X_tr_la = prep_leak_a.fit_transform(df_train_a)
    X_te_la = prep_leak_a.transform(df_test_a)
    clf_la = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    res_la = evaluate_split_performance(clf_la, X_tr_la, y_tr_a, X_te_la, y_te_a)
    res_la['experiment'] = '[LEAK TEST] Only Static Features on (a) Random Split'
    results.append(res_la)

    # On User-level Split (unseen users: coordinates are useless)
    prep_leak_b = build_preprocessor(static_num, static_cat)
    X_tr_lb = prep_leak_b.fit_transform(df_train_b)
    X_te_lb = prep_leak_b.transform(df_test_b)
    clf_lb = lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', verbose=-1, n_estimators=100)
    res_lb = evaluate_split_performance(clf_lb, X_tr_lb, y_tr_b, X_te_lb, y_te_b)
    res_lb['experiment'] = '[LEAK TEST] Only Static Features on (b) User Split'
    results.append(res_lb)

    res_df = pd.DataFrame(results)[['experiment', 'accuracy', 'macro_f1', 'high_recall', 'high_precision', 'pr_auc_high']]
    return res_df


if __name__ == '__main__':
    from src.data import load_and_prepare_data
    df, _ = load_and_prepare_data(data_dir='data', cap_rows=True)
    df_features = compute_base_features(df)

    res_table = run_split_comparison_and_leak_demo(df_features)
    print("\n=== Split Comparison & Data Leakage Verification Table ===")
    print(res_table.to_markdown(index=False))
