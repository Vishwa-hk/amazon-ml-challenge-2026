"""
Amazon ML Challenge 2026 — baseline pipeline scaffold
=====================================================
Goal: get a VALID, scored submission on the leaderboard within the first hour
of Day 1, then iterate. Problem-agnostic on purpose. On Day 1 you only touch the
blocks marked  # >>> TODO(DAY1) <<<  — everything else already runs.

Design assumption (matches past editions): a tabular/text problem with a single
target column and one scoring metric. If it turns out to be an IMAGE/multimodal
problem (like 2024), keep the CONFIG + CV + submission machinery and swap the
feature block for an image-embedding path (see README).

Run:
    python baseline_pipeline.py

Dependencies (all on Colab / SageMaker by default):
    pandas, numpy, scikit-learn.  (Optionally lightgbm — auto-falls back if absent.)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------
# CONFIG  — edit these first on Day 1
# ----------------------------------------------------------------------------
class CFG:
    data_dir      = "."                 # where train/test live
    train_file    = "train.csv"         # >>> TODO(DAY1) <<< real filename
    test_file     = "test.csv"          # >>> TODO(DAY1) <<< real filename
    sample_sub    = "sample_submission.csv"  # if provided — defines the output format
    out_file      = "submission.csv"

    id_col        = "id"                # >>> TODO(DAY1) <<< id column name
    target_col    = "target"           # >>> TODO(DAY1) <<< label column name
    task          = "classification"   # "classification" | "regression"  >>> TODO(DAY1) <<<

    text_cols     = []                 # >>> TODO(DAY1) <<< e.g. ["title", "description", "bullet_points"]
    num_cols      = []                 # >>> TODO(DAY1) <<< numeric feature columns
    cat_cols      = []                 # >>> TODO(DAY1) <<< low-cardinality categoricals

    n_folds       = 5
    seed          = 42
    tfidf_max_features = 50000         # dial down if RAM/time is tight


# ----------------------------------------------------------------------------
# METRIC  — wire in the EXACT competition metric on Day 1. This is what wins.
# ----------------------------------------------------------------------------
def competition_metric(y_true, y_pred) -> float:
    """
    >>> TODO(DAY1) <<< Replace with the exact metric from the problem statement.
    Common cases:
      from sklearn.metrics import f1_score      # F1 (often macro): f1_score(y_true, y_pred, average="macro")
      from sklearn.metrics import accuracy_score
      from sklearn.metrics import mean_absolute_error            # MAE
      # SMAPE / MAPE: define by hand if they ask for it.
    Note whether HIGHER is better (F1/acc) or LOWER (MAE) — used only for logging here.
    """
    from sklearn.metrics import f1_score, mean_absolute_error
    if CFG.task == "classification":
        return f1_score(y_true, y_pred, average="macro")
    return -mean_absolute_error(y_true, y_pred)  # negate so "higher = better" holds in logs


# ----------------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(os.path.join(CFG.data_dir, CFG.train_file))
    test  = pd.read_csv(os.path.join(CFG.data_dir, CFG.test_file))
    print(f"train {train.shape} | test {test.shape}")
    print("columns:", list(train.columns))
    print(train.head(3).to_string())
    # Sanity: target present in train, absent in test
    assert CFG.target_col in train.columns, f"target '{CFG.target_col}' not in train — fix CFG"
    return train, test


# ----------------------------------------------------------------------------
# FEATURES  — cheap, robust baseline: TF-IDF on text + imputed/scaled numerics.
# Friend A owns growing this from EDA. Keep it FAST for the first submission.
# ----------------------------------------------------------------------------
def build_features(train: pd.DataFrame, test: pd.DataFrame):
    parts_tr, parts_te = [], []

    # --- text -> TF-IDF (concatenate all text cols into one field) ---
    if CFG.text_cols:
        tr_text = train[CFG.text_cols].fillna("").agg(" ".join, axis=1)
        te_text = test[CFG.text_cols].fillna("").agg(" ".join, axis=1)
        tfidf = TfidfVectorizer(max_features=CFG.tfidf_max_features,
                                ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        parts_tr.append(tfidf.fit_transform(tr_text))
        parts_te.append(tfidf.transform(te_text))
        print(f"tfidf: {parts_tr[-1].shape[1]} features")

    # --- numerics -> impute + scale ---
    if CFG.num_cols:
        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler(with_mean=False)  # with_mean=False keeps sparse-friendly
        tr_num = sc.fit_transform(imp.fit_transform(train[CFG.num_cols]))
        te_num = sc.transform(imp.transform(test[CFG.num_cols]))
        from scipy.sparse import csr_matrix
        parts_tr.append(csr_matrix(tr_num)); parts_te.append(csr_matrix(te_num))

    # --- categoricals -> one-hot (only if low cardinality; else target-encode later) ---
    if CFG.cat_cols:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(handle_unknown="ignore")
        parts_tr.append(ohe.fit_transform(train[CFG.cat_cols].astype(str)))
        parts_te.append(ohe.transform(test[CFG.cat_cols].astype(str)))

    if not parts_tr:
        raise ValueError("No features configured — set text_cols / num_cols / cat_cols in CFG.")

    from scipy.sparse import hstack
    X  = hstack(parts_tr).tocsr()
    Xt = hstack(parts_te).tocsr()
    print(f"feature matrix: train {X.shape} | test {Xt.shape}")
    return X, Xt


# ----------------------------------------------------------------------------
# MODEL  — LightGBM if available, else a fast sklearn linear baseline.
# Friend B owns making this better once a valid submission exists.
# ----------------------------------------------------------------------------
def make_model():
    try:
        import lightgbm as lgb
        if CFG.task == "classification":
            return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                      num_leaves=63, subsample=0.8,
                                      colsample_bytree=0.8, random_state=CFG.seed)
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8,
                                 colsample_bytree=0.8, random_state=CFG.seed)
    except ImportError:
        # Fast, scales to big sparse text — a strong text baseline in its own right.
        if CFG.task == "classification":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, n_jobs=-1)
        from sklearn.linear_model import Ridge
        return Ridge()


# ----------------------------------------------------------------------------
# CV + PREDICT  — trustworthy local CV so we don't overfit the public leaderboard.
# ----------------------------------------------------------------------------
def cross_val_and_predict(X, y, Xt):
    if CFG.task == "classification":
        folds = StratifiedKFold(CFG.n_folds, shuffle=True, random_state=CFG.seed)
        splitter = folds.split(X, y)
    else:
        folds = KFold(CFG.n_folds, shuffle=True, random_state=CFG.seed)
        splitter = folds.split(X)

    test_preds = np.zeros((Xt.shape[0], CFG.n_folds), dtype=float)
    scores = []
    for i, (tr_idx, va_idx) in enumerate(splitter):
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        va_pred = model.predict(X[va_idx])
        score = competition_metric(y[va_idx], va_pred)
        scores.append(score)
        test_preds[:, i] = model.predict(Xt)
        print(f"  fold {i}: local metric = {score:.5f}")

    print(f"CV mean = {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    # Aggregate fold test predictions:
    if CFG.task == "classification":
        # majority vote across folds
        from scipy.stats import mode
        final = mode(test_preds, axis=1, keepdims=False).mode
    else:
        final = test_preds.mean(axis=1)
    return final


# ----------------------------------------------------------------------------
# SUBMISSION  — match the sample_submission format EXACTLY (a format mismatch
# is the #1 reason a submission silently scores 0).
# ----------------------------------------------------------------------------
def write_submission(test: pd.DataFrame, preds):
    sample_path = os.path.join(CFG.data_dir, CFG.sample_sub)
    if os.path.exists(sample_path):
        sub = pd.read_csv(sample_path)
        pred_col = [c for c in sub.columns if c != CFG.id_col][0]
        sub[CFG.id_col] = test[CFG.id_col].values
        sub[pred_col]   = preds
    else:
        sub = pd.DataFrame({CFG.id_col: test[CFG.id_col].values, CFG.target_col: preds})
    sub.to_csv(CFG.out_file, index=False)
    print(f"wrote {CFG.out_file}  {sub.shape}")
    print(sub.head().to_string())


# ----------------------------------------------------------------------------
def main():
    np.random.seed(CFG.seed)
    train, test = load_data()
    y = train[CFG.target_col].values
    X, Xt = build_features(train, test)
    preds = cross_val_and_predict(X, y, Xt)
    write_submission(test, preds)
    print("\nDone. Upload submission.csv, confirm it scores, THEN start improving.")


if __name__ == "__main__":
    main()
