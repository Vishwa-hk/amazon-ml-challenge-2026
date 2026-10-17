import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from config import MODEL_CONFIG

FEATURE_COLS = [
    'name_exact',
    'name_norm_exact',
    'name_token_jaccard',
    'name_char_jaccard',
    'name_lev_sim',
    'name_prefix_match',
    'name_len_diff',
    'addr_token_jaccard',
    'addr_char_jaccard',
    'addr_empty',
    'postal_match',
    'number_match',
    'country_match',
    'source_pair'
]

def build_training_pairs(
    candidate_features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs labeled dataset for LightGBM with TRUE Hard Negative Mining.
    Ground truth positive pairs receive label = 1.
    Candidate non-ground-truth pairs are ranked by name/address similarity descending;
    the top-ranked non-matching pairs are mined as Hard Negatives.
    """
    if candidate_features_df.empty:
        return pd.DataFrame(), pd.Series()

    # 1. Parse ground truth positive set of (s1_id, matched_id)
    gt_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        for m in matches:
            m = m.strip()
            if m:
                gt_pairs.add((s1id, m))
                
    # 2. Assign binary label
    labels = []
    for _, row in candidate_features_df.iterrows():
        pair = (row['source1_entity_id'], row['candidate_id'])
        labels.append(1 if pair in gt_pairs else 0)
        
    candidate_features_df['label'] = labels
    
    pos_df = candidate_features_df[candidate_features_df['label'] == 1].copy()
    neg_df = candidate_features_df[candidate_features_df['label'] == 0].copy()
    
    n_pos = len(pos_df)
    if n_pos == 0:
        return candidate_features_df[FEATURE_COLS], candidate_features_df['label']
        
    # 3. Mine True Hard Negatives: Sort negatives by name similarity descending
    neg_df = neg_df.sort_values(by=['name_char_jaccard', 'name_token_jaccard'], ascending=False)
    
    max_hard_negs = MODEL_CONFIG['num_hard_negatives_per_pos'] * n_pos
    max_rand_negs = MODEL_CONFIG['num_random_negatives_per_pos'] * n_pos
    
    hard_negs = neg_df.head(max_hard_negs)
    rem_negs = neg_df.iloc[max_hard_negs:]
    
    if len(rem_negs) > max_rand_negs:
        rand_negs = rem_negs.sample(n=max_rand_negs, random_state=MODEL_CONFIG['random_state'])
    else:
        rand_negs = rem_negs
        
    sampled_neg_df = pd.concat([hard_negs, rand_negs])
    
    balanced_df = pd.concat([pos_df, sampled_neg_df]).sample(frac=1.0, random_state=MODEL_CONFIG['random_state'])
    
    return balanced_df[FEATURE_COLS], balanced_df['label']

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None
) -> lgb.LGBMClassifier:
    """Trains LightGBM classifier on extracted pairwise features."""
    params = MODEL_CONFIG['lgb_params'].copy()
    model = lgb.LGBMClassifier(**params)
    
    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)
        
    return model

def predict_match_probabilities(
    model: lgb.LGBMClassifier,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Predicts pair match probabilities for candidate pairs."""
    if features_df.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_id', 'prob'])
        
    X = features_df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    
    result_df = features_df[['source1_entity_id', 'candidate_id']].copy()
    result_df['prob'] = probs
    return result_df
