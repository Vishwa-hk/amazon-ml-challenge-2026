import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.isotonic import IsotonicRegression
from config import MODEL_CONFIG

FEATURE_COLS = [
    'name_exact',
    'name_norm_exact',
    'name_agg_exact',
    'name_token_jaccard',
    'name_char_jaccard',
    'name_lev_sim',
    'name_containment',
    'containment_bidi',
    'first_token_sim',
    'last_token_sim',
    'phonetic_sim_first',
    'phonetic_sim_key',
    'name_prefix_match',
    'acronym_match',
    'name_len_diff',
    'addr_token_jaccard',
    'addr_char_jaccard',
    'addr_lev_sim',
    'addr_empty',
    'postal_match',
    'postal_missing',
    'number_match',
    'house_match',
    'house_conflict',
    'house_postal_conflict',
    'acronym_postal_match',
    'name_strong_addr_strong',
    'name_strong_postal',
    'name_strong_house_num',
    'exact_postal_similar_name',
    'exact_name_conflicting_addr',
    'exact_name_conflicting_postal',
    'country_match',
    'source_pair',
    'blocking_score',
    'blocking_rule_count',
    'exact_name_hit',
    'agg_exact_hit',
    'rare_token_hit',
    'prefix_hit',
    'postal_house_hit',
    'cross_field_hit',
    'phonetic_postal_hit',
    'residual_tfidf_hit'
]

def build_training_pairs(
    candidate_features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs labeled dataset for LightGBM with TRUE Hard Negative Mining.
    Ground truth positive pairs receive label = 1.
    Candidate non-ground-truth pairs are ranked by:
      0.45 * name_lev_sim + 0.30 * blocking_score + 0.25 * postal_match
    descending to mine the hardest confusing negatives.
    """
    if candidate_features_df.empty:
        return pd.DataFrame(), pd.Series()

    # 1. Parse ground truth positive set of (s1_id, matched_id)
    gt_pairs = set()
    for s1id, matched in zip(ground_truth_df['source1_entity_id'].values, ground_truth_df['matched_entity_ids'].values):
        if matched and pd.notna(matched):
            for m in str(matched).split(','):
                m = m.strip()
                if m:
                    gt_pairs.add((s1id, m))
                
    # 2. Assign binary label
    pair_tuples = list(zip(candidate_features_df['source1_entity_id'].values, candidate_features_df['candidate_id'].values))
    candidate_features_df['label'] = np.array([1 if p in gt_pairs else 0 for p in pair_tuples], dtype=np.int32)
    
    pos_df = candidate_features_df[candidate_features_df['label'] == 1].copy()
    neg_df = candidate_features_df[candidate_features_df['label'] == 0].copy()
    
    n_pos = len(pos_df)
    if n_pos == 0:
        available_cols = [c for c in FEATURE_COLS if c in candidate_features_df.columns]
        return candidate_features_df[available_cols], candidate_features_df['label']
        
    # 3. Mine True Hard Negatives: Rank by weighted similarity score (precision-tuned)
    neg_df['hard_rank_score'] = (
        0.50 * neg_df.get('name_lev_sim', 0.0) +
        0.25 * (neg_df.get('blocking_score', 0.0) / 10.0) +
        0.25 * neg_df.get('postal_match', 0.0)
    )
    neg_df = neg_df.sort_values(by='hard_rank_score', ascending=False)
    
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
    
    available_cols = [c for c in FEATURE_COLS if c in balanced_df.columns]
    return balanced_df[available_cols], balanced_df['label']

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None
) -> lgb.LGBMClassifier:
    """Trains LightGBM classifier on extracted pairwise features and fits Isotonic Calibration."""
    params = MODEL_CONFIG['lgb_params'].copy()
    model = lgb.LGBMClassifier(**params)
    
    available_cols = [c for c in FEATURE_COLS if c in X_train.columns]
    X_tr = X_train[available_cols]
    model.fit(X_tr, y_train)
    
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    if X_val is not None and not X_val.empty and y_val is not None and len(y_val) > 0:
        X_v = X_val[available_cols]
        val_raw_probs = model.predict_proba(X_v)[:, 1]
        calibrator.fit(val_raw_probs, y_val)
    else:
        tr_raw_probs = model.predict_proba(X_tr)[:, 1]
        calibrator.fit(tr_raw_probs, y_train)
        
    model.calibrator = calibrator
    return model

def predict_match_probabilities(
    model: lgb.LGBMClassifier,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Predicts calibrated pair match probabilities for candidate pairs and preserves evidence features."""
    if features_df.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_id', 'prob'])
        
    available_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    X = features_df[available_cols]
    raw_probs = model.predict_proba(X)[:, 1]
    
    if hasattr(model, 'calibrator') and model.calibrator is not None:
        probs = model.calibrator.transform(raw_probs)
    else:
        probs = raw_probs
        
    probs = np.clip(probs, 0.0, 1.0)
    
    result_df = features_df[['source1_entity_id', 'candidate_id']].copy()
    result_df['prob'] = probs
    
    # Preserve key evidence flags for Policy R post-filtering
    for col in ['name_lev_sim', 'postal_match', 'exact_name_hit', 'blocking_score', 'postal_house_hit']:
        if col in features_df.columns:
            result_df[col] = features_df[col].values
        else:
            result_df[col] = 0.0
            
    return result_df
