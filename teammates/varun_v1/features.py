import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from typing import List, Tuple, Dict

def token_jaccard_vec(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """Computes true token Jaccard similarity between string arrays."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    for i, (str1, str2) in enumerate(zip(s1_arr, s2_arr)):
        if not str1 or not str2:
            scores[i] = 0.0
            continue
        set1 = set(str1.split())
        set2 = set(str2.split())
        union = len(set1 | set2)
        scores[i] = (len(set1 & set2) / union) if union > 0 else 0.0
    return scores

def char_ngram_jaccard_vec(s1_arr: np.ndarray, s2_arr: np.ndarray, n: int = 3) -> np.ndarray:
    """Computes true character n-gram Jaccard similarity between string arrays."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    for i, (str1, str2) in enumerate(zip(s1_arr, s2_arr)):
        if not str1 or not str2:
            scores[i] = 0.0
            continue
        ng1 = set([str1[k:k+n] for k in range(len(str1) - n + 1)])
        ng2 = set([str2[k:k+n] for k in range(len(str2) - n + 1)])
        union = len(ng1 | ng2)
        scores[i] = (len(ng1 & ng2) / union) if union > 0 else 0.0
    return scores

def lev_sim_vec(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """Computes true normalized Levenshtein similarity via RapidFuzz."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    for i, (str1, str2) in enumerate(zip(s1_arr, s2_arr)):
        if not str1 or not str2:
            scores[i] = 0.0
        else:
            scores[i] = Levenshtein.normalized_similarity(str1[:60], str2[:60])
    return scores

def extract_pairwise_features_vectorized(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fast Inverted-Index Pairwise Feature Extractor.
    Operates strictly on top candidate pairs (max 15/S1 entity).
    Eliminates global TF-IDF matrix fitting and vectorization.
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame()
        
    # 1. Explode candidate pairs into flat arrays
    s1_ids_raw = candidate_pairs_df['source1_entity_id'].values
    cand_ids_raw = candidate_pairs_df['candidate_entity_ids'].values
    
    s1_list = []
    target_list = []
    
    for s1id, cands in zip(s1_ids_raw, cand_ids_raw):
        if cands:
            for cid in cands.split(','):
                cid = cid.strip()
                if cid:
                    s1_list.append(s1id)
                    target_list.append(cid)
                    
    if not s1_list:
        return pd.DataFrame()
        
    pair_df = pd.DataFrame({
        'source1_entity_id': s1_list,
        'candidate_id': target_list
    })
    
    # 2. Fast integer indexing using pandas Index
    s1_indexer = pd.Index(s1_df['entity_id'])
    target_indexer = pd.Index(target_df['entity_id'])
    
    s1_pos = s1_indexer.get_indexer(pair_df['source1_entity_id'])
    target_pos = target_indexer.get_indexer(pair_df['candidate_id'])
    
    valid_mask = (s1_pos >= 0) & (target_pos >= 0)
    if not np.all(valid_mask):
        pair_df = pair_df[valid_mask].reset_index(drop=True)
        s1_pos = s1_pos[valid_mask]
        target_pos = target_pos[valid_mask]
        
    # 3. Name Features
    n1_raw = s1_df['business_name'].values[s1_pos]
    n2_raw = target_df['business_name'].values[target_pos]
    
    n1_norm = s1_df['name_norm'].values[s1_pos]
    n2_norm = target_df['name_norm'].values[target_pos]
    
    name_exact = (n1_raw == n2_raw).astype(np.float32)
    name_norm_exact = (n1_norm == n2_norm).astype(np.float32)
    
    n1_pref = s1_df['name_prefix_3'].values[s1_pos]
    n2_pref = target_df['name_prefix_3'].values[target_pos]
    name_prefix_match = (n1_pref == n2_pref).astype(np.float32)
    
    l1 = np.vectorize(len)(n1_norm)
    l2 = np.vectorize(len)(n2_norm)
    max_l = np.maximum(l1, l2)
    name_len_diff = np.where(max_l > 0, np.abs(l1 - l2) / max_l, 0.0).astype(np.float32)
    
    # Fast Pairwise Similarity Calculations
    name_token_jaccard = token_jaccard_vec(n1_norm, n2_norm)
    name_char_jaccard = char_ngram_jaccard_vec(n1_norm, n2_norm, n=3)
    name_lev_sim = lev_sim_vec(n1_norm, n2_norm)
    
    # 4. Address Features
    a1_norm = s1_df['address_clean'].values[s1_pos]
    a2_norm = target_df['address_clean'].values[target_pos]
    addr_empty = ((a1_norm == '') | (a2_norm == '')).astype(np.float32)
    
    addr_token_jaccard = token_jaccard_vec(a1_norm, a2_norm)
    addr_char_jaccard = char_ngram_jaccard_vec(a1_norm, a2_norm, n=3)
    
    p1 = s1_df['postal_code'].values[s1_pos]
    p2 = target_df['postal_code'].values[target_pos]
    postal_has_both = (p1 != '') & (p2 != '')
    postal_match = np.where(postal_has_both, (p1 == p2).astype(np.float32), 0.5).astype(np.float32)
    
    num1 = s1_df['addr_numbers'].values[s1_pos]
    num2 = target_df['addr_numbers'].values[target_pos]
    num_has_both = (num1 != '') & (num2 != '')
    num1_first = np.array([x.split()[0] if x else '' for x in num1])
    num2_first = np.array([x.split()[0] if x else '' for x in num2])
    number_match = np.where(num_has_both, (num1_first == num2_first).astype(np.float32), 0.5).astype(np.float32)
    
    # 5. Structural & Source Features
    c1 = s1_df['country'].values[s1_pos]
    c2 = target_df['country'].values[target_pos]
    country_match = (c1 == c2).astype(np.float32)
    source_pair = np.where(pair_df['candidate_id'].str.startswith('S2-'), 1.0, 2.0).astype(np.float32)

    # Assemble feature DataFrame
    pair_df['name_exact'] = name_exact
    pair_df['name_norm_exact'] = name_norm_exact
    pair_df['name_token_jaccard'] = name_token_jaccard
    pair_df['name_char_jaccard'] = name_char_jaccard
    pair_df['name_lev_sim'] = name_lev_sim
    pair_df['name_prefix_match'] = name_prefix_match
    pair_df['name_len_diff'] = name_len_diff
    pair_df['addr_token_jaccard'] = addr_token_jaccard
    pair_df['addr_char_jaccard'] = addr_char_jaccard
    pair_df['addr_empty'] = addr_empty
    pair_df['postal_match'] = postal_match
    pair_df['number_match'] = number_match
    pair_df['country_match'] = country_match
    pair_df['source_pair'] = source_pair
    
    return pair_df
