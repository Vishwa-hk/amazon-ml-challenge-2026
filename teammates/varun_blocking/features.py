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

def char_ngram_jaccard_vec(s1_arr: np.ndarray, s2_arr: np.ndarray, n: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """Computes true character n-gram Jaccard similarity between string arrays."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    if mask is None:
        indices = range(len(s1_arr))
    else:
        indices = np.where(mask)[0]
    for i in indices:
        str1, str2 = s1_arr[i], s2_arr[i]
        if not str1 or not str2:
            scores[i] = 0.0
            continue
        ng1 = set([str1[k:k+n] for k in range(len(str1) - n + 1)])
        ng2 = set([str2[k:k+n] for k in range(len(str2) - n + 1)])
        union = len(ng1 | ng2)
        scores[i] = (len(ng1 & ng2) / union) if union > 0 else 0.0
    return scores

def lev_sim_vec(s1_arr: np.ndarray, s2_arr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Computes true normalized Levenshtein similarity via RapidFuzz."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    if mask is None:
        indices = range(len(s1_arr))
    else:
        indices = np.where(mask)[0]
    for i in indices:
        str1, str2 = s1_arr[i], s2_arr[i]
        if not str1 or not str2:
            scores[i] = 0.0
        else:
            scores[i] = Levenshtein.normalized_similarity(str1[:60], str2[:60])
    return scores

def token_containment_vec(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """Computes token containment similarity: |S1 n S2| / min(|S1|, |S2|)."""
    scores = np.zeros(len(s1_arr), dtype=np.float32)
    for i, (str1, str2) in enumerate(zip(s1_arr, s2_arr)):
        if not str1 or not str2:
            scores[i] = 0.0
            continue
        set1 = set(str1.split())
        set2 = set(str2.split())
        min_len = min(len(set1), len(set2))
        scores[i] = (len(set1 & set2) / min_len) if min_len > 0 else 0.0
    return scores

def extract_pairwise_features_vectorized(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    evidence_dict: Dict = None
) -> pd.DataFrame:
    """
    Fast Inverted-Index Pairwise Feature Extractor.
    Extracts 28+ rich name, address, cross-field, missingness, and evidence features.
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
        'source1_entity_id': np.array(s1_list, dtype=object),
        'candidate_id': np.array(target_list, dtype=object)
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
        
    # 3. Evidence Features Extraction (First so we can gate expensive features by blocking_score >= 4.0)
    blocking_score = np.zeros(len(pair_df), dtype=np.float32)
    blocking_rule_count = np.zeros(len(pair_df), dtype=np.float32)
    exact_name_hit = np.zeros(len(pair_df), dtype=np.float32)
    agg_exact_hit = np.zeros(len(pair_df), dtype=np.float32)
    rare_token_hit = np.zeros(len(pair_df), dtype=np.float32)
    prefix_hit = np.zeros(len(pair_df), dtype=np.float32)
    postal_house_hit = np.zeros(len(pair_df), dtype=np.float32)
    cross_field_hit = np.zeros(len(pair_df), dtype=np.float32)
    phonetic_postal_hit = np.zeros(len(pair_df), dtype=np.float32)
    residual_tfidf_hit = np.zeros(len(pair_df), dtype=np.float32)

    if evidence_dict:
        pairs = list(zip(pair_df['source1_entity_id'].values, pair_df['candidate_id'].values))
        ev_list = [evidence_dict.get(p, None) for p in pairs]
        n_p = len(pairs)
        
        blocking_score = np.fromiter((ev.get('blocking_score', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        blocking_rule_count = np.fromiter((ev.get('blocking_rule_count', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        exact_name_hit = np.fromiter((ev.get('exact_name_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        agg_exact_hit = np.fromiter((ev.get('agg_exact_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        rare_token_hit = np.fromiter((ev.get('rare_token_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        prefix_hit = np.fromiter((ev.get('prefix_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        postal_house_hit = np.fromiter((ev.get('postal_house_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        cross_field_hit = np.fromiter((ev.get('cross_field_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        phonetic_postal_hit = np.fromiter((ev.get('phonetic_postal_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)
        residual_tfidf_hit = np.fromiter((ev.get('residual_tfidf_hit', 0.0) if ev else 0.0 for ev in ev_list), dtype=np.float32, count=n_p)

    # Expensive Feature Gating Mask: blocking_score >= 2.0 or strong hits or evidence_dict is None
    if evidence_dict is None:
        expensive_gate = None
    else:
        expensive_gate = (blocking_score >= 2.0) | (exact_name_hit == 1.0) | (agg_exact_hit == 1.0) | (postal_house_hit == 1.0) | (residual_tfidf_hit == 1.0) | (cross_field_hit == 1.0)

    # 4. Name Features
    n1_raw = s1_df['business_name'].values[s1_pos]
    n2_raw = target_df['business_name'].values[target_pos]
    
    n1_norm = s1_df['name_norm'].values[s1_pos]
    n2_norm = target_df['name_norm'].values[target_pos]
    
    n1_agg = s1_df['name_agg_norm'].values[s1_pos] if 'name_agg_norm' in s1_df.columns else n1_norm
    n2_agg = target_df['name_agg_norm'].values[target_pos] if 'name_agg_norm' in target_df.columns else n2_norm
    
    name_exact = (n1_raw == n2_raw).astype(np.float32)
    name_norm_exact = (n1_norm == n2_norm).astype(np.float32)
    name_agg_exact = (n1_agg == n2_agg).astype(np.float32)
    
    n1_pref = s1_df['name_prefix_3'].values[s1_pos]
    n2_pref = target_df['name_prefix_3'].values[target_pos]
    name_prefix_match = (n1_pref == n2_pref).astype(np.float32)
    
    ac1 = s1_df['acronym'].values[s1_pos] if 'acronym' in s1_df.columns else np.repeat('', len(s1_pos))
    ac2 = target_df['acronym'].values[target_pos] if 'acronym' in target_df.columns else np.repeat('', len(target_pos))
    acronym_match = np.where((ac1 != '') & (ac2 != ''), (ac1 == ac2).astype(np.float32), 0.0).astype(np.float32)
    
    ft1 = s1_df['first_token'].values[s1_pos] if 'first_token' in s1_df.columns else n1_pref
    ft2 = target_df['first_token'].values[target_pos] if 'first_token' in target_df.columns else n2_pref
    first_token_sim = lev_sim_vec(ft1, ft2, mask=expensive_gate)
    
    lt1 = s1_df['last_token'].values[s1_pos] if 'last_token' in s1_df.columns else n1_pref
    lt2 = target_df['last_token'].values[target_pos] if 'last_token' in target_df.columns else n2_pref
    last_token_sim = lev_sim_vec(lt1, lt2, mask=expensive_gate)
    
    l1 = np.array([len(x) for x in n1_norm], dtype=np.int32)
    l2 = np.array([len(x) for x in n2_norm], dtype=np.int32)
    max_l = np.maximum(l1, l2)
    name_len_diff = np.where(max_l > 0, np.abs(l1 - l2) / max_l, 0.0).astype(np.float32)
    
    name_token_jaccard = token_jaccard_vec(n1_norm, n2_norm)
    name_char_jaccard = char_ngram_jaccard_vec(n1_norm, n2_norm, n=3, mask=expensive_gate)
    name_lev_sim = lev_sim_vec(n1_norm, n2_norm, mask=expensive_gate)
    name_containment = token_containment_vec(n1_norm, n2_norm)
    
    # 5. Address Features
    a1_norm = s1_df['address_clean'].values[s1_pos]
    a2_norm = target_df['address_clean'].values[target_pos]
    addr_empty = ((a1_norm == '') | (a2_norm == '')).astype(np.float32)
    
    addr_token_jaccard = token_jaccard_vec(a1_norm, a2_norm)
    addr_char_jaccard = char_ngram_jaccard_vec(a1_norm, a2_norm, n=3, mask=expensive_gate)
    addr_lev_sim = lev_sim_vec(a1_norm, a2_norm, mask=expensive_gate)
    
    p1 = s1_df['postal_code'].values[s1_pos]
    p2 = target_df['postal_code'].values[target_pos]
    postal_has_both = (p1 != '') & (p2 != '')
    postal_match = np.where(postal_has_both, (p1 == p2).astype(np.float32), 0.5).astype(np.float32)
    postal_missing = (~postal_has_both).astype(np.float32)
    
    h1 = s1_df['house_number'].values[s1_pos] if 'house_number' in s1_df.columns else np.repeat('', len(s1_pos))
    h2 = target_df['house_number'].values[target_pos] if 'house_number' in target_df.columns else np.repeat('', len(target_pos))
    house_has_both = (h1 != '') & (h2 != '')
    house_match = np.where(house_has_both, (h1 == h2).astype(np.float32), 0.5).astype(np.float32)
    house_conflict = np.where(house_has_both & (h1 != h2), 1.0, 0.0).astype(np.float32)
    
    num1 = s1_df['addr_numbers'].values[s1_pos]
    num2 = target_df['addr_numbers'].values[target_pos]
    num_has_both = (num1 != '') & (num2 != '')
    num1_first = np.array([x.split()[0] if x else '' for x in num1])
    num2_first = np.array([x.split()[0] if x else '' for x in num2])
    number_match = np.where(num_has_both, (num1_first == num2_first).astype(np.float32), 0.5).astype(np.float32)
    
    # 6. Cross-Field Interaction Features
    name_strong_addr_strong = np.where((name_lev_sim > 0.8) & (addr_lev_sim > 0.8), 1.0, 0.0).astype(np.float32)
    name_strong_postal = np.where((name_lev_sim > 0.8) & (postal_match == 1.0), 1.0, 0.0).astype(np.float32)
    name_strong_house_num = np.where((name_lev_sim > 0.8) & (house_match == 1.0), 1.0, 0.0).astype(np.float32)
    exact_postal_similar_name = np.where((postal_match == 1.0) & (name_lev_sim > 0.6), 1.0, 0.0).astype(np.float32)
    exact_name_conflicting_addr = np.where((name_norm_exact == 1.0) & (addr_lev_sim < 0.3) & (addr_empty == 0.0), 1.0, 0.0).astype(np.float32)
    exact_name_conflicting_postal = np.where((name_norm_exact == 1.0) & (postal_match == 0.0), 1.0, 0.0).astype(np.float32)
    
    # Phonetic Features
    pf1 = s1_df['phonetic_first'].values[s1_pos] if 'phonetic_first' in s1_df.columns else np.repeat('', len(s1_pos))
    pf2 = target_df['phonetic_first'].values[target_pos] if 'phonetic_first' in target_df.columns else np.repeat('', len(target_pos))
    phonetic_sim_first = np.where((pf1 != '') & (pf2 != ''), (pf1 == pf2).astype(np.float32), 0.0).astype(np.float32)

    pk1_s1 = s1_df['phonetic_key1'].values[s1_pos] if 'phonetic_key1' in s1_df.columns else pf1
    pk1_t2 = target_df['phonetic_key1'].values[target_pos] if 'phonetic_key1' in target_df.columns else pf2
    phonetic_sim_key = np.where((pk1_s1 != '') & (pk1_t2 != ''), (pk1_s1 == pk1_t2).astype(np.float32), 0.0).astype(np.float32)

    # Bidirectional containment
    containment_bidi = np.maximum(name_containment, token_containment_vec(n2_norm, n1_norm))

    # House + Postal Conflict flag (postal match but house conflict)
    house_postal_conflict = np.where((postal_match == 1.0) & (house_conflict == 1.0), 1.0, 0.0).astype(np.float32)

    # Acronym + Postal interaction
    acronym_postal_match = np.where((acronym_match == 1.0) & (postal_match == 1.0), 1.0, 0.0).astype(np.float32)

    # Structural & Source Features
    c1 = s1_df['country'].values[s1_pos]
    c2 = target_df['country'].values[target_pos]
    country_match = (c1 == c2).astype(np.float32)
    source_pair = np.where(pair_df['candidate_id'].str.startswith('S2-'), 1.0, 2.0).astype(np.float32)

    # Assemble feature DataFrame
    pair_df['name_exact'] = name_exact
    pair_df['name_norm_exact'] = name_norm_exact
    pair_df['name_agg_exact'] = name_agg_exact
    pair_df['name_token_jaccard'] = name_token_jaccard
    pair_df['name_char_jaccard'] = name_char_jaccard
    pair_df['name_lev_sim'] = name_lev_sim
    pair_df['name_containment'] = name_containment
    pair_df['containment_bidi'] = containment_bidi
    pair_df['first_token_sim'] = first_token_sim
    pair_df['last_token_sim'] = last_token_sim
    pair_df['phonetic_sim_first'] = phonetic_sim_first
    pair_df['phonetic_sim_key'] = phonetic_sim_key
    pair_df['name_prefix_match'] = name_prefix_match
    pair_df['acronym_match'] = acronym_match
    pair_df['name_len_diff'] = name_len_diff
    pair_df['addr_token_jaccard'] = addr_token_jaccard
    pair_df['addr_char_jaccard'] = addr_char_jaccard
    pair_df['addr_lev_sim'] = addr_lev_sim
    pair_df['addr_empty'] = addr_empty
    pair_df['postal_match'] = postal_match
    pair_df['postal_missing'] = postal_missing
    pair_df['number_match'] = number_match
    pair_df['house_match'] = house_match
    pair_df['house_conflict'] = house_conflict
    pair_df['house_postal_conflict'] = house_postal_conflict
    pair_df['acronym_postal_match'] = acronym_postal_match
    pair_df['name_strong_addr_strong'] = name_strong_addr_strong
    pair_df['name_strong_postal'] = name_strong_postal
    pair_df['name_strong_house_num'] = name_strong_house_num
    pair_df['exact_postal_similar_name'] = exact_postal_similar_name
    pair_df['exact_name_conflicting_addr'] = exact_name_conflicting_addr
    pair_df['exact_name_conflicting_postal'] = exact_name_conflicting_postal
    pair_df['country_match'] = country_match
    pair_df['source_pair'] = source_pair
    pair_df['blocking_score'] = blocking_score
    pair_df['blocking_rule_count'] = blocking_rule_count
    pair_df['exact_name_hit'] = exact_name_hit
    pair_df['agg_exact_hit'] = agg_exact_hit
    pair_df['rare_token_hit'] = rare_token_hit
    pair_df['prefix_hit'] = prefix_hit
    pair_df['postal_house_hit'] = postal_house_hit
    pair_df['cross_field_hit'] = cross_field_hit
    pair_df['phonetic_postal_hit'] = phonetic_postal_hit
    pair_df['residual_tfidf_hit'] = residual_tfidf_hit
    
    return pair_df
