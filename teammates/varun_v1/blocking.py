import collections
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, List
from config import BLOCKING_CONFIG

def generate_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,  # Combined S2 + S3 dataframe
    top_k: int = 15
) -> pd.DataFrame:
    """
    Ultra-Fast Pure Inverted-Index Candidate Generator (Blocking).
    
    Eliminates expensive global TF-IDF sparse matrix multiplication completely.
    Uses hash maps and rare inverted indexes for 1.7M x 10M scalability:
    1. Exact normalized name hash lookup (O(1))
    2. Rare informative token inverted index (small bucket lookup)
    3. Prefix / character signature index (3-char bucket lookup)
    4. Postal code + house number index
    
    Returns candidates ranked by accumulated blocking confidence score descending (top 15 max).
    """
    scores_dict = collections.defaultdict(lambda: collections.defaultdict(float))
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    # --- Stage 1: Exact Normalized Name Hash Map ---
    exact_map = collections.defaultdict(list)
    for tid, name in zip(target_ids, target_df['name_norm'].values):
        if name:
            exact_map[name].append(tid)
            
    for s1id, name in zip(s1_ids, s1_df['name_norm'].values):
        if name and name in exact_map:
            for match_id in exact_map[name][:15]:
                scores_dict[s1id][match_id] += 5.0
                
    # --- Stage 2: Rare Informative Token Inverted Index ---
    token_map = collections.defaultdict(list)
    for tid, info_toks in zip(target_ids, target_df['name_info_tokens'].values):
        if info_toks:
            for tok in info_toks.split():
                if len(tok) >= 3:
                    token_map[tok].append(tid)
                    
    # Index ONLY rare tokens (bucket size <= 500) to ensure fast lookup & high specificity
    rare_token_map = {k: v for k, v in token_map.items() if len(v) <= 500}
    
    for s1id, info_toks in zip(s1_ids, s1_df['name_info_tokens'].values):
        if info_toks:
            for tok in info_toks.split():
                if tok in rare_token_map:
                    for tid in rare_token_map[tok]:
                        scores_dict[s1id][tid] += 3.0
                        
    # --- Stage 3: Prefix / Character Signature Index ---
    prefix_map = collections.defaultdict(list)
    for tid, pref in zip(target_ids, target_df['name_prefix_3'].values):
        if pref and len(pref) >= 3:
            prefix_map[pref].append(tid)
            
    rare_prefix_map = {k: v for k, v in prefix_map.items() if len(v) <= 300}
    
    for s1id, pref in zip(s1_ids, s1_df['name_prefix_3'].values):
        if pref and pref in rare_prefix_map:
            for tid in rare_prefix_map[pref]:
                scores_dict[s1id][tid] += 2.0

    # --- Stage 4: Postal Code + House Number Index ---
    postal_map = collections.defaultdict(list)
    for tid, post, num in zip(target_ids, target_df['postal_code'].values, target_df['addr_numbers'].values):
        if post:
            postal_map[post].append((tid, set(num.split())))
            
    for s1id, post, num in zip(s1_ids, s1_df['postal_code'].values, s1_df['addr_numbers'].values):
        if post and post in postal_map:
            s1_nums = set(num.split())
            for tid, t_nums in postal_map[post][:50]:
                if not s1_nums or not t_nums or s1_nums.intersection(t_nums):
                    scores_dict[s1id][tid] += 2.5

    # Convert scores_dict to structured DataFrame with candidates SORTED by score descending
    records = []
    for s1id in s1_df['entity_id']:
        cand_map = scores_dict.get(s1id, {})
        if cand_map:
            sorted_cands = sorted(cand_map.items(), key=lambda x: x[1], reverse=True)
            cand_list = [c[0] for c in sorted_cands[:top_k]]
        else:
            cand_list = []
            
        records.append({
            'source1_entity_id': s1id,
            'candidate_entity_ids': ",".join(cand_list)
        })
        
    return pd.DataFrame(records)
