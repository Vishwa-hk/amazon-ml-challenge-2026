import collections
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from evaluate import compute_macro_f05

def generate_matching_results(
    prob_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Groups pair probabilities by source1_entity_id and selects matches
    above the probability threshold. Returns formatted matching_results DataFrame.
    """
    s1_matches = collections.defaultdict(list)
    
    if not prob_df.empty:
        filtered = prob_df[prob_df['prob'] >= threshold].sort_values(
            by=['source1_entity_id', 'prob'], ascending=[True, False]
        )
        
        for _, row in filtered.iterrows():
            s1id = row['source1_entity_id']
            cid = row['candidate_id']
            s1_matches[s1id].append(cid)
            
    records = []
    for s1id in all_s1_ids:
        matches = s1_matches.get(s1id, [])
        seen = set()
        unique_matches = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique_matches.append(m)
                
        records.append({
            'source1_entity_id': s1id,
            'matched_entity_ids': ",".join(unique_matches)
        })
        
    return pd.DataFrame(records)

def optimize_decision_threshold(
    prob_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold_range: List[float] = None
) -> Tuple[float, float]:
    """
    Sweeps probability thresholds to find optimal threshold maximizing macro F0.5 score.
    Returns (best_threshold, best_f05_score).
    """
    if threshold_range is None:
        threshold_range = [round(x, 2) for x in np.arange(0.15, 0.85, 0.05)]
        
    best_threshold = 0.50
    best_f05 = 0.0
    
    for t in threshold_range:
        pred_df = generate_matching_results(prob_df, all_s1_ids, threshold=t)
        eval_res = compute_macro_f05(pred_df, ground_truth_df)
        f05 = eval_res['f05_macro']
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            
    return best_threshold, best_f05
