import collections
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from evaluate import compute_macro_f05

def prepare_candidate_ranks(prob_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Pre-sorts and groups candidate details per S1 entity ONCE for fast policy evaluation."""
    cand_ranks = collections.defaultdict(list)
    if prob_df.empty:
        return cand_ranks
        
    sorted_df = prob_df.sort_values(by=['source1_entity_id', 'prob'], ascending=[True, False])
    
    s1ids = sorted_df['source1_entity_id'].values
    cids = sorted_df['candidate_id'].values
    probs = sorted_df['prob'].values
    
    levs = sorted_df['name_lev_sim'].values if 'name_lev_sim' in sorted_df.columns else np.ones(len(sorted_df))
    posts = sorted_df['postal_match'].values if 'postal_match' in sorted_df.columns else np.zeros(len(sorted_df))
    exacts = sorted_df['exact_name_hit'].values if 'exact_name_hit' in sorted_df.columns else np.zeros(len(sorted_df))
    block_scs = sorted_df['blocking_score'].values if 'blocking_score' in sorted_df.columns else np.zeros(len(sorted_df))
    post_houses = sorted_df['postal_house_hit'].values if 'postal_house_hit' in sorted_df.columns else np.zeros(len(sorted_df))
    
    for s1id, cid, prob, lev, post, exact, block_sc, post_house in zip(
        s1ids, cids, probs, levs, posts, exacts, block_scs, post_houses
    ):
        cand_ranks[s1id].append({
            'cid': cid,
            'prob': float(prob),
            'name_lev_sim': float(lev),
            'postal_match': float(post),
            'exact_name_hit': float(exact),
            'blocking_score': float(block_sc),
            'postal_house_hit': float(post_house)
        })
    return cand_ranks

def generate_matching_results(
    prob_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: float = 0.50,
    policy: str = 'M',
    margin: float = 0.05,
    secondary_threshold: float = 0.65,
    cand_ranks: Dict = None
) -> pd.DataFrame:
    """
    Groups pair probabilities by source1_entity_id and selects matches according to specified decision policy:
      - Policy M: High-Precision Multi-Match Policy (proper multi-match without artificial caps).
      - Policy S: Extreme-Precision Evidence Gate Policy.
      - Policy D: Dual-Threshold Policy (primary threshold + secondary threshold).
      - Policy C: Top-1 Margin Policy.
      - Policy B: Top-1 Single Match Policy.
      - Policy A: Global Threshold Multi-Match Policy.
    """
    if cand_ranks is None:
        cand_ranks = prepare_candidate_ranks(prob_df)

    s1_matches = collections.defaultdict(list)
    
    for s1id in all_s1_ids:
        cands = cand_ranks.get(s1id, [])
        if not cands:
            continue
            
        p1 = cands[0]['prob']
        p2 = cands[1]['prob'] if len(cands) > 1 else 0.0
        m_val = p1 - p2
        
        if policy == 'M':
            # Policy M: High-Precision Adaptive Multi-Match Policy
            # Floor check: protect true singletons aggressively
            if p1 < threshold:
                continue
                
            c1_dict = cands[0]
            s1_matches[s1id].append(c1_dict['cid'])
            
            # Secondary candidate selection: support multi-matches (3, 4, 5, up to all candidates)
            sec_t = max(threshold, secondary_threshold)
            for k in range(1, len(cands)):
                ck_dict = cands[k]
                pk = ck_dict['prob']
                if pk < sec_t:
                    break
                    
                p_next = cands[k+1]['prob'] if k + 1 < len(cands) else 0.0
                cand_margin = pk - p_next
                
                ck_exact = ck_dict.get('exact_name_hit', 0.0) == 1.0
                ck_postal_house = ck_dict.get('postal_house_hit', 0.0) == 1.0
                ck_lev = ck_dict.get('name_lev_sim', 0.0)
                
                # Accept secondary if probability is high and either margin or strong evidence holds
                if (pk >= sec_t and (cand_margin >= margin or ck_exact or ck_postal_house or ck_lev >= 0.85)):
                    s1_matches[s1id].append(ck_dict['cid'])
                else:
                    # Break on confidence cliff
                    if pk < 0.80:
                        break

        elif policy == 'S':
            # Policy S: Extreme-Precision Policy
            if p1 < 0.70:
                continue
                
            c1_dict = cands[0]
            c1_exact = c1_dict.get('exact_name_hit', 0.0) == 1.0
            c1_postal_house = c1_dict.get('postal_house_hit', 0.0) == 1.0
            c1_lev = c1_dict.get('name_lev_sim', 0.0)
            
            evidence_gate = c1_exact or c1_postal_house or c1_lev >= 0.88
            margin_gate = m_val >= margin or (c1_exact and c1_postal_house)
            
            if p1 >= threshold and evidence_gate and margin_gate:
                s1_matches[s1id].append(c1_dict['cid'])
                
                for k in range(1, len(cands)):
                    ck = cands[k]
                    if ck['prob'] >= max(threshold, 0.85):
                        ck_exact = ck.get('exact_name_hit', 0.0) == 1.0
                        ck_ph = ck.get('postal_house_hit', 0.0) == 1.0
                        if ck_exact or ck_ph or ck.get('name_lev_sim', 0.0) >= 0.85:
                            s1_matches[s1id].append(ck['cid'])
                    else:
                        break

        elif policy == 'D':
            # Policy D: Dual-Threshold
            if p1 >= threshold:
                s1_matches[s1id].append(cands[0]['cid'])
                sec_t = max(threshold + 0.10, secondary_threshold)
                for k in range(1, len(cands)):
                    if cands[k]['prob'] >= sec_t:
                        s1_matches[s1id].append(cands[k]['cid'])
                    else:
                        break

        elif policy == 'C':
            # Policy C: Top-1 with Relative Margin
            if p1 >= threshold and m_val >= margin:
                s1_matches[s1id].append(cands[0]['cid'])

        elif policy == 'B':
            # Policy B: Top-1 only
            if p1 >= threshold:
                s1_matches[s1id].append(cands[0]['cid'])

        else: # Policy A: Global Threshold
            for item in cands:
                if item['prob'] >= threshold:
                    s1_matches[s1id].append(item['cid'])

    records = []
    for s1id in all_s1_ids:
        matches = s1_matches.get(s1id, [])
        records.append({
            'source1_entity_id': s1id,
            'matched_entity_ids': ",".join(matches)
        })
        
    return pd.DataFrame(records)

def optimize_decision_threshold(
    prob_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold_range: List[float] = None
) -> Tuple[float, float, str, float]:
    """
    Sweeps decision policies (M, D, S, A, B), thresholds, and margins
    to maximize the official competition metric: Macro F0.5.
    Prioritizes configurations achieving high precision without sacrificing F0.5.
    """
    if threshold_range is None:
        threshold_range = [round(x, 2) for x in np.arange(0.25, 0.85, 0.03)]
        
    cand_ranks = prepare_candidate_ranks(prob_df)
    
    best_threshold = 0.50
    best_f05 = 0.0
    best_prec = 0.0
    best_policy = 'M'
    best_margin = 0.05
    
    policies = ['M', 'D', 'A', 'S', 'B']
    policy_margins = {
        'M': [0.00, 0.04, 0.08, 0.12],
        'S': [0.00, 0.05, 0.10],
        'D': [0.00],
        'A': [0.00],
        'B': [0.00]
    }
    
    for pol in policies:
        m_list = policy_margins.get(pol, [0.0])
        for t in threshold_range:
            for m in m_list:
                sec_t = max(t + 0.10, 0.60)
                pred_df = generate_matching_results(
                    prob_df, all_s1_ids, threshold=t, policy=pol, margin=m, secondary_threshold=sec_t, cand_ranks=cand_ranks
                )
                eval_res = compute_macro_f05(pred_df, ground_truth_df)
                f05 = eval_res['f05_macro']
                prec = eval_res['precision_macro']
                
                # Maximize Macro F0.5; break ties using higher precision
                if f05 > best_f05 or (abs(f05 - best_f05) < 1e-4 and prec > best_prec):
                    best_f05 = f05
                    best_prec = prec
                    best_threshold = t
                    best_policy = pol
                    best_margin = m
                    
    return best_threshold, best_f05, best_policy, best_margin
