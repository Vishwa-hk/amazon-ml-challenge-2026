import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple

def compute_macro_f05(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes official macro-averaged F0.5 score, Precision, and Recall across all Source 1 entities.
    Singletons correctly predicted as empty score 1.0; false merges on singletons score 0.0.
    Precision is weighted 2x over Recall: F0.5 = (1.25 * P * R) / (0.25 * P + R).
    """
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        gt_map[s1id] = set([m.strip() for m in matches if m.strip()])
        
    pred_map = {}
    for _, row in pred_df.iterrows():
        s1id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        pred_map[s1id] = set([m.strip() for m in matches if m.strip()])
        
    f05_scores = []
    precisions = []
    recalls = []
    
    n_singletons = 0
    correct_singletons = 0
    singleton_false_positives = 0
    
    n_multimatch = 0
    exact_multimatch = 0
    false_merges = 0
    
    for s1id, true_set in gt_map.items():
        pred_set = pred_map.get(s1id, set())
        
        # Case A: True Singleton (no matches in ground truth)
        if len(true_set) == 0:
            n_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                singleton_false_positives += 1
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            continue
            
        # Case B: True Non-Singleton
        if len(true_set) > 1:
            n_multimatch += 1
            if pred_set == true_set:
                exact_multimatch += 1
                
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        intersection = len(true_set & pred_set)
        prec = intersection / len(pred_set)
        rec = intersection / len(true_set)
        
        if len(pred_set - true_set) > 0:
            false_merges += 1
            
        precisions.append(prec)
        recalls.append(rec)
        
        denom = (0.25 * prec) + rec
        if denom == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / denom
            f05_scores.append(f05)
            
    return {
        'f05_macro': float(np.mean(f05_scores)),
        'precision_macro': float(np.mean(precisions)),
        'recall_macro': float(np.mean(recalls)),
        'total_eval_entities': len(gt_map),
        'n_singletons': n_singletons,
        'singleton_accuracy': (correct_singletons / n_singletons) if n_singletons > 0 else 1.0,
        'singleton_false_positives': singleton_false_positives,
        'false_merges': false_merges,
        'n_multimatch': n_multimatch,
        'multimatch_accuracy': (exact_multimatch / n_multimatch) if n_multimatch > 0 else 1.0
    }

def compute_candidate_recall_ceiling(
    candidate_pairs_df: pd.DataFrame,
    gt_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluates Candidate Generation (Blocking) quality by computing
    how many true positive pairs were captured in the candidate set.
    """
    total_true_pairs = 0
    captured_true_pairs = 0
    
    cand_map = {}
    for _, row in candidate_pairs_df.iterrows():
        s1id = row['source1_entity_id']
        cands = str(row['candidate_entity_ids']).split(',') if row['candidate_entity_ids'] else []
        cand_map[s1id] = set([c.strip() for c in cands if c.strip()])
        
    for _, row in gt_df.iterrows():
        s1id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        true_matches = set([m.strip() for m in matches if m.strip()])
        
        total_true_pairs += len(true_matches)
        cand_set = cand_map.get(s1id, set())
        captured_true_pairs += len(true_matches & cand_set)
        
    recall_ceiling = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    return {
        'total_true_pairs': total_true_pairs,
        'captured_true_pairs': captured_true_pairs,
        'candidate_recall_ceiling': recall_ceiling
    }
