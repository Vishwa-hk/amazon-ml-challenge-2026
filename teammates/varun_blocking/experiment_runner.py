import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    MODEL_CONFIG, PROJECT_ROOT
)
from io_utils import read_full_tsv
from preprocess import preprocess_df
from blocking import CandidateIndex
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from model import build_training_pairs, train_model, predict_match_probabilities, FEATURE_COLS
from decision import generate_matching_results, optimize_decision_threshold
from evaluate import compute_macro_f05, compute_candidate_recall_ceiling

def load_and_preprocess_split(sample_size: int = 5000):
    """Loads a reproducible sample of training data for experiments."""
    s1_raw = read_full_tsv(TRAIN_S1_PATH).head(sample_size)
    gt_raw = read_full_tsv(TRAIN_GT_PATH)
    gt_raw = gt_raw[gt_raw['source1_entity_id'].isin(set(s1_raw['entity_id']))]

    gt_matched_ids = set()
    for ids in gt_raw['matched_entity_ids']:
        if ids:
            gt_matched_ids.update(str(ids).split(','))

    s2_raw = read_full_tsv(TRAIN_S2_PATH)
    s3_raw = read_full_tsv(TRAIN_S3_PATH)

    s2_sub = s2_raw[s2_raw['entity_id'].isin(gt_matched_ids) | (s2_raw.index < sample_size * 5)]
    s3_sub = s3_raw[s3_raw['entity_id'].isin(gt_matched_ids) | (s3_raw.index < sample_size * 5)]

    s1_prep = preprocess_df(s1_raw)
    target_prep = pd.concat([preprocess_df(s2_sub), preprocess_df(s3_sub)], ignore_index=True)

    # 80/20 train/val split
    s1_ids = s1_prep['entity_id'].unique()
    n_val = int(len(s1_ids) * 0.20)
    np_rand = pd.Series(s1_ids).sample(frac=1.0, random_state=42).values
    val_s1_ids = set(np_rand[:n_val])
    train_s1_ids = set(np_rand[n_val:])

    s1_tr = s1_prep[s1_prep['entity_id'].isin(train_s1_ids)].copy()
    gt_tr = gt_raw[gt_raw['source1_entity_id'].isin(train_s1_ids)].copy()

    s1_val = s1_prep[s1_prep['entity_id'].isin(val_s1_ids)].copy()
    gt_val = gt_raw[gt_raw['source1_entity_id'].isin(val_s1_ids)].copy()

    return s1_tr, gt_tr, s1_val, gt_val, target_prep

def run_candidate_recall_experiments(s1_val, gt_val, target_prep):
    """Experiment 1: Candidate Recall @ K Benchmarking."""
    print("\n=======================================================", flush=True)
    print("      EXPERIMENT 1: CANDIDATE RECALL BENCHMARK      ", flush=True)
    print("=======================================================", flush=True)

    index = CandidateIndex()
    index.build(target_prep)

    k_values = [10, 15, 25, 35, 50, 75, 100]
    recall_results = []

    for k in k_values:
        cand_df, _ = index.query(s1_val, top_k=k)
        rec_info = compute_candidate_recall_ceiling(cand_df, gt_val)
        cand_recall = rec_info['candidate_recall_ceiling']
        captured = rec_info['captured_true_pairs']
        total = rec_info['total_true_pairs']
        missed = total - captured
        
        recall_results.append({
            'K': k,
            'CandidateRecall': cand_recall,
            'Captured': captured,
            'Missed': missed,
            'Total': total
        })
        print(f"Recall@{k:3d}: {cand_recall:.4f} ({captured}/{total} captured, {missed} missed)", flush=True)

    return pd.DataFrame(recall_results), index

def run_missed_match_failure_analysis(s1_val, gt_val, target_prep, index, top_k=15):
    """Experiment 2: Detailed Failure Analysis on Missed True Matches."""
    print("\n=======================================================", flush=True)
    print("    EXPERIMENT 2: MISSED TRUE MATCH FAILURE ANALYSIS    ", flush=True)
    print("=======================================================", flush=True)

    cand_df, _ = index.query(s1_val, top_k=top_k)
    cand_map = {}
    for _, row in cand_df.iterrows():
        cands = str(row['candidate_entity_ids']).split(',') if row['candidate_entity_ids'] else []
        cand_map[row['source1_entity_id']] = set([c.strip() for c in cands if c.strip()])

    s1_dict = s1_val.set_index('entity_id').to_dict('index')
    target_dict = target_prep.set_index('entity_id').to_dict('index')

    missed_examples = []
    for _, row in gt_val.iterrows():
        s1id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        true_matches = set([m.strip() for m in matches if m.strip()])
        cands = cand_map.get(s1id, set())

        missed = true_matches - cands
        for m_id in missed:
            s1_info = s1_dict.get(s1id, {})
            t_info = target_dict.get(m_id, {})
            
            # Diagnose root cause
            s1_name = s1_info.get('business_name', '')
            t_name = t_info.get('business_name', '')
            s1_addr = s1_info.get('business_address', '')
            t_addr = t_info.get('business_address', '')
            s1_post = s1_info.get('postal_code', '')
            t_post = t_info.get('postal_code', '')
            
            reason = "other"
            if not s1_post or not t_post or s1_post != t_post:
                reason = "postal_code_mismatch_or_missing"
            elif len(set(s1_name.lower().split()) & set(t_name.lower().split())) == 0:
                reason = "zero_name_token_overlap"
            elif len(s1_name) < 4 or len(t_name) < 4:
                reason = "short_name_typo"
            
            missed_examples.append({
                's1_id': s1id,
                'target_id': m_id,
                's1_name': s1_name,
                'target_name': t_name,
                's1_address': s1_addr,
                'target_address': t_addr,
                'reason': reason
            })

    missed_df = pd.DataFrame(missed_examples)
    print(f"Total Missed True Match Pairs (at K={top_k}): {len(missed_df)}")
    if not missed_df.empty:
        print("\nFailure Cause Breakdown:")
        print(missed_df['reason'].value_counts())
        print("\nSample Missed Examples:")
        for idx, r in missed_df.head(5).iterrows():
            print(f"  S1 ({r['s1_id']}): '{r['s1_name']}' | Addr: '{r['s1_address']}'")
            print(f"  Target ({r['target_id']}): '{r['target_name']}' | Addr: '{r['target_address']}'")
            print(f"  Reason: {r['reason']}\n")
    return missed_df

def run_end_to_end_ablation_grid(s1_tr, gt_tr, s1_val, gt_val, target_prep):
    """Experiment 3: End-to-End Systematic Ablation Grid."""
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT 3: SYSTEMATIC ABLATION & OPTIMIZATION    ", flush=True)
    print("=======================================================", flush=True)

    tracemalloc.start()
    t0 = time.time()

    index = CandidateIndex()
    index.build(target_prep)

    print("Precomputing Candidate Index & Features for max K=35...", flush=True)
    cand_tr_max, ev_tr_max = index.query(s1_tr, top_k=35)
    cand_val_max, ev_val_max = index.query(s1_val, top_k=35)

    feat_tr_max = extract_pairwise_features(cand_tr_max, s1_tr, target_prep, evidence_dict=ev_tr_max)
    feat_val_max = extract_pairwise_features(cand_val_max, s1_val, target_prep, evidence_dict=ev_val_max)

    k_options = [15, 25, 35]
    hard_neg_options = [3, 5, 8]
    
    experiment_rows = []

    for k in k_options:
        # Fast slice of candidate pairs to top-k
        cand_val_k, _ = index.query(s1_val, top_k=k)
        rec_val = compute_candidate_recall_ceiling(cand_val_k, gt_val)
        cand_rec = rec_val['candidate_recall_ceiling']

        valid_s1_pairs_tr = set()
        for _, row in cand_tr_max.iterrows():
            s1id = row['source1_entity_id']
            cands = str(row['candidate_entity_ids']).split(',')[:k]
            for c in cands:
                if c.strip():
                    valid_s1_pairs_tr.add(f"{s1id}||{c.strip()}")

        valid_s1_pairs_val = set()
        for _, row in cand_val_k.iterrows():
            s1id = row['source1_entity_id']
            cands = str(row['candidate_entity_ids']).split(',')[:k]
            for c in cands:
                if c.strip():
                    valid_s1_pairs_val.add(f"{s1id}||{c.strip()}")

        # Ultra-fast vectorized pandas .isin() mask
        tr_keys = feat_tr_max['source1_entity_id'].astype(str) + "||" + feat_tr_max['candidate_id'].astype(str)
        val_keys = feat_val_max['source1_entity_id'].astype(str) + "||" + feat_val_max['candidate_id'].astype(str)

        feat_tr = feat_tr_max[tr_keys.isin(valid_s1_pairs_tr)].copy()
        feat_val = feat_val_max[val_keys.isin(valid_s1_pairs_val)].copy()

        for h_neg in hard_neg_options:
            MODEL_CONFIG['num_hard_negatives_per_pos'] = h_neg
            X_tr, y_tr = build_training_pairs(feat_tr, gt_tr)
            X_v, y_v = build_training_pairs(feat_val, gt_val)

            model = train_model(X_tr, y_tr, X_v, y_v)
            prob_val = predict_match_probabilities(model, feat_val)

            best_t, best_f05, best_pol, best_m = optimize_decision_threshold(
                prob_val, gt_val, list(s1_val['entity_id'])
            )

            val_matching = generate_matching_results(
                prob_val, list(s1_val['entity_id']), threshold=best_t, policy=best_pol, margin=best_m
            )
            val_eval = compute_macro_f05(val_matching, gt_val)

            # Failure counts
            gt_map = {row['source1_entity_id']: set(str(row['matched_entity_ids']).split(',')) for _, row in gt_val.iterrows()}
            pred_map = {row['source1_entity_id']: set(str(row['matched_entity_ids']).split(',')) for _, row in val_matching.iterrows()}

            false_merges = 0
            singleton_errors = 0
            multi_errors = 0

            for s1id, true_set in gt_map.items():
                true_set = set([x.strip() for x in true_set if x.strip()])
                pred_set = pred_map.get(s1id, set())
                pred_set = set([x.strip() for x in pred_set if x.strip()])

                if len(true_set) == 0 and len(pred_set) > 0:
                    singleton_errors += 1
                elif len(true_set) > 0 and len(pred_set) > 0 and pred_set != true_set:
                    false_merges += 1
                elif len(true_set) > 1 and pred_set != true_set:
                    multi_errors += 1

            t_elapsed = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)

            config_str = f"K={k}, HardNeg={h_neg}:1, Pol={best_pol}, Tau={best_t:.2f}"
            experiment_rows.append({
                'Configuration': config_str,
                'CandidateRecall': cand_rec,
                'Precision': val_eval['precision_macro'],
                'Recall': val_eval['recall_macro'],
                'MacroF05': val_eval['f05_macro'],
                'FalseMerges': false_merges,
                'SingletonErrors': singleton_errors,
                'MultiMatchErrors': multi_errors,
                'RuntimeSec': round(t_elapsed, 2),
                'PeakRAM_MB': round(peak_mb, 1)
            })

            print(f"Run [{config_str}] -> Macro F0.5: {val_eval['f05_macro']:.4f} (P: {val_eval['precision_macro']:.4f}, R: {val_eval['recall_macro']:.4f}) | SingErrs: {singleton_errors}", flush=True)

    tracemalloc.stop()
    res_df = pd.DataFrame(experiment_rows).sort_values(by='MacroF05', ascending=False)
    return res_df

def main():
    print("=== Launching Comprehensive Experiment Runner ===", flush=True)
    s1_tr, gt_tr, s1_val, gt_val, target_prep = load_and_preprocess_split(sample_size=5000)

    # 1. Candidate Recall Experiments
    recall_df, index = run_candidate_recall_experiments(s1_val, gt_val, target_prep)

    # 2. Missed Match Failure Analysis
    missed_df = run_missed_match_failure_analysis(s1_val, gt_val, target_prep, index, top_k=15)

    # 3. Systematic End-to-End Ablation Grid
    leaderboard_df = run_end_to_end_ablation_grid(s1_tr, gt_tr, s1_val, gt_val, target_prep)

    print("\n=======================================================", flush=True)
    print("              FINAL EXPERIMENT LEADERBOARD              ", flush=True)
    print("=======================================================", flush=True)
    print(leaderboard_df.to_string(index=False))

if __name__ == '__main__':
    main()
