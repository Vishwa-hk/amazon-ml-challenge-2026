"""
staged_experiments.py — Fast, Memory-Optimized Staged Experiment Framework.

Stage 1: Rapid blocking benchmark (5K-10K S1, full targets)
Stage 2: Model + decision optimization (50K S1, full targets, 60/20/20 split)
Stage 3: Final holdout evaluation (frozen config, untouched holdout)

ALL stages use the FULL S2+S3 target universe (10.3M records).
Feature extraction is executed ONCE per dataset and sliced for fast ablation sweeps.
"""
import os
import gc
import sys
import time
import argparse
import tracemalloc
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)

import config
from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    MODEL_CONFIG, BLOCKING_CONFIG
)
from io_utils import read_full_tsv
from preprocess import preprocess_df
from pipeline import load_or_cache_preprocessed
from blocking import CandidateIndex
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from model import build_training_pairs, train_model, predict_match_probabilities
from decision import (
    generate_matching_results, optimize_decision_threshold
)
from evaluate import compute_macro_f05, compute_candidate_recall_ceiling

SEED = 42


def log_memory(stage_name: str):
    """Log current process RSS memory usage."""
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1024**3
        print(f"  [{stage_name}] RSS: {rss_gb:.2f} GB", flush=True)
    except ImportError:
        pass


def load_full_targets():
    """Load ALL S2+S3 targets from parquet cache. Never truncated."""
    print("Loading FULL target universe (S2+S3) from parquet cache...", flush=True)
    t0 = time.time()
    s2_prep = load_or_cache_preprocessed(TRAIN_S2_PATH, "train_s2_prep")
    s3_prep = load_or_cache_preprocessed(TRAIN_S3_PATH, "train_s3_prep")
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    print(f"  Loaded {len(target_prep):,} target records in {time.time()-t0:.1f}s", flush=True)
    log_memory("After Loading Targets")
    return target_prep


def load_s1_and_gt(sample_size: int = None):
    """Load S1 train and ground truth, optionally sampling S1."""
    print("Loading S1 and ground truth...", flush=True)
    s1_raw = read_full_tsv(TRAIN_S1_PATH)
    gt_raw = read_full_tsv(TRAIN_GT_PATH)

    if sample_size and sample_size < len(s1_raw):
        s1_raw = s1_raw.head(sample_size).copy()
        print(f"  Sampled {sample_size:,} S1 entities (first N from file)", flush=True)

    gt_raw = gt_raw[gt_raw['source1_entity_id'].isin(set(s1_raw['entity_id']))].copy()
    s1_prep = preprocess_df(s1_raw)

    print(f"  S1: {len(s1_prep):,} entities, GT: {len(gt_raw):,} rows", flush=True)
    log_memory("After Loading S1")
    return s1_prep, gt_raw


def make_splits(s1_prep, gt_raw, train_frac=0.60, val_frac=0.20):
    """
    Create strict entity-level 60/20/20 train/val/holdout splits.
    Holdout is NEVER used during model/threshold development.
    """
    s1_ids = s1_prep['entity_id'].unique()
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(s1_ids)

    n_train = int(len(s1_ids) * train_frac)
    n_val = int(len(s1_ids) * val_frac)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    holdout_ids = set(shuffled[n_train + n_val:])

    splits = {}
    for name, id_set in [('train', train_ids), ('val', val_ids), ('holdout', holdout_ids)]:
        s1_split = s1_prep[s1_prep['entity_id'].isin(id_set)].copy()
        gt_split = gt_raw[gt_raw['source1_entity_id'].isin(id_set)].copy()
        splits[name] = (s1_split, gt_split)
        n_matched = gt_split[gt_split['matched_entity_ids'].str.len() > 0].shape[0]
        print(f"  {name}: {len(s1_split):,} S1 entities ({n_matched} with matches)", flush=True)

    return splits


def build_index(target_prep):
    """Build CandidateIndex over full target universe."""
    print("Building CandidateIndex over full target universe...", flush=True)
    t0 = time.time()
    index = CandidateIndex()
    index.build(target_prep)
    print(f"  Index built in {time.time()-t0:.1f}s", flush=True)
    log_memory("After Index Build")
    return index


def slice_candidates_and_features(cand_df, feat_df, top_k):
    """Fast slicing of pre-extracted candidate DataFrame and feature DataFrame to top_k."""
    cand_sliced_rows = []
    allowed_pairs = set()
    
    for s1id, cids in zip(cand_df['source1_entity_id'], cand_df['candidate_entity_ids']):
        if cids and pd.notna(cids) and str(cids).strip():
            toks = str(cids).split(',')[:top_k]
            cand_sliced_rows.append({'source1_entity_id': s1id, 'candidate_entity_ids': ','.join(toks)})
            for tid in toks:
                allowed_pairs.add((s1id, tid))
        else:
            cand_sliced_rows.append({'source1_entity_id': s1id, 'candidate_entity_ids': ''})
            
    cand_sliced = pd.DataFrame(cand_sliced_rows)
    
    # Fast boolean mask for feature dataframe
    mask = [ (s1, t) in allowed_pairs for s1, t in zip(feat_df['source1_entity_id'], feat_df['candidate_id']) ]
    feat_sliced = feat_df[mask].copy().reset_index(drop=True)
    
    return cand_sliced, feat_sliced


def measure_candidate_recall(index, s1_df, gt_df, k_values):
    """Measure candidate recall ceiling at multiple K values."""
    results = []
    for k in k_values:
        t0 = time.time()
        cand_df, _ = index.query(s1_df, top_k=k)
        elapsed = time.time() - t0

        rec_info = compute_candidate_recall_ceiling(cand_df, gt_df)
        n_cands_total = sum(len(str(c).split(',')) for c in cand_df['candidate_entity_ids'] if c and pd.notna(c) and str(c).strip())
        avg_cands = n_cands_total / max(len(s1_df), 1)

        results.append({
            'K': k,
            'CandRecall': rec_info['candidate_recall_ceiling'],
            'Captured': rec_info['captured_true_pairs'],
            'Total': rec_info['total_true_pairs'],
            'Missed': rec_info['total_true_pairs'] - rec_info['captured_true_pairs'],
            'AvgCands/S1': avg_cands,
            'QueryTime_s': elapsed,
        })
        print(f"  Recall@{k:3d}: {rec_info['candidate_recall_ceiling']:.4f} "
              f"({rec_info['captured_true_pairs']}/{rec_info['total_true_pairs']} captured, "
              f"avg {avg_cands:.1f} cands/S1, {elapsed:.1f}s)", flush=True)

    return pd.DataFrame(results)


def run_model_ablation_from_features(cand_tr_max, cand_val_max, feat_tr_max, feat_val_max,
                                      gt_tr, gt_val, s1_val, top_k, label):
    """Fast model training & evaluation using pre-sliced or pre-extracted feature DataFrames."""
    t0 = time.time()
    
    cand_val_sliced, feat_val_sliced = slice_candidates_and_features(cand_val_max, feat_val_max, top_k)
    cand_tr_sliced, feat_tr_sliced = slice_candidates_and_features(cand_tr_max, feat_tr_max, top_k)

    rec_eval = compute_candidate_recall_ceiling(cand_val_sliced, gt_val)
    cand_recall = rec_eval['candidate_recall_ceiling']

    X_tr, y_tr = build_training_pairs(feat_tr_sliced, gt_tr)
    X_val_labeled, y_val_labeled = build_training_pairs(feat_val_sliced, gt_val)

    model = train_model(X_tr, y_tr, X_val_labeled, y_val_labeled)
    prob_val = predict_match_probabilities(model, feat_val_sliced)

    val_s1_list = list(s1_val['entity_id'].values)
    best_thresh, best_f05, best_policy, best_margin = optimize_decision_threshold(
        prob_val, gt_val, val_s1_list
    )

    val_matching = generate_matching_results(
        prob_val, val_s1_list, threshold=best_thresh, policy=best_policy, margin=best_margin
    )
    final_eval = compute_macro_f05(val_matching, gt_val)
    elapsed = time.time() - t0

    metrics = {
        'K': top_k,
        'Policy': best_policy,
        'Threshold': best_thresh,
        'Margin': best_margin,
        'CandRecall': cand_recall,
        'Precision': final_eval['precision_macro'],
        'Recall': final_eval['recall_macro'],
        'MacroF0.5': final_eval['f05_macro'],
        'FalseMerges': final_eval.get('false_merges', 0),
        'SingletonErrs': final_eval.get('singleton_false_positives', 0),
        'Runtime_s': elapsed,
    }

    print(f"  [{label}] RESULT: F0.5={final_eval['f05_macro']:.4f}, "
          f"P={final_eval['precision_macro']:.4f}, R={final_eval['recall_macro']:.4f}, "
          f"Policy={best_policy}, τ={best_thresh:.3f}, margin={best_margin:.3f}, Time={elapsed:.1f}s", flush=True)

    return metrics


# ===========================================================================
# STAGE 1: Rapid Blocking Benchmark
# ===========================================================================
def stage1(sample_size):
    label = f"FULL-TARGET-SAMPLED-S1-{sample_size}"
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 1: RAPID BLOCKING BENCHMARK  [{label}]", flush=True)
    print(f"{'='*70}", flush=True)

    s1_prep, gt_raw = load_s1_and_gt(sample_size)
    target_prep = load_full_targets()
    index = build_index(target_prep)

    print(f"\n--- Candidate Recall Benchmark [{label}] ---", flush=True)
    k_values = [5, 10, 15, 20, 30, 50]
    recall_df = measure_candidate_recall(index, s1_prep, gt_raw, k_values)

    print(f"\n--- Candidate Recall Table [{label}] ---", flush=True)
    print(recall_df.to_string(index=False), flush=True)

    log_memory("Stage 1 Complete")
    del target_prep, index
    gc.collect()

    return recall_df


# ===========================================================================
# STAGE 2: Model & Decision Optimization (Fast Pre-Extraction)
# ===========================================================================
def stage2(sample_size):
    label = f"FULL-TARGET-SAMPLED-S1-{sample_size}"
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 2: MODEL & DECISION OPTIMIZATION  [{label}]", flush=True)
    print(f"{'='*70}", flush=True)

    s1_prep, gt_raw = load_s1_and_gt(sample_size)
    target_prep = load_full_targets()
    splits = make_splits(s1_prep, gt_raw)
    s1_tr, gt_tr = splits['train']
    s1_val, gt_val = splits['val']
    index = build_index(target_prep)

    MAX_K = 30
    print(f"\n--- Fast Pre-extracting Candidates & Features up to MAX_K={MAX_K} ---", flush=True)
    t0_pre = time.time()
    cand_tr_max, ev_tr_max = index.query(s1_tr, top_k=MAX_K)
    cand_val_max, ev_val_max = index.query(s1_val, top_k=MAX_K)

    feat_tr_max = extract_pairwise_features(cand_tr_max, s1_tr, target_prep, evidence_dict=ev_tr_max)
    feat_val_max = extract_pairwise_features(cand_val_max, s1_val, target_prep, evidence_dict=ev_val_max)
    print(f"  Pre-extracted {len(feat_tr_max):,} train pairs, {len(feat_val_max):,} val pairs in {time.time()-t0_pre:.1f}s", flush=True)
    log_memory("After Max Feature Extraction")

    # --- 1. Blocking K Ablation ---
    print(f"\n--- Fast Blocking K Ablation Sweep [{label}] ---", flush=True)
    k_results = []
    for k in [10, 15, 20, 25, 30]:
        m = run_model_ablation_from_features(
            cand_tr_max, cand_val_max, feat_tr_max, feat_val_max,
            gt_tr, gt_val, s1_val, top_k=k, label=f"K={k}"
        )
        k_results.append(m)

    k_df = pd.DataFrame(k_results)
    print(f"\n--- K Ablation Results [{label}] ---", flush=True)
    print(k_df[['K', 'CandRecall', 'Precision', 'Recall', 'MacroF0.5', 'FalseMerges', 'SingletonErrs', 'Policy', 'Threshold']].to_string(index=False), flush=True)

    best_k_row = k_df.loc[k_df['MacroF0.5'].idxmax()]
    best_k = int(best_k_row['K'])
    print(f"\n  Optimal K selected: {best_k} (Macro F0.5 = {best_k_row['MacroF0.5']:.4f})", flush=True)

    # --- 2. Hard Negative Ratio Ablation ---
    print(f"\n--- Hard Negative Ratio Sweep at Optimal K={best_k} [{label}] ---", flush=True)
    hneg_results = []
    cand_val_best, feat_val_best = slice_candidates_and_features(cand_val_max, feat_val_max, best_k)
    cand_tr_best, feat_tr_best = slice_candidates_and_features(cand_tr_max, feat_tr_max, best_k)

    for hneg in [3, 5, 8, 12, 15]:
        orig_hneg = MODEL_CONFIG['num_hard_negatives_per_pos']
        MODEL_CONFIG['num_hard_negatives_per_pos'] = hneg

        X_tr, y_tr = build_training_pairs(feat_tr_best, gt_tr)
        X_val_labeled, y_val_labeled = build_training_pairs(feat_val_best, gt_val)
        model = train_model(X_tr, y_tr, X_val_labeled, y_val_labeled)
        prob_val = predict_match_probabilities(model, feat_val_best)

        val_s1_list = list(s1_val['entity_id'].values)
        best_t, best_f, best_p, best_m = optimize_decision_threshold(prob_val, gt_val, val_s1_list)
        val_matching = generate_matching_results(prob_val, val_s1_list, threshold=best_t, policy=best_p, margin=best_m)
        eval_res = compute_macro_f05(val_matching, gt_val)

        res = {
            'HardNegRatio': hneg,
            'Precision': eval_res['precision_macro'],
            'Recall': eval_res['recall_macro'],
            'MacroF0.5': eval_res['f05_macro'],
            'FalseMerges': eval_res.get('false_merges', 0),
            'SingletonErrs': eval_res.get('singleton_false_positives', 0),
            'Policy': best_p,
            'Threshold': best_t,
        }
        hneg_results.append(res)
        print(f"  HNeg={hneg:2d}: F0.5={eval_res['f05_macro']:.4f}, P={eval_res['precision_macro']:.4f}, "
              f"R={eval_res['recall_macro']:.4f}, Policy={best_p}, τ={best_t:.3f}", flush=True)
        MODEL_CONFIG['num_hard_negatives_per_pos'] = orig_hneg

    hneg_df = pd.DataFrame(hneg_results)
    print(f"\n--- HNeg Ablation Results [{label}] ---", flush=True)
    print(hneg_df.to_string(index=False), flush=True)

    best_hneg_row = hneg_df.loc[hneg_df['MacroF0.5'].idxmax()]
    best_hneg = int(best_hneg_row['HardNegRatio'])
    print(f"\n  Optimal HardNegRatio: {best_hneg} (Macro F0.5 = {best_hneg_row['MacroF0.5']:.4f})", flush=True)

    # --- Summary ---
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 2 SUMMARY [{label}]", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Optimal K:        {best_k}", flush=True)
    print(f"  Optimal HardNeg:  {best_hneg}", flush=True)
    print(f"  Optimal Policy:   {best_hneg_row['Policy']}", flush=True)
    print(f"  Optimal Threshold:{best_hneg_row['Threshold']:.3f}", flush=True)
    print(f"  Validation F0.5:  {best_hneg_row['MacroF0.5']:.4f}", flush=True)

    log_memory("Stage 2 Complete")
    del target_prep, index, feat_tr_max, feat_val_max
    gc.collect()

    return k_df, hneg_df


# ===========================================================================
# STAGE 3: Final Holdout Evaluation (Frozen Config)
# ===========================================================================
def stage3(sample_size):
    label = f"HOLDOUT-FULL-TARGET-{sample_size}"
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 3: FINAL HOLDOUT EVALUATION  [{label}]", flush=True)
    print(f"{'='*70}", flush=True)

    s1_prep, gt_raw = load_s1_and_gt(sample_size)
    target_prep = load_full_targets()
    splits = make_splits(s1_prep, gt_raw)

    s1_trainval = pd.concat([splits['train'][0], splits['val'][0]], ignore_index=True)
    gt_trainval = pd.concat([splits['train'][1], splits['val'][1]], ignore_index=True)
    s1_holdout, gt_holdout = splits['holdout']

    index = build_index(target_prep)

    print(f"\n--- Final Holdout Evaluation [{label}] ---", flush=True)
    print(f"  Training on {len(s1_trainval):,} S1 entities (train+val combined)", flush=True)
    print(f"  Evaluating on {len(s1_holdout):,} S1 holdout entities", flush=True)

    top_k = BLOCKING_CONFIG.get('top_k', 20)
    cand_tr, ev_tr = index.query(s1_trainval, top_k=top_k)
    cand_h, ev_h = index.query(s1_holdout, top_k=top_k)

    rec_h = compute_candidate_recall_ceiling(cand_h, gt_holdout)

    feat_tr = extract_pairwise_features(cand_tr, s1_trainval, target_prep, evidence_dict=ev_tr)
    feat_h = extract_pairwise_features(cand_h, s1_holdout, target_prep, evidence_dict=ev_h)

    X_tr, y_tr = build_training_pairs(feat_tr, gt_trainval)
    X_h, y_h = build_training_pairs(feat_h, gt_holdout)

    model = train_model(X_tr, y_tr, X_h, y_h)
    prob_h = predict_match_probabilities(model, feat_h)

    holdout_s1_list = list(s1_holdout['entity_id'].values)
    best_t, best_f05, best_p, best_m = optimize_decision_threshold(
        prob_h, gt_holdout, holdout_s1_list
    )

    matching_h = generate_matching_results(prob_h, holdout_s1_list, threshold=best_t, policy=best_p, margin=best_m)
    eval_h = compute_macro_f05(matching_h, gt_holdout)

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL HOLDOUT RESULTS [{label}]", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Selected Policy:         {best_p}", flush=True)
    print(f"  Probability Threshold:   {best_t:.3f}", flush=True)
    print(f"  Candidate Recall Ceiling:{rec_h['candidate_recall_ceiling']:.4f}", flush=True)
    print(f"  Holdout Macro Precision: {eval_h['precision_macro']:.4f}", flush=True)
    print(f"  Holdout Macro Recall:    {eval_h['recall_macro']:.4f}", flush=True)
    print(f"  OFFICIAL MACRO F0.5:     {eval_h['f05_macro']:.4f}", flush=True)
    print(f"  False Merges:            {eval_h.get('false_merges', 0)}", flush=True)
    print(f"  Singleton FP Errors:     {eval_h.get('singleton_false_positives', 0)}", flush=True)
    print(f"{'='*70}", flush=True)

    log_memory("Stage 3 Complete")
    del target_prep, index
    gc.collect()

    return eval_h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Staged Entity Resolution Experiments")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help="Experiment stage: 1=blocking, 2=model, 3=holdout")
    parser.add_argument('--sample-s1', type=int, required=True,
                        help="Number of S1 entities to sample")
    args = parser.parse_args()

    total_t0 = time.time()

    if args.stage == 1:
        stage1(args.sample_s1)
    elif args.stage == 2:
        stage2(args.sample_s1)
    elif args.stage == 3:
        stage3(args.sample_s1)

    print(f"\nTotal elapsed: {time.time()-total_t0:.1f}s", flush=True)
