import os
import sys
import gc
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    CACHE_DIR, OUTPUT_DIR, MODEL_CONFIG, BLOCKING_CONFIG, PROJECT_ROOT
)
from io_utils import read_full_tsv, save_parquet, load_parquet
from preprocess import preprocess_df
from blocking import CandidateIndex, generate_candidates
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from model import build_training_pairs, train_model, predict_match_probabilities, FEATURE_COLS
from decision import generate_matching_results, optimize_decision_threshold
from evaluate import compute_macro_f05, compute_candidate_recall_ceiling

VALIDATE_SCRIPT = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\utils\validate_submission.py")
TEST_DATASET_DIR = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\dataset\test")

def load_or_cache_preprocessed(tsv_path: Path, cache_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads preprocessed DataFrame from parquet cache if present, otherwise computes and caches it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_name}.parquet"
    
    if nrows is None and cache_file.exists():
        print(f"Loading cached {cache_name} from {cache_file.name}...", flush=True)
        return load_parquet(cache_file)
        
    print(f"Reading and preprocessing {tsv_path.name} (nrows={nrows})...", flush=True)
    t0 = time.time()
    df_raw = read_full_tsv(tsv_path) if nrows is None else pd.read_csv(tsv_path, sep="\t", nrows=nrows, dtype=str, keep_default_na=False)
    prep_df = preprocess_df(df_raw)
    print(f"  Preprocessed {len(prep_df):,} rows in {time.time()-t0:.2f}s", flush=True)
    
    if nrows is None:
        print(f"  Saving cache to {cache_file.name}...", flush=True)
        save_parquet(prep_df, cache_file)
    return prep_df

def _log_memory(stage_name: str):
    try:
        import psutil
        process = psutil.Process()
        print(f"[{stage_name}] Peak RSS: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    except ImportError:
        pass

def run_pipeline(
    train_sample_size: int = None,
    test_sample_size: int = None,
    skip_test: bool = False,
    top_k: int = BLOCKING_CONFIG.get('top_k', 30),
    hard_neg_ratio: int = MODEL_CONFIG.get('num_hard_negatives_per_pos', 8),
    debug_reduced_targets: bool = False
) -> Dict:
    """
    Executes end-to-end Entity Resolution Pipeline with zero leakage.
    Supports FAST_EXPERIMENT mode (via train_sample_size) and FULL_VALIDATION mode.
    """
    print("=================================================================", flush=True)
    print("  LAUNCHING AMAZON ML CHALLENGE 2026 ENTITY RESOLUTION PIPELINE  ", flush=True)
    print("=================================================================", flush=True)

    # 1. Load S1 Train & Ground Truth
    print("\n--- Step 1: Loading Training Reference Population (Source 1) ---", flush=True)
    s1_train_full = read_full_tsv(TRAIN_S1_PATH)
    gt_train_full = read_full_tsv(TRAIN_GT_PATH)

    if debug_reduced_targets:
        mode_label = "[DEBUG-REDUCED-TARGET]"
    elif train_sample_size and train_sample_size < len(s1_train_full):
        mode_label = "[FULL-TARGET-SAMPLED-S1]"
    else:
        mode_label = "[FULL-SCALE]"

    print(f"Current Mode: {mode_label}", flush=True)

    if train_sample_size and train_sample_size < len(s1_train_full):
        print(f"FAST_EXPERIMENT MODE: Sampling {train_sample_size:,} S1 entities...", flush=True)
        s1_train_df = s1_train_full.head(train_sample_size).copy()
        gt_train_df = gt_train_full[gt_train_full['source1_entity_id'].isin(set(s1_train_df['entity_id']))].copy()
    else:
        print(f"FULL DATA MODE: Using all {len(s1_train_full):,} S1 entities.", flush=True)
        s1_train_df = s1_train_full
        gt_train_df = gt_train_full

    s1_train_prep = preprocess_df(s1_train_df)
    _log_memory("After Loading S1")

    # 2. Entity-Level Validation Split (80/20) with NO entity leakage
    print("\n--- Step 2: Strict Entity-Level Train/Validation Split (80/20) ---", flush=True)
    s1_ids = s1_train_prep['entity_id'].unique()
    n_val = int(len(s1_ids) * MODEL_CONFIG['val_entity_split'])
    
    np.random.seed(MODEL_CONFIG['random_state'])
    shuffled_s1_ids = np.random.permutation(s1_ids)
    val_s1_ids = set(shuffled_s1_ids[:n_val])
    train_s1_ids = set(shuffled_s1_ids[n_val:])

    s1_tr = s1_train_prep[s1_train_prep['entity_id'].isin(train_s1_ids)].copy()
    gt_tr = gt_train_df[gt_train_df['source1_entity_id'].isin(train_s1_ids)].copy()

    s1_val = s1_train_prep[s1_train_prep['entity_id'].isin(val_s1_ids)].copy()
    gt_val = gt_train_df[gt_train_df['source1_entity_id'].isin(val_s1_ids)].copy()
    print(f"Train split S1 entities: {len(s1_tr):,} | Validation split S1 entities: {len(s1_val):,}", flush=True)

    # 3. Load & Preprocess Target Records (Source 2 + Source 3)
    print("\n--- Step 3: Loading Target Knowledge Base (Source 2 + Source 3) ---", flush=True)
    target_nrows = None
    if debug_reduced_targets and train_sample_size:
        target_nrows = train_sample_size * 25 # Clean un-leaked target pool for fast mode
        
    s2_prep = load_or_cache_preprocessed(TRAIN_S2_PATH, "train_s2_prep", nrows=target_nrows)
    s3_prep = load_or_cache_preprocessed(TRAIN_S3_PATH, "train_s3_prep", nrows=target_nrows)
    target_train_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    print(f"Total Target Records in Index: {len(target_train_prep):,}", flush=True)
    _log_memory("After Loading Targets")

    # 4. Build CandidateIndex & Multi-Stage Retrieval
    print("\n--- Step 4: Inverted Index Build & Candidate Retrieval ---", flush=True)
    t0_idx = time.time()
    index = CandidateIndex()
    index.build(target_train_prep)
    print(f"CandidateIndex built in {time.time()-t0_idx:.2f}s", flush=True)
    _log_memory("After Building CandidateIndex")

    print(f"Querying train candidates (top_k={top_k})...", flush=True)
    cand_tr, ev_tr = index.query(s1_tr, top_k=top_k)
    recall_tr = compute_candidate_recall_ceiling(cand_tr, gt_tr)
    avg_cands_tr = sum(len(c.split(',')) if c else 0 for c in cand_tr['candidate_entity_ids']) / max(len(s1_tr), 1)
    print(f"  Train Candidate Recall Ceiling: {recall_tr['candidate_recall_ceiling']:.4f} | Avg Candidates/S1: {avg_cands_tr:.2f}", flush=True)

    print(f"Querying validation candidates (top_k={top_k})...", flush=True)
    cand_val, ev_val = index.query(s1_val, top_k=top_k)
    recall_val = compute_candidate_recall_ceiling(cand_val, gt_val)
    avg_cands_val = sum(len(c.split(',')) if c else 0 for c in cand_val['candidate_entity_ids']) / max(len(s1_val), 1)
    print(f"  Val Candidate Recall Ceiling:   {recall_val['candidate_recall_ceiling']:.4f} | Avg Candidates/S1: {avg_cands_val:.2f}", flush=True)

    # 5. Extract Pairwise Features
    print("\n--- Step 5: Vectorized Pairwise Feature Extraction ---", flush=True)
    t0_feat = time.time()
    feat_tr = extract_pairwise_features(cand_tr, s1_tr, target_train_prep, evidence_dict=ev_tr)
    feat_val = extract_pairwise_features(cand_val, s1_val, target_train_prep, evidence_dict=ev_val)
    print(f"Extracted features for {len(feat_tr):,} train pairs and {len(feat_val):,} val pairs in {time.time()-t0_feat:.2f}s", flush=True)
    _log_memory("After Feature Extraction")

    # 6. Hard Negative Mining & Model Training
    print("\n--- Step 6: Hard-Negative Mining & LightGBM Training ---", flush=True)
    MODEL_CONFIG['num_hard_negatives_per_pos'] = hard_neg_ratio
    X_tr, y_tr = build_training_pairs(feat_tr, gt_tr)
    X_v, y_v = build_training_pairs(feat_val, gt_val)
    print(f"Training dataset: {len(X_tr):,} labeled pairs ({y_tr.sum():,} positive, {len(X_tr)-y_tr.sum():,} mined hard negatives)", flush=True)

    model = train_model(X_tr, y_tr, X_v, y_v)
    _log_memory("After Model Training")

    # 7. Model Calibration on Validation Set
    print("\n--- Step 7: Probability Calibration & Validation Scoring ---", flush=True)
    prob_val = predict_match_probabilities(model, feat_val)
    _log_memory("After Prediction")

    # 8. Decision Policy & Threshold Optimization
    print("\n--- Step 8: Entity Decision Engine & Metric Optimization ---", flush=True)
    val_s1_list = list(s1_val['entity_id'].values)
    best_thresh, best_val_f05, best_policy, best_margin = optimize_decision_threshold(
        prob_val, gt_val, val_s1_list
    )

    val_matching = generate_matching_results(
        prob_val, val_s1_list, threshold=best_thresh, policy=best_policy, margin=best_margin
    )
    final_val_eval = compute_macro_f05(val_matching, gt_val)

    print(f"\n=======================================================", flush=True)
    print(f"    VALIDATION BENCHMARK RESULTS {mode_label}    ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  Selected Policy:              Policy {best_policy}", flush=True)
    print(f"  Optimal Probability Threshold: {best_thresh:.2f}", flush=True)
    print(f"  Optimal Decision Margin:      {best_margin:.2f}", flush=True)
    print(f"  Validation Candidate Recall:  {recall_val['candidate_recall_ceiling']:.4f}", flush=True)
    print(f"  Validation Macro Precision:   {final_val_eval['precision_macro']:.4f}", flush=True)
    print(f"  Validation Macro Recall:      {final_val_eval['recall_macro']:.4f}", flush=True)
    print(f"  OFFICIAL MACRO F0.5 SCORE:    {final_val_eval['f05_macro']:.4f}", flush=True)
    print(f"  Singleton Accuracy:           {final_val_eval.get('singleton_accuracy', 0.0):.4f} ({final_val_eval.get('singleton_false_positives', 0)} FP)", flush=True)
    print(f"  False Merges (Non-Singleton): {final_val_eval.get('false_merges', 0)}", flush=True)
    print(f"  Average Candidates / S1:      {avg_cands_val:.2f}", flush=True)
    print(f"=======================================================", flush=True)

    summary_metrics = {
        'best_threshold': best_thresh,
        'best_policy': best_policy,
        'best_margin': best_margin,
        'val_f05': final_val_eval['f05_macro'],
        'val_precision': final_val_eval['precision_macro'],
        'val_recall': final_val_eval['recall_macro'],
        'val_candidate_recall': recall_val['candidate_recall_ceiling'],
        'val_avg_candidates_per_s1': avg_cands_val,
        'singleton_accuracy': final_val_eval.get('singleton_accuracy', 0.0),
        'singleton_false_positives': final_val_eval.get('singleton_false_positives', 0),
        'false_merges': final_val_eval.get('false_merges', 0)
    }

    if skip_test:
        return summary_metrics

    # 9. Retrain on 100% of Training Labels
    print("\n--- Step 9: Retraining LightGBM on 100% of Labeled Training Data ---", flush=True)
    cand_all, ev_all = index.query(s1_train_prep, top_k=top_k)
    feat_all = extract_pairwise_features(cand_all, s1_train_prep, target_train_prep, evidence_dict=ev_all)
    X_all, y_all = build_training_pairs(feat_all, gt_train_df)
    final_model = train_model(X_all, y_all)
    print(f"Final model retrained on {len(X_all):,} labeled pairs.", flush=True)

    # 10. Test Inference & Streaming Output Generation
    print("\n--- Step 10: Full Test Candidate Generation & Inference ---", flush=True)
    del index, target_train_prep, s2_prep, s3_prep
    gc.collect()

    s1_test_raw = read_full_tsv(TEST_S1_PATH)
    if test_sample_size and test_sample_size < len(s1_test_raw):
        print(f"Sampling test set to {test_sample_size:,} S1 entities...", flush=True)
        s1_test_raw = s1_test_raw.head(test_sample_size).copy()

    s1_test_prep = preprocess_df(s1_test_raw)
    
    test_target_nrows = None
    if debug_reduced_targets and test_sample_size:
        test_target_nrows = test_sample_size * 25
        
    s2_test_prep = load_or_cache_preprocessed(TEST_S2_PATH, "test_s2_prep", nrows=test_target_nrows)
    s3_test_prep = load_or_cache_preprocessed(TEST_S3_PATH, "test_s3_prep", nrows=test_target_nrows)
    target_test_prep = pd.concat([s2_test_prep, s3_test_prep], ignore_index=True)

    index_test = CandidateIndex()
    index_test.build(target_test_prep)

    # Process test in memory-safe chunks
    CHUNK_SIZE = 50000
    n_test_s1 = len(s1_test_prep)
    all_matching_records = []
    all_candidate_records = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    print(f"Generating test predictions in chunks of {CHUNK_SIZE:,}...", flush=True)
    for chunk_start in range(0, n_test_s1, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_test_s1)
        s1_chunk = s1_test_prep.iloc[chunk_start:chunk_end]
        
        cand_chunk, ev_chunk = index_test.query(s1_chunk, top_k=top_k)
        feat_chunk = extract_pairwise_features(cand_chunk, s1_chunk, target_test_prep, evidence_dict=ev_chunk)
        
        prob_chunk = predict_match_probabilities(final_model, feat_chunk)
        match_chunk = generate_matching_results(
            prob_chunk, list(s1_chunk['entity_id'].values),
            threshold=best_thresh, policy=best_policy, margin=best_margin
        )
        
        mode = "w" if chunk_start == 0 else "a"
        header = chunk_start == 0
        match_chunk.to_csv(matching_path, sep="\t", index=False, header=header, mode=mode, encoding="utf-8")
        cand_chunk.to_csv(candidate_path, sep="\t", index=False, header=header, mode=mode, encoding="utf-8")
        
        print(f"  Processed test chunk {chunk_start:,} - {chunk_end:,} / {n_test_s1:,}", flush=True)

    print(f"\nWrote final matching results to {matching_path}", flush=True)
    print(f"Wrote final candidate pairs to {candidate_path}", flush=True)

    summary_metrics['test_records_processed'] = n_test_s1
    return summary_metrics
