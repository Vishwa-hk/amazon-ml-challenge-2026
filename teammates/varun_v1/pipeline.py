import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    CACHE_DIR, OUTPUT_DIR, MODEL_CONFIG, PROJECT_ROOT
)
from io_utils import read_full_tsv, save_parquet, load_parquet
from preprocess import preprocess_df
from blocking import generate_candidates
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from model import build_training_pairs, train_model, predict_match_probabilities
from decision import generate_matching_results, optimize_decision_threshold
from evaluate import compute_macro_f05, compute_candidate_recall_ceiling

def run_pipeline(
    train_sample_size: int = None,
    test_sample_size: int = None,
    skip_test: bool = False
) -> Dict:
    """
    Executes end-to-end Entity Resolution Pipeline.
    Supports sample size capping for fast development/testing runs.
    """
    print("=== Step 1: Loading & Preprocessing Data ===", flush=True)
    s1_train = read_full_tsv(TRAIN_S1_PATH)
    s2_train = read_full_tsv(TRAIN_S2_PATH)
    s3_train = read_full_tsv(TRAIN_S3_PATH)
    gt_train = read_full_tsv(TRAIN_GT_PATH)

    if train_sample_size and train_sample_size < len(s1_train):
        print(f"Sampling training set to {train_sample_size} S1 entities...", flush=True)
        s1_train = s1_train.head(train_sample_size)
        gt_train = gt_train[gt_train['source1_entity_id'].isin(set(s1_train['entity_id']))]
        
        gt_matched_ids = set()
        for ids in gt_train['matched_entity_ids']:
            if ids:
                gt_matched_ids.update(str(ids).split(','))
                
        s2_train = s2_train[s2_train['entity_id'].isin(gt_matched_ids) | (s2_train.index < train_sample_size * 5)]
        s3_train = s3_train[s3_train['entity_id'].isin(gt_matched_ids) | (s3_train.index < train_sample_size * 5)]

    print(f"Training S1 records: {len(s1_train)} | S2: {len(s2_train)} | S3: {len(s3_train)}", flush=True)

    s1_train_prep = preprocess_df(s1_train)
    target_train_prep = pd.concat([preprocess_df(s2_train), preprocess_df(s3_train)], ignore_index=True)

    print("=== Step 2: Entity-Level Validation Split (80/20) ===", flush=True)
    s1_ids = s1_train_prep['entity_id'].unique()
    n_val = int(len(s1_ids) * MODEL_CONFIG['val_entity_split'])
    
    np_rand = pd.Series(s1_ids).sample(frac=1.0, random_state=MODEL_CONFIG['random_state']).values
    val_s1_ids = set(np_rand[:n_val])
    train_s1_ids = set(np_rand[n_val:])

    s1_tr = s1_train_prep[s1_train_prep['entity_id'].isin(train_s1_ids)].copy()
    gt_tr = gt_train[gt_train['source1_entity_id'].isin(train_s1_ids)].copy()

    s1_val = s1_train_prep[s1_train_prep['entity_id'].isin(val_s1_ids)].copy()
    gt_val = gt_train[gt_train['source1_entity_id'].isin(val_s1_ids)].copy()

    print(f"Train split S1 entities: {len(s1_tr)} | Validation split S1 entities: {len(s1_val)}", flush=True)

    print("=== Step 3: Candidate Generation (Blocking) & Recall Evaluation ===", flush=True)
    cand_tr = generate_candidates(s1_tr, target_train_prep)
    recall_tr = compute_candidate_recall_ceiling(cand_tr, gt_tr)
    print(f"Train Candidate Recall Ceiling: {recall_tr['candidate_recall_ceiling']:.4f} ({recall_tr['captured_true_pairs']}/{recall_tr['total_true_pairs']})", flush=True)

    cand_val = generate_candidates(s1_val, target_train_prep)
    recall_val = compute_candidate_recall_ceiling(cand_val, gt_val)
    print(f"Val Candidate Recall Ceiling:   {recall_val['candidate_recall_ceiling']:.4f} ({recall_val['captured_true_pairs']}/{recall_val['total_true_pairs']})", flush=True)

    print("=== Step 4: Extracting Pairwise Features ===", flush=True)
    feat_tr = extract_pairwise_features(cand_tr, s1_tr, target_train_prep)
    feat_val = extract_pairwise_features(cand_val, s1_val, target_train_prep)

    print("=== Step 5: Hard Negative Mining & Training LightGBM Classifier ===", flush=True)
    X_tr, y_tr = build_training_pairs(feat_tr, gt_tr)
    X_v, y_v = build_training_pairs(feat_val, gt_val)

    model = train_model(X_tr, y_tr, X_v, y_v)

    print("=== Step 6: Validation Decision & Threshold Optimization ===", flush=True)
    prob_val = predict_match_probabilities(model, feat_val)
    best_thresh, best_val_f05 = optimize_decision_threshold(
        prob_val, gt_val, list(s1_val['entity_id'])
    )

    val_matching = generate_matching_results(prob_val, list(s1_val['entity_id']), threshold=best_thresh)
    final_val_eval = compute_macro_f05(val_matching, gt_val)

    print(f"Optimal Probability Threshold: {best_thresh:.2f}", flush=True)
    print(f"Validation Macro F0.5 Score:   {final_val_eval['f05_macro']:.4f}", flush=True)
    print(f"Validation Macro Precision:   {final_val_eval['precision_macro']:.4f}", flush=True)
    print(f"Validation Macro Recall:      {final_val_eval['recall_macro']:.4f}", flush=True)

    if skip_test:
        return {
            'best_threshold': best_thresh,
            'val_f05': final_val_eval['f05_macro'],
            'val_precision': final_val_eval['precision_macro'],
            'val_recall': final_val_eval['recall_macro']
        }

    print("=== Step 7: Test Set Candidate Generation & Inference ===", flush=True)
    s1_test = read_full_tsv(TEST_S1_PATH)
    s2_test = read_full_tsv(TEST_S2_PATH)
    s3_test = read_full_tsv(TEST_S3_PATH)

    if test_sample_size and test_sample_size < len(s1_test):
        print(f"Sampling test set to {test_sample_size} S1 entities...", flush=True)
        s1_test = s1_test.head(test_sample_size)
        s2_test = s2_test.head(test_sample_size * 5)
        s3_test = s3_test.head(test_sample_size * 5)

    s1_test_prep = preprocess_df(s1_test)
    target_test_prep = pd.concat([preprocess_df(s2_test), preprocess_df(s3_test)], ignore_index=True)

    cand_test = generate_candidates(s1_test_prep, target_test_prep)
    feat_test = extract_pairwise_features(cand_test, s1_test_prep, target_test_prep)

    prob_test = predict_match_probabilities(model, feat_test)
    test_matching = generate_matching_results(prob_test, list(s1_test_prep['entity_id']), threshold=best_thresh)

    print("=== Step 8: Writing Output Files & Package Submission ===", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    test_matching.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")
    cand_test.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")

    print(f"Wrote {matching_path} ({len(test_matching)} rows)", flush=True)
    print(f"Wrote {candidate_path} ({len(cand_test)} rows)", flush=True)

    return {
        'best_threshold': best_thresh,
        'val_f05': final_val_eval['f05_macro'],
        'val_precision': final_val_eval['precision_macro'],
        'val_recall': final_val_eval['recall_macro'],
        'test_records_processed': len(s1_test_prep)
    }
