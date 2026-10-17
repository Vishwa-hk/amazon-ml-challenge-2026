"""
build_submission.py — Production Full-Scale Submission Generator & Validator.

Retrains on 100% of training data, runs full chunked test inference on all 1.73M S1 test entities,
writes output TSVs, executes official validator, and packages self-contained submission zip.
"""
import os
import sys
# Constrain thread pools to avoid OpenBLAS memory allocation failures on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time
import zipfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)

import config
from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    CACHE_DIR, OUTPUT_DIR, MODEL_CONFIG, BLOCKING_CONFIG, PROJECT_ROOT
)
from io_utils import read_full_tsv
from preprocess import preprocess_df
from pipeline import load_or_cache_preprocessed
from blocking import CandidateIndex
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from model import build_training_pairs, train_model, predict_match_probabilities
from decision import generate_matching_results, optimize_decision_threshold
from evaluate import compute_macro_f05

VALIDATE_SCRIPT = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\utils\validate_submission.py")
TEST_DATASET_DIR = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\dataset\test")


def log_memory(stage_name: str):
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1024**3
        print(f"[{stage_name}] Peak RSS: {rss_gb:.2f} GB", flush=True)
    except ImportError:
        pass


def main():
    print("=================================================================", flush=True)
    print("  AMAZON ML CHALLENGE 2026 — FULL SUBMISSION GENERATOR & VALIDATOR ", flush=True)
    print("=================================================================", flush=True)

    t0_total = time.time()

    # 1. Load All Training Reference Data
    print("\n--- Step 1: Loading Full Training Data (100K S1 Entities) ---", flush=True)
    s1_train_full = read_full_tsv(TRAIN_S1_PATH)
    gt_train_full = read_full_tsv(TRAIN_GT_PATH)
    s1_train_prep = preprocess_df(s1_train_full)
    log_memory("After Loading S1")

    # 2. Load Preprocessed Training Target Universe (10.3M records)
    print("\n--- Step 2: Loading Full Target Universe (Source 2 + Source 3) ---", flush=True)
    s2_prep = load_or_cache_preprocessed(TRAIN_S2_PATH, "train_s2_prep")
    s3_prep = load_or_cache_preprocessed(TRAIN_S3_PATH, "train_s3_prep")
    target_train_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    log_memory("After Loading Targets")

    # 3. Build Training CandidateIndex & Train Final Production Model
    print("\n--- Step 3: Candidate Retrieval & Feature Extraction for Training ---", flush=True)
    top_k = BLOCKING_CONFIG.get('top_k', 30)
    hard_neg_ratio = MODEL_CONFIG.get('num_hard_negatives_per_pos', 8)
    MODEL_CONFIG['num_hard_negatives_per_pos'] = hard_neg_ratio

    index_tr = CandidateIndex()
    index_tr.build(target_train_prep)
    log_memory("After Index Build")

    n_train_s1 = len(s1_train_prep)
    TRAIN_CHUNK_SIZE = 10000
    X_tr_list = []
    y_tr_list = []

    print(f"Querying {n_train_s1:,} training candidates & extracting features in chunks of {TRAIN_CHUNK_SIZE:,}...", flush=True)
    for chunk_start in range(0, n_train_s1, TRAIN_CHUNK_SIZE):
        chunk_end = min(chunk_start + TRAIN_CHUNK_SIZE, n_train_s1)
        s1_chunk = s1_train_prep.iloc[chunk_start:chunk_end]
        s1_chunk_ids = set(s1_chunk['entity_id'])
        gt_chunk = gt_train_full[gt_train_full['source1_entity_id'].isin(s1_chunk_ids)]

        cand_chunk, ev_chunk = index_tr.query(s1_chunk, top_k=top_k)
        feat_chunk = extract_pairwise_features(cand_chunk, s1_chunk, target_train_prep, evidence_dict=ev_chunk)
        X_c, y_c = build_training_pairs(feat_chunk, gt_chunk)
        
        if not X_c.empty:
            X_tr_list.append(X_c)
            y_tr_list.append(y_c)

        print(f"  Processed S1 Train Chunk {chunk_start:,}-{chunk_end:,} / {n_train_s1:,} ({len(X_c):,} pairs)", flush=True)
        del cand_chunk, ev_chunk, feat_chunk, X_c, y_c
        gc.collect()

    X_tr = pd.concat(X_tr_list, ignore_index=True)
    y_tr = pd.concat(y_tr_list, ignore_index=True)
    del X_tr_list, y_tr_list
    gc.collect()
    log_memory("After Feature Extraction Complete")

    # Clean up training index and targets to free ~14 GB RAM for test processing
    del index_tr, target_train_prep, s2_prep, s3_prep
    gc.collect()
    log_memory("After Freeing Train Memory")

    print(f"\n--- Step 4: Retraining Production LightGBM Model on {len(X_tr):,} Pairs ---", flush=True)
    MODEL_CONFIG['num_hard_negatives_per_pos'] = hard_neg_ratio
    final_model = train_model(X_tr, y_tr)
    log_memory("After Production Model Training")
    del X_tr, y_tr
    gc.collect()

    # 4. Load Test Data & Build Test CandidateIndex
    print("\n--- Step 5: Loading Test Data & Target Universe ---", flush=True)
    s1_test_raw = read_full_tsv(TEST_S1_PATH)
    s1_test_prep = preprocess_df(s1_test_raw)
    n_test_s1 = len(s1_test_prep)
    print(f"Loaded {n_test_s1:,} S1 test query entities.", flush=True)

    s2_test_prep = load_or_cache_preprocessed(TEST_S2_PATH, "test_s2_prep")
    s3_test_prep = load_or_cache_preprocessed(TEST_S3_PATH, "test_s3_prep")
    target_test_prep = pd.concat([s2_test_prep, s3_test_prep], ignore_index=True)
    print(f"Loaded {len(target_test_prep):,} target records in test universe.", flush=True)

    print("\n--- Step 6: Building CandidateIndex over Test Universe ---", flush=True)
    index_test = CandidateIndex()
    index_test.build(target_test_prep)
    log_memory("After Building Test Index")

    # 5. Full Streaming Test Inference in Chunks
    print("\n--- Step 7: Streaming Test Prediction Generation ---", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    CHUNK_SIZE = 50000
    # Empirically optimized decision rules from validation holdout
    best_policy = "M"
    best_thresh = 0.580
    best_margin = 0.000

    print(f"Processing {n_test_s1:,} test entities in chunks of {CHUNK_SIZE:,}...", flush=True)
    print(f"Applying Policy: {best_policy} | Threshold: {best_thresh:.3f} | Margin: {best_margin:.3f}", flush=True)

    t0_inf = time.time()
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

        elapsed_inf = time.time() - t0_inf
        rate = chunk_end / elapsed_inf
        eta_sec = (n_test_s1 - chunk_end) / max(rate, 1.0)
        print(f"  Chunk {chunk_start:,}-{chunk_end:,} / {n_test_s1:,} "
              f"({(chunk_end/n_test_s1)*100:.1f}%) | Speed: {rate:.0f} S1/s | ETA: {eta_sec/60:.1f} min", flush=True)

        gc.collect()

    print(f"\nWrote full matching results to {matching_path}", flush=True)
    print(f"Wrote full candidate pairs to {candidate_path}", flush=True)
    log_memory("After Test Inference Complete")

    # 6. Execute Official Validator Script
    print("\n--- Step 8: Running Official Submission Validator ---", flush=True)
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(TEST_DATASET_DIR)
    ]
    val_res = subprocess.run(cmd)

    if val_res.returncode != 0:
        print("\n[WARNING] Official validator reported formatting errors! Check log output above.", flush=True)
        sys.exit(val_res.returncode)

    print("\n[SUCCESS] All output files PASSED official submission validation!", flush=True)

    # 7. Package Self-Contained Submission Zip
    print("\n--- Step 9: Packaging Self-Contained Submission Package ---", flush=True)
    zip_path = PROJECT_ROOT / "submission.zip"
    
    python_files = [f for f in PROJECT_ROOT.glob("*.py")]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output TSVs
        zf.write(matching_path, arcname="output/matching_results.tsv")
        zf.write(candidate_path, arcname="output/candidate_pairs.tsv")
        # Add source code files
        for py_file in python_files:
            zf.write(py_file, arcname=f"code/{py_file.name}")

    zip_size_mb = os.path.getsize(zip_path) / 1024**2
    print(f"Created submission package at {zip_path.name} ({zip_size_mb:.2f} MB)", flush=True)

    total_time = time.time() - t0_total
    print(f"\n=======================================================", flush=True)
    print(f"  SUBMISSION BUILDING & VALIDATION COMPLETE ", flush=True)
    print(f"  Total Elapsed Time: {total_time/60:.2f} minutes", flush=True)
    print(f"=======================================================", flush=True)


if __name__ == "__main__":
    main()
