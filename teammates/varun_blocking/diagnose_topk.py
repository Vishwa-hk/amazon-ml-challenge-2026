"""Quick test: what's the recall ceiling at higher top_k values?"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from config import (TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, MODEL_CONFIG, BLOCKING_CONFIG)
from preprocess import preprocess_df
from blocking import CandidateIndex
from evaluate import compute_candidate_recall_ceiling
from io_utils import read_full_tsv

print("Loading data...", flush=True)
s1 = read_full_tsv(TRAIN_S1_PATH).head(4000)
s2 = read_full_tsv(TRAIN_S2_PATH)
s3 = read_full_tsv(TRAIN_S3_PATH)
gt = read_full_tsv(TRAIN_GT_PATH)
gt = gt[gt['source1_entity_id'].isin(set(s1['entity_id']))]

gt_matched_ids = set()
for ids in gt['matched_entity_ids']:
    if ids and pd.notna(ids):
        for m in str(ids).split(','):
            m = m.strip()
            if m:
                gt_matched_ids.add(m)
s2 = s2[s2['entity_id'].isin(gt_matched_ids) | (s2.index < 20000)]
s3 = s3[s3['entity_id'].isin(gt_matched_ids) | (s3.index < 20000)]

s1_prep = preprocess_df(s1)
target_prep = pd.concat([preprocess_df(s2), preprocess_df(s3)], ignore_index=True)

print(f"S1: {len(s1_prep)}, Target: {len(target_prep)}", flush=True)

print("Building index...", flush=True)
index = CandidateIndex()
index.build(target_prep)

for k in [10, 15, 20, 30, 50]:
    cand_df, _ = index.query(s1_prep, top_k=k)
    recall = compute_candidate_recall_ceiling(cand_df, gt)
    cands_count = sum([len(str(c).split(',')) if c else 0 for c in cand_df['candidate_entity_ids']])
    avg_cands = cands_count / len(s1_prep)
    print(f"top_k={k:3d} | Recall Ceiling: {recall['candidate_recall_ceiling']:.4f} | "
          f"Captured: {recall['captured_true_pairs']}/{recall['total_true_pairs']} | "
          f"Avg Candidates/S1: {avg_cands:.2f}", flush=True)
