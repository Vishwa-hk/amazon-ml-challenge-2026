"""Diagnose entity-level blocking recall distribution."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import collections
from config import (TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, MODEL_CONFIG, BLOCKING_CONFIG)
from preprocess import preprocess_df
from blocking import CandidateIndex
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

print("Building index...", flush=True)
index = CandidateIndex()
index.build(target_prep)
cand_df, _ = index.query(s1_prep, top_k=15)

# Build maps
cand_map = {}
for _, row in cand_df.iterrows():
    s1id = row['source1_entity_id']
    cands = str(row['candidate_entity_ids']).split(',') if row['candidate_entity_ids'] else []
    cand_map[s1id] = set([c.strip() for c in cands if c.strip()])

gt_map = {}
for _, row in gt.iterrows():
    s1id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
    gt_map[s1id] = set([m.strip() for m in matches if m.strip()])

# Analyze entity-level recall distribution
n_singletons = 0
n_non_singletons = 0
n_all_captured = 0
n_some_captured = 0
n_none_captured = 0
entity_recalls = []

target_entity_ids = set(target_prep['entity_id'].values)

for s1id in s1['entity_id'].values:
    true_set = gt_map.get(s1id, set())
    cand_set = cand_map.get(s1id, set())
    
    if len(true_set) == 0:
        n_singletons += 1
        entity_recalls.append(1.0)  # singleton correctly predicted empty
        continue
    
    n_non_singletons += 1
    
    # How many true matches are in target pool?
    in_pool = sum(1 for m in true_set if m in target_entity_ids)
    captured = len(true_set & cand_set)
    
    if captured == len(true_set):
        n_all_captured += 1
    elif captured > 0:
        n_some_captured += 1
    else:
        n_none_captured += 1
    
    entity_recall = captured / len(true_set)
    entity_recalls.append(entity_recall)

macro_recall = np.mean(entity_recalls)
print(f"\n=== Entity-Level Blocking Recall Analysis ({len(s1)} S1 entities) ===")
print(f"Singletons (no true matches): {n_singletons} ({100*n_singletons/len(s1):.1f}%)")
print(f"Non-singletons: {n_non_singletons} ({100*n_non_singletons/len(s1):.1f}%)")
print(f"  ALL true matches captured: {n_all_captured} ({100*n_all_captured/n_non_singletons:.1f}%)")
print(f"  SOME true matches captured: {n_some_captured} ({100*n_some_captured/n_non_singletons:.1f}%)")
print(f"  ZERO true matches captured: {n_none_captured} ({100*n_none_captured/n_non_singletons:.1f}%)")
print(f"\nEntity-level macro recall (blocking ceiling): {macro_recall:.4f}")

# Distribution of per-entity recall
non_singleton_recalls = [r for r, s in zip(entity_recalls, s1['entity_id'].values) if len(gt_map.get(s, set())) > 0]
print(f"\nPer-entity recall distribution (non-singletons only):")
for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
    count = sum(1 for r in non_singleton_recalls if r >= t)
    print(f"  recall >= {t:.2f}: {count}/{len(non_singleton_recalls)} ({100*count/len(non_singleton_recalls):.1f}%)")

# Distribution of true match counts
match_counts = [len(gt_map.get(s1id, set())) for s1id in s1['entity_id'].values]
print(f"\nTrue match count distribution:")
for n in [0, 1, 2, 3, 5, 10, 20]:
    count = sum(1 for c in match_counts if c >= n)
    print(f"  >= {n} matches: {count} entities")

# Check: how many true matches are NOT in target pool at all?
not_in_pool = 0
total_true = 0
for s1id in s1['entity_id'].values:
    true_set = gt_map.get(s1id, set())
    total_true += len(true_set)
    for m in true_set:
        if m not in target_entity_ids:
            not_in_pool += 1

print(f"\nTrue matches NOT in target pool: {not_in_pool}/{total_true} ({100*not_in_pool/total_true:.1f}%)")
