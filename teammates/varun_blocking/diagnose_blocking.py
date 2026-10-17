"""
Diagnostic script to analyze blocking recall misses and find improvement opportunities.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import collections
from config import (TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, 
                    MODEL_CONFIG, BLOCKING_CONFIG)
from preprocess import preprocess_df
from blocking import CandidateIndex
from io_utils import read_full_tsv

print("Loading data...", flush=True)
s1 = read_full_tsv(TRAIN_S1_PATH)
s2 = read_full_tsv(TRAIN_S2_PATH)
s3 = read_full_tsv(TRAIN_S3_PATH)
gt = read_full_tsv(TRAIN_GT_PATH)

# Sample for speed
SAMPLE = 4000
s1 = s1.head(SAMPLE)
gt = gt[gt['source1_entity_id'].isin(set(s1['entity_id']))]

gt_matched_ids = set()
for ids in gt['matched_entity_ids']:
    if ids and pd.notna(ids):
        for m in str(ids).split(','):
            m = m.strip()
            if m:
                gt_matched_ids.add(m)

s2 = s2[s2['entity_id'].isin(gt_matched_ids) | (s2.index < SAMPLE * 5)]
s3 = s3[s3['entity_id'].isin(gt_matched_ids) | (s3.index < SAMPLE * 5)]

print(f"S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT rows: {len(gt)}", flush=True)

s1_prep = preprocess_df(s1)
target_prep = pd.concat([preprocess_df(s2), preprocess_df(s3)], ignore_index=True)

print("Building index...", flush=True)
index = CandidateIndex()
index.build(target_prep)

print("Querying...", flush=True)
cand_df, ev_dict = index.query(s1_prep, top_k=10)

# Build candidate map
cand_map = {}
for _, row in cand_df.iterrows():
    s1id = row['source1_entity_id']
    cands = str(row['candidate_entity_ids']).split(',') if row['candidate_entity_ids'] else []
    cand_map[s1id] = set([c.strip() for c in cands if c.strip()])

# Parse GT
gt_map = {}
for _, row in gt.iterrows():
    s1id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
    gt_map[s1id] = set([m.strip() for m in matches if m.strip()])

# Analyze misses
total_true = 0
captured = 0
missed_pairs = []

# Build target lookup for analysis
target_lookup = {}
for _, row in target_prep.iterrows():
    target_lookup[row['entity_id']] = row

s1_lookup = {}
for _, row in s1_prep.iterrows():
    s1_lookup[row['entity_id']] = row

for s1id, true_set in gt_map.items():
    total_true += len(true_set)
    cand_set = cand_map.get(s1id, set())
    captured += len(true_set & cand_set)
    
    for m in true_set:
        if m not in cand_set:
            missed_pairs.append((s1id, m))

recall = captured / total_true if total_true > 0 else 1.0
print(f"\nCandidate Recall: {recall:.4f}")
print(f"Total true pairs: {total_true}")
print(f"Captured: {captured}")
print(f"Missed: {len(missed_pairs)}")

# Analyze WHY misses happen
print(f"\n=== Analyzing {min(len(missed_pairs), 50)} missed pairs ===")

miss_reasons = collections.Counter()
for s1id, tid in missed_pairs[:50]:
    s1_row = s1_lookup.get(s1id)
    t_row = target_lookup.get(tid)
    
    if s1_row is None or t_row is None:
        miss_reasons['target_not_in_pool'] += 1
        if t_row is None:
            print(f"\n  MISS: {s1id} -> {tid}: TARGET NOT IN PREPROCESSED POOL")
        continue
    
    s1_name = s1_row.get('name_norm', '')
    t_name = t_row.get('name_norm', '')
    s1_name_agg = s1_row.get('name_agg_norm', '')
    t_name_agg = t_row.get('name_agg_norm', '')
    s1_info = s1_row.get('name_info_tokens', '')
    t_info = t_row.get('name_info_tokens', '')
    s1_prefix = s1_row.get('name_prefix_3', '')
    t_prefix = t_row.get('name_prefix_3', '')
    s1_postal = s1_row.get('postal_code', '')
    t_postal = t_row.get('postal_code', '')
    s1_house = s1_row.get('house_number', '')
    t_house = t_row.get('house_number', '')
    s1_first = s1_row.get('first_token', '')
    t_first = t_row.get('first_token', '')
    s1_last = s1_row.get('last_token', '')
    t_last = t_row.get('last_token', '')
    s1_phonetic = s1_row.get('phonetic_first', '')
    t_phonetic = t_row.get('phonetic_first', '')
    
    reasons = []
    
    # Check each blocking stage
    exact_match = (s1_name == t_name and s1_name)
    agg_match = (s1_name_agg == t_name_agg and s1_name_agg)
    
    s1_toks = set(s1_info.split()) if s1_info else set()
    t_toks = set(t_info.split()) if t_info else set()
    common_toks = s1_toks & t_toks
    
    prefix_match = (s1_prefix == t_prefix and len(s1_prefix) >= 3)
    postal_house_match = (s1_postal == t_postal and s1_house == t_house and s1_postal and s1_house)
    postal_match = (s1_postal == t_postal and s1_postal)
    cross_name_postal = (s1_first == t_first and s1_postal == t_postal and s1_first and s1_postal)
    cross_name_house = (s1_first == t_first and s1_house == t_house and s1_first and s1_house)
    cross_last_postal = (s1_last == t_last and s1_postal == t_postal and s1_last and s1_postal)
    phonetic_postal = (s1_phonetic == t_phonetic and s1_postal == t_postal and s1_phonetic and s1_postal)
    
    if not exact_match:
        reasons.append('no_exact')
    if not agg_match:
        reasons.append('no_agg')
    if not common_toks:
        reasons.append('no_common_tokens')
    if not prefix_match:
        reasons.append('no_prefix')
    if not postal_match:
        reasons.append('no_postal')
    if not cross_name_postal:
        reasons.append('no_cross_name_postal')
    
    # Was this entity in any blocking bucket at all?
    any_overlap = (exact_match or agg_match or len(common_toks) > 0 or prefix_match or 
                   postal_house_match or cross_name_postal or cross_name_house or 
                   cross_last_postal or phonetic_postal)
    
    if not any_overlap:
        miss_reasons['no_blocking_overlap'] += 1
    else:
        miss_reasons['in_bucket_but_capped'] += 1
    
    for r in reasons:
        miss_reasons[r] += 1
    
    print(f"\n  MISS: S1='{s1_row.get('business_name', '')}' -> Target='{t_row.get('business_name', '')}'")
    print(f"    name_norm: '{s1_name}' vs '{t_name}'")
    print(f"    name_agg:  '{s1_name_agg}' vs '{t_name_agg}'")
    print(f"    info_toks: '{s1_info}' vs '{t_info}'")
    print(f"    prefix_3:  '{s1_prefix}' vs '{t_prefix}'")
    print(f"    postal:    '{s1_postal}' vs '{t_postal}'")
    print(f"    house:     '{s1_house}' vs '{t_house}'")
    print(f"    first_tok: '{s1_first}' vs '{t_first}'")
    print(f"    phonetic:  '{s1_phonetic}' vs '{t_phonetic}'")
    print(f"    common_toks: {common_toks}")
    print(f"    any_overlap: {any_overlap}")
    print(f"    reasons: {reasons}")

print(f"\n=== Miss Reason Summary (top {min(len(missed_pairs), 50)} misses) ===")
for reason, count in miss_reasons.most_common():
    print(f"  {reason}: {count}")
