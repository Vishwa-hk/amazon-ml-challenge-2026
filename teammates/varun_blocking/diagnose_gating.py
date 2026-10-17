"""Quick check: how many candidate pairs have blocking_score < 4.0 and get gated."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from config import (TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, MODEL_CONFIG, BLOCKING_CONFIG)
from preprocess import preprocess_df
from blocking import CandidateIndex
from features import extract_pairwise_features_vectorized as extract_pairwise_features
from io_utils import read_full_tsv

print("Loading data...", flush=True)
s1 = read_full_tsv(TRAIN_S1_PATH).head(2000)
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
s2 = s2[s2['entity_id'].isin(gt_matched_ids) | (s2.index < 10000)]
s3 = s3[s3['entity_id'].isin(gt_matched_ids) | (s3.index < 10000)]

s1_prep = preprocess_df(s1)
target_prep = pd.concat([preprocess_df(s2), preprocess_df(s3)], ignore_index=True)

print("Building blocking index...", flush=True)
index = CandidateIndex()
index.build(target_prep)
cand_df, ev_dict = index.query(s1_prep, top_k=10)

print("Extracting features...", flush=True)
feat_df = extract_pairwise_features(cand_df, s1_prep, target_prep, evidence_dict=ev_dict)

# Parse GT pairs
gt_pairs = set()
for _, row in gt.iterrows():
    s1id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
    for m in matches:
        m = m.strip()
        if m:
            gt_pairs.add((s1id, m))

feat_df['is_true_match'] = feat_df.apply(
    lambda r: 1 if (r['source1_entity_id'], r['candidate_id']) in gt_pairs else 0, axis=1
)

total = len(feat_df)
true_matches = feat_df[feat_df['is_true_match'] == 1]
gated = feat_df[feat_df['blocking_score'] < 4.0]
gated_true = true_matches[true_matches['blocking_score'] < 4.0]

print(f"\nTotal candidate pairs: {total}")
print(f"True match pairs in candidate set: {len(true_matches)}")
print(f"Pairs with blocking_score < 4.0 (GATED): {len(gated)} ({100*len(gated)/total:.1f}%)")
print(f"TRUE MATCHES with blocking_score < 4.0 (GATED): {len(gated_true)} ({100*len(gated_true)/len(true_matches):.1f}% of true matches)")

print(f"\nBlocking score distribution (true matches):")
for threshold in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
    count = len(true_matches[true_matches['blocking_score'] >= threshold])
    print(f"  >= {threshold}: {count} ({100*count/len(true_matches):.1f}%)")

print(f"\nBlocking score distribution (all candidates):")
for threshold in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
    count = len(feat_df[feat_df['blocking_score'] >= threshold])
    print(f"  >= {threshold}: {count} ({100*count/total:.1f}%)")

# Check what features look like for gated true matches
if len(gated_true) > 0:
    print(f"\nGated true matches - features that are zeroed:")
    for col in ['name_lev_sim', 'name_char_jaccard', 'first_token_sim', 'last_token_sim', 'addr_lev_sim']:
        zero_count = (gated_true[col] == 0.0).sum()
        print(f"  {col}: {zero_count}/{len(gated_true)} are zero ({100*zero_count/len(gated_true):.0f}%)")
