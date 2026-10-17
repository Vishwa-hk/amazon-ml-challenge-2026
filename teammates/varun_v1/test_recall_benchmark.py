import os
import collections
import numpy as np
import pandas as pd
from preprocess import preprocess_df
from evaluate import compute_candidate_recall_ceiling
from blocking import generate_candidates

base = r'C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\dataset\train'

s1_raw = pd.read_csv(os.path.join(base, 'train_source1.tsv'), sep='\t', nrows=5000)
gt_raw = pd.read_csv(os.path.join(base, 'train_ground_truth.tsv'), sep='\t')
gt_raw = gt_raw[gt_raw['source1_entity_id'].isin(set(s1_raw['entity_id']))]

gt_matched_ids = set()
for ids in gt_raw['matched_entity_ids']:
    if ids:
        gt_matched_ids.update(str(ids).split(','))

s2_raw = pd.read_csv(os.path.join(base, 'train_source2.tsv'), sep='\t')
s3_raw = pd.read_csv(os.path.join(base, 'train_source3.tsv'), sep='\t')

# Include all true matched targets + 100k random target records
s2_sub = s2_raw[s2_raw['entity_id'].isin(gt_matched_ids) | (s2_raw.index < 100000)]
s3_sub = s3_raw[s3_raw['entity_id'].isin(gt_matched_ids) | (s3_raw.index < 100000)]

s1_prep = preprocess_df(s1_raw)
target_prep = pd.concat([preprocess_df(s2_sub), preprocess_df(s3_sub)], ignore_index=True)

print(f"S1 count: {len(s1_prep)} | Target count: {len(target_prep)}")

cand_df = generate_candidates(s1_prep, target_prep, top_k=35)
res = compute_candidate_recall_ceiling(cand_df, gt_raw)

print(f"CANDIDATE RECALL CEILING: {res['candidate_recall_ceiling']:.4f} ({res['captured_true_pairs']}/{res['total_true_pairs']})")
