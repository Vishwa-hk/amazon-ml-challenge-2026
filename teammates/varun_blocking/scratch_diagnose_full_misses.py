import time, collections
import pandas as pd
import numpy as np
from config import CACHE_DIR, TRAIN_GT_PATH
from io_utils import load_parquet, read_full_tsv
from blocking import CandidateIndex

print("Loading data...", flush=True)
s1_prep = load_parquet(CACHE_DIR / 'train_s1_prep.parquet').head(3000)
s2_prep = load_parquet(CACHE_DIR / 'train_s2_prep.parquet')
s3_prep = load_parquet(CACHE_DIR / 'train_s3_prep.parquet')
target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
del s2_prep, s3_prep

gt_all = read_full_tsv(TRAIN_GT_PATH)
gt_sample = gt_all[gt_all['source1_entity_id'].isin(set(s1_prep['entity_id']))]

print("Building CandidateIndex over 10.3M targets...", flush=True)
index = CandidateIndex()
index.build(target_prep)

print("Querying candidate pairs (K=30)...", flush=True)
cand_df, _ = index.query(s1_prep, top_k=30)

cand_map = {}
for _, row in cand_df.iterrows():
    s1id = row['source1_entity_id']
    cands = str(row['candidate_entity_ids']).split(',') if row['candidate_entity_ids'] else []
    cand_map[s1id] = set([c.strip() for c in cands if c.strip()])

gt_map = {}
for _, row in gt_sample.iterrows():
    s1id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
    gt_map[s1id] = set([m.strip() for m in matches if m.strip()])

# Build fast lookups for 3000 S1 and target records
s1_lookup = s1_prep.set_index('entity_id').to_dict('index')

missed_pairs = []
for s1id, true_set in gt_map.items():
    cand_set = cand_map.get(s1id, set())
    for m in true_set:
        if m not in cand_set:
            missed_pairs.append((s1id, m))

print(f"\nTotal missed pairs at K=30: {len(missed_pairs)}")

target_lookup = target_prep[target_prep['entity_id'].isin(set(m for _, m in missed_pairs))].set_index('entity_id').to_dict('index')

print(f"\n--- Analyzing First 30 Missed True Match Pairs ---")
for idx, (s1id, tid) in enumerate(missed_pairs[:30], 1):
    s1_row = s1_lookup.get(s1id, {})
    t_row = target_lookup.get(tid, {})
    
    print(f"\nMiss #{idx}: S1={s1id} -> Target={tid}")
    print(f"  S1 Name:     '{s1_row.get('business_name', '')}' | Norm: '{s1_row.get('name_norm', '')}' | Info: '{s1_row.get('name_info_tokens', '')}'")
    print(f"  Target Name: '{t_row.get('business_name', '')}' | Norm: '{t_row.get('name_norm', '')}' | Info: '{t_row.get('name_info_tokens', '')}'")
    print(f"  S1 Addr:     '{s1_row.get('business_address', '')}' | Post: '{s1_row.get('postal_code', '')}' | House: '{s1_row.get('house_number', '')}'")
    print(f"  Target Addr: '{t_row.get('business_address', '')}' | Post: '{t_row.get('postal_code', '')}' | House: '{t_row.get('house_number', '')}'")
