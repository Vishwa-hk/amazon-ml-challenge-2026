import time, psutil
import pandas as pd
from config import CACHE_DIR, TRAIN_GT_PATH
from io_utils import load_parquet, read_full_tsv
from blocking import CandidateIndex
from evaluate import compute_candidate_recall_ceiling

print("Loading full target prep parquet (10.3M rows)...", flush=True)
t0 = time.time()
s2_prep = load_parquet(CACHE_DIR / 'train_s2_prep.parquet')
s3_prep = load_parquet(CACHE_DIR / 'train_s3_prep.parquet')
target_df = pd.concat([s2_prep, s3_prep], ignore_index=True)
print(f"Loaded {len(target_df):,} target rows in {time.time()-t0:.2f}s | RAM RSS: {psutil.Process().memory_info().rss/(1024**2):.1f} MB", flush=True)

del s2_prep, s3_prep

print("\nBuilding CandidateIndex over full 10.3M target knowledge base...", flush=True)
t_b0 = time.time()
index = CandidateIndex()
index.build(target_df)
print(f"CandidateIndex built in {time.time()-t_b0:.2f}s | RAM RSS: {psutil.Process().memory_info().rss/(1024**2):.1f} MB", flush=True)

print("\nLoading held-out S1 sample (3,000 entities) and Ground Truth...", flush=True)
s1_prep = load_parquet(CACHE_DIR / 'train_s1_prep.parquet').head(3000)
gt_all = read_full_tsv(TRAIN_GT_PATH)
gt_sample = gt_all[gt_all['source1_entity_id'].isin(set(s1_prep['entity_id']))]

print("\nEvaluating Candidate Recall Ceiling against FULL 10.3M targets across K values:", flush=True)
for k in [10, 15, 20, 25, 30]:
    t_q0 = time.time()
    cand_df, _ = index.query(s1_prep, top_k=k)
    q_time = time.time() - t_q0
    rec = compute_candidate_recall_ceiling(cand_df, gt_sample)
    avg_cands = sum(len(c.split(',')) if c else 0 for c in cand_df['candidate_entity_ids']) / len(s1_prep)
    print(f"  K={k:2d}: Candidate Recall Ceiling = {rec['candidate_recall_ceiling']:.4f} ({rec['captured_true_pairs']}/{rec['total_true_pairs']} pairs) | Avg Cands: {avg_cands:.2f} | Query time: {q_time:.2f}s", flush=True)
