"""
test_blocking_ablations.py — Isolated Blocking Ablation Experiments on Full Targets.

Measures for each ablation:
- Candidate Recall@10, @20, @30, @50
- Average Candidates / S1
- Retrieval Latency
- Peak RAM (MB)
- Final Macro F0.5
"""
import os
import gc
import sys
import time
import re
import array
import collections
import tracemalloc
import numpy as np
import pandas as pd

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)

from typing import Dict, Set, Tuple, List, Optional
import config
from config import TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, BLOCKING_CONFIG
from io_utils import read_full_tsv
from preprocess import preprocess_df
from pipeline import load_or_cache_preprocessed
from evaluate import compute_candidate_recall_ceiling
from blocking import CandidateIndex
import psutil

# 1. Load Data
print("=== Isolated Blocking Ablation Benchmarks ===", flush=True)
print("Loading 5,000 S1 sample and full 10.3M target universe...", flush=True)

s1_raw = read_full_tsv(TRAIN_S1_PATH).head(5000)
gt_raw = read_full_tsv(TRAIN_GT_PATH)
gt_raw = gt_raw[gt_raw['source1_entity_id'].isin(set(s1_raw['entity_id']))].copy()

s1_prep = preprocess_df(s1_raw)
s2_prep = load_or_cache_preprocessed(TRAIN_S2_PATH, "train_s2_prep")
s3_prep = load_or_cache_preprocessed(TRAIN_S3_PATH, "train_s3_prep")
target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

print(f"S1: {len(s1_prep):,} rows | Target Universe: {len(target_prep):,} rows", flush=True)

class CandidateIndexEnhanced(CandidateIndex):
    """
    Enhanced CandidateIndex testing:
    1. (first_token, last_token) name composite key
    2. (first_token, house_number) cross-field key
    3. (phonetic_first, house_number) cross-field key
    4. Relaxed bucket caps for rare/medium tokens
    """
    def __init__(self,
                 enable_fl_key=True,
                 enable_fh_key=True,
                 enable_ph_key=True,
                 max_tok_cap=500,
                 max_hits_per_tok=60):
        super().__init__()
        self.enable_fl_key = enable_fl_key
        self.enable_fh_key = enable_fh_key
        self.enable_ph_key = enable_ph_key
        self.max_tok_cap = max_tok_cap
        self.max_hits_per_tok = max_hits_per_tok
        
        self.first_last_map: Dict[Tuple[str, str], array.array] = collections.defaultdict(lambda: array.array('I'))
        self.first_house_map: Dict[Tuple[str, str], array.array] = collections.defaultdict(lambda: array.array('I'))
        self.phonetic_house_enhanced_map: Dict[Tuple[str, str], array.array] = collections.defaultdict(lambda: array.array('I'))

    def build(self, target_df: pd.DataFrame):
        super().build(target_df)
        
        n_targets = len(self.target_ids)
        first_toks = target_df['first_token'].values
        last_toks = target_df['last_token'].values
        phonetic_firsts = target_df['phonetic_first'].values if 'phonetic_first' in target_df.columns else np.repeat('', n_targets)
        houses = target_df['house_number'].values
        
        for idx in range(n_targets):
            ftok = first_toks[idx]
            ltok = last_toks[idx]
            pfirst = phonetic_firsts[idx]
            house = houses[idx]
            
            # 1. First-Last Token Composite Key
            if self.enable_fl_key and ftok and ltok and len(ftok) >= 3 and len(ltok) >= 3 and ftok != ltok:
                self.first_last_map[(ftok, ltok)].append(idx)
                
            # 2. First Token + House Number Key
            if self.enable_fh_key and ftok and house and len(ftok) >= 3 and len(house) >= 2:
                self.first_house_map[(ftok, house)].append(idx)
                
            # 3. Phonetic First + House Number Key
            if self.enable_ph_key and pfirst and house and pfirst != '0000' and len(house) >= 2:
                self.phonetic_house_enhanced_map[(pfirst, house)].append(idx)

    def query(self, s1_df: pd.DataFrame, top_k: int = 20) -> Tuple[pd.DataFrame, Dict]:
        s1_ids = s1_df['entity_id'].values
        name_norms = s1_df['name_norm'].values
        name_aggs = s1_df['name_agg_norm'].values
        info_toks_list = s1_df['name_info_tokens'].values
        prefixes = s1_df['name_prefix_3'].values
        first_toks = s1_df['first_token'].values
        last_toks = s1_df['last_token'].values
        acronyms = s1_df['acronym'].values if 'acronym' in s1_df.columns else np.repeat('', len(s1_df))
        phonetic_firsts = s1_df['phonetic_first'].values if 'phonetic_first' in s1_df.columns else np.repeat('', len(s1_df))
        postals = s1_df['postal_code'].values
        houses = s1_df['house_number'].values

        target_ids_arr = self.target_ids
        records = []
        flat_evidence = {}

        for i in range(len(s1_ids)):
            s1id = s1_ids[i]
            name = name_norms[i]
            agg = name_aggs[i]
            info_toks = info_toks_list[i]
            pref = prefixes[i]
            ftok = first_toks[i]
            ltok = last_toks[i]
            acr = acronyms[i]
            pfirst = phonetic_firsts[i]
            post = postals[i]
            house = houses[i]

            cand_scores: Dict[int, float] = collections.defaultdict(float)
            cand_ev: Dict[int, Dict[str, float]] = collections.defaultdict(dict)

            # Stage A: Exact Name (6.5)
            if name and name in self.exact_map:
                for tid in self.exact_map[name][:35]:
                    cand_scores[tid] += 6.5
                    cand_ev[tid]['exact_name_hit'] = 6.5

            # Stage B: Aggressive Name (4.8)
            if agg and agg in self.agg_exact_map:
                for tid in self.agg_exact_map[agg][:25]:
                    cand_scores[tid] += 4.8
                    cand_ev[tid]['agg_exact_hit'] = 4.8

            # New Stage B2: First-Last Token Composite Key (4.5)
            if self.enable_fl_key and ftok and ltok and len(ftok) >= 3 and len(ltok) >= 3 and ftok != ltok:
                if (ftok, ltok) in self.first_last_map:
                    tids = self.first_last_map[(ftok, ltok)]
                    if len(tids) <= 300:
                        for tid in tids[:25]:
                            cand_scores[tid] += 4.5
                            cand_ev[tid]['first_last_hit'] = 4.5

            # Stage C: Informative Tokens (relaxed caps)
            if info_toks:
                for tok in info_toks.split():
                    if tok in self.token_map:
                        tids = self.token_map[tok]
                        n_tids = len(tids)
                        if n_tids <= 40:
                            for tid in tids:
                                cand_scores[tid] += 4.2
                                cand_ev[tid]['rare_token_hit'] = 4.2
                        elif n_tids <= self.max_tok_cap:
                            for tid in tids[:self.max_hits_per_tok]:
                                cand_scores[tid] += 2.8
                                cand_ev[tid]['rare_token_hit'] = 2.8
                        elif n_tids <= 3000:
                            for tid in tids[:20]:
                                cand_scores[tid] += 1.5
                                cand_ev[tid]['token_hit'] = 1.5

                # Stage D: Token Pairs / Bigrams (3.8)
                for pair_key in self._make_token_pairs(info_toks):
                    if pair_key in self.token_pair_map:
                        tids = self.token_pair_map[pair_key]
                        if len(tids) <= 800:
                            for tid in tids[:30]:
                                cand_scores[tid] += 3.8
                                cand_ev[tid]['token_pair_hit'] = 3.8

                # Stage E: Sorted Token Set Key (5.0)
                stk = self._make_sorted_token_key(info_toks)
                if stk and stk in self.sorted_token_set_map:
                    tids = self.sorted_token_set_map[stk]
                    if len(tids) <= 350:
                        for tid in tids[:30]:
                            cand_scores[tid] += 5.0
                            cand_ev[tid]['sorted_token_set_hit'] = 5.0

            # Stage F: Prefix Maps
            if pref and pref in self.prefix3_map:
                tids = self.prefix3_map[pref]
                n_tids = len(tids)
                if n_tids <= 200:
                    for tid in tids[:30]:
                        cand_scores[tid] += 1.8
                        cand_ev[tid]['prefix_hit'] = 1.8
                elif n_tids <= 3000:
                    for tid in tids[:15]:
                        cand_scores[tid] += 1.2
                        cand_ev[tid]['prefix_hit'] = 1.2

            # Stage G: Acronym Index (2.2)
            if acr and len(acr) >= 2 and acr in self.acronym_map:
                tids = self.acronym_map[acr]
                if len(tids) <= 300:
                    for tid in tids[:20]:
                        cand_scores[tid] += 2.2
                        cand_ev[tid]['acronym_hit'] = 2.2

            # Stage H: Last Token standalone (1.5)
            if ltok and len(ltok) >= 4 and ltok in self.last_token_map:
                tids = self.last_token_map[ltok]
                if len(tids) <= 400:
                    for tid in tids[:18]:
                        cand_scores[tid] += 1.5
                        cand_ev[tid]['last_token_hit'] = 1.5

            # Stage I: Postal + House (4.5) / Postal alone (1.6)
            if post and house and (post, house) in self.postal_house_map:
                for tid in self.postal_house_map[(post, house)][:30]:
                    cand_scores[tid] += 4.5
                    cand_ev[tid]['postal_house_hit'] = 4.5
            elif post and post in self.postal_map:
                tids = self.postal_map[post]
                if len(tids) <= 1500:
                    for tid in tids[:18]:
                        cand_scores[tid] += 1.6
                        cand_ev[tid]['postal_hit'] = 1.6

            # New Stage I2: First-Token + House Number Key (3.8)
            if self.enable_fh_key and ftok and house and len(ftok) >= 3 and len(house) >= 2:
                if (ftok, house) in self.first_house_map:
                    tids = self.first_house_map[(ftok, house)]
                    if len(tids) <= 200:
                        for tid in tids[:20]:
                            cand_scores[tid] += 3.8
                            cand_ev[tid]['first_house_hit'] = 3.8

            # New Stage I3: Phonetic First + House Number Key (3.0)
            if self.enable_ph_key and pfirst and house and pfirst != '0000' and len(house) >= 2:
                if (pfirst, house) in self.phonetic_house_enhanced_map:
                    tids = self.phonetic_house_enhanced_map[(pfirst, house)]
                    if len(tids) <= 150:
                        for tid in tids[:15]:
                            cand_scores[tid] += 3.0
                            cand_ev[tid]['phonetic_house_hit'] = 3.0

            # Stage J: House number alone (1.0)
            if house and len(house) >= 3 and house in self.house_map:
                tids = self.house_map[house]
                if len(tids) <= 200:
                    for tid in tids[:12]:
                        cand_scores[tid] += 1.0
                        cand_ev[tid]['house_only_hit'] = 1.0

            # Stage K: Cross-Field Keys (3.4)
            if ftok and post and (ftok, post) in self.name_postal_map:
                for tid in self.name_postal_map[(ftok, post)][:25]:
                    cand_scores[tid] += 3.4
                    cand_ev[tid]['cross_field_hit'] = 3.4
            if ftok and house and (ftok, house) in self.name_house_map:
                for tid in self.name_house_map[(ftok, house)][:25]:
                    cand_scores[tid] += 3.4
                    cand_ev[tid]['cross_field_hit'] = 3.4
            if ltok and post and (ltok, post) in self.last_token_postal_map:
                for tid in self.last_token_postal_map[(ltok, post)][:20]:
                    cand_scores[tid] += 3.4
                    cand_ev[tid]['cross_field_hit'] = 3.4

            # Stage M: Residual Fallback via Character Trigrams (if < 5 candidates)
            residual_min = 5
            if len(cand_scores) < residual_min and name and len(name) >= 4:
                trigrams = self._extract_char_trigrams(name)
                tri_hits: Dict[int, int] = collections.defaultdict(int)
                for tri in set(trigrams[:8]):
                    if tri in self.char_trigram_map:
                        tids = self.char_trigram_map[tri]
                        if len(tids) <= 200:
                            for tid in tids[:15]:
                                tri_hits[tid] += 1
                for tid, h_cnt in tri_hits.items():
                    if h_cnt >= 2:
                        sc = float(h_cnt) * 1.2
                        cand_scores[tid] += sc
                        cand_ev[tid]['residual_tfidf_hit'] = sc

            # Select Top-K Candidates
            if cand_scores:
                sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                cand_ids = [target_ids_arr[tid] for tid, _ in sorted_cands]
            else:
                cand_ids = []

            records.append({
                'source1_entity_id': s1id,
                'candidate_entity_ids': ",".join(cand_ids)
            })

        return pd.DataFrame(records), flat_evidence


def evaluate_index_ablation(name: str, index_obj, k_list=[10, 20, 30, 50]):
    print(f"\n--- Testing Ablation: {name} ---", flush=True)
    results = []
    process = psutil.Process()
    
    t0_build = time.time()
    if not index_obj.is_built:
        index_obj.build(target_prep)
    t_build = time.time() - t0_build
    ram_mb = process.memory_info().rss / 1024**2
    
    print(f"  Index Build Time: {t_build:.1f}s | Peak RAM: {ram_mb/1024:.2f} GB", flush=True)
    
    for k in k_list:
        t0_q = time.time()
        cand_df, _ = index_obj.query(s1_prep, top_k=k)
        t_q = time.time() - t0_q
        
        rec_info = compute_candidate_recall_ceiling(cand_df, gt_raw)
        
        n_cands_total = 0
        for cids in cand_df['candidate_entity_ids']:
            if cids and pd.notna(cids) and str(cids).strip():
                n_cands_total += len(str(cids).split(','))
        avg_cands = n_cands_total / max(len(s1_prep), 1)
        
        results.append({
            'Ablation': name,
            'K': k,
            'CandRecall': rec_info['candidate_recall_ceiling'],
            'Captured': rec_info['captured_true_pairs'],
            'Total': rec_info['total_true_pairs'],
            'Missed': rec_info['total_true_pairs'] - rec_info['captured_true_pairs'],
            'AvgCands': avg_cands,
            'QueryTime_s': t_q,
            'BuildTime_s': t_build,
            'PeakRAM_GB': ram_mb / 1024
        })
        print(f"  K={k:2d}: CandRecall={rec_info['candidate_recall_ceiling']:.4f} "
              f"({rec_info['captured_true_pairs']}/{rec_info['total_true_pairs']} captured, "
              f"avg {avg_cands:.1f} cands/S1, {t_q:.2f}s)", flush=True)
              
    return pd.DataFrame(results)

# Run Baseline and Enhanced Ablations
base_index = CandidateIndex()
base_df = evaluate_index_ablation("BASELINE (Current 15-Stage)", base_index)

enhanced_index = CandidateIndexEnhanced()
enhanced_df = evaluate_index_ablation("ENHANCED (FL-Key + FH-Key + Relaxed Caps)", enhanced_index)

all_df = pd.concat([base_df, enhanced_df], ignore_index=True)

print("\n=======================================================")
print("     ISOLATED BLOCKING ABLATION SUMMARY TABLE         ")
print("=======================================================")
print(all_df[['Ablation', 'K', 'CandRecall', 'Captured', 'Missed', 'AvgCands', 'QueryTime_s', 'BuildTime_s', 'PeakRAM_GB']].to_string(index=False))
