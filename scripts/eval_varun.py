"""
Evaluate Varun's pipeline (teammates/varun_v1) WITHOUT modifying his files.

The only intervention: his config.py points at his own machine's data folder, so
we overwrite those path constants in memory before his pipeline is imported.

Modes
  A  his own quick-test mode (run_pipeline(train_sample_size=N, skip_test=True)),
     exactly as his README documents it — reproduces the number he reports.
  B  realistic: his exact functions (preprocess, blocking, features, hard-negative
     mining, LightGBM, threshold search) and his exact 80/20 entity split, but
     against the FULL train S2/S3 pool (10.3M records, all decoys included), on a
     random sample of his train/validation S1 entities. Blocking and decisions are
     per-S1-entity, so sampling S1 entities with the full pool gives an unbiased
     estimate of his full-scale validation score. Also reports:
       - our scorer's F0.5 (cross-checked against his evaluator)
       - the best-possible-judge (oracle) ceiling on his validation short-lists
       - our baseline v1 on the SAME validation entities, for comparison

Usage:  .venv/Scripts/python.exe scripts/eval_varun.py --mode A --n 2000
        .venv/Scripts/python.exe scripts/eval_varun.py --mode B --n-train 40000 --n-val 20000
"""
import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pandas as pd

pd.set_option("future.infer_string", False)  # pandas-2 style object strings, as his code expects

ROOT = Path(__file__).resolve().parents[1]
VARUN = ROOT / "teammates" / "varun_v1"
DATA = ROOT / "student_resource" / "dataset"
sys.path.insert(0, str(VARUN))
sys.path.insert(1, str(ROOT / "code" / "business_entity_resolution" / "src"))

import config  # noqa: E402  (his config)

for split, names in (("train", ("source1", "source2", "source3", "ground_truth")), ("test", ("source1", "source2", "source3"))):
    for nm in names:
        attr = f"{split.upper()}_{'GT' if nm == 'ground_truth' else nm.replace('source', 'S')}_PATH"
        setattr(config, attr, DATA / split / f"{split}_{nm}.tsv")
config.OUTPUT_DIR = VARUN / "output"
config.CACHE_DIR = VARUN / "cache"

from scorer import score as our_score, parse_id_list_file  # noqa: E402


def peak_mem_gb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e9


def lists_to_dict(df, id_col, list_col):
    return {r[0]: set(x for x in (r[1] or "").split(",") if x) for r in df[[id_col, list_col]].itertuples(index=False)}


def mode_a(n):
    from pipeline import run_pipeline
    t = time.time()
    m = run_pipeline(train_sample_size=n, skip_test=True)
    print(f"\n[A] his quick-test mode, --sample-train {n}: {m}  ({time.time()-t:.0f}s, peak mem {peak_mem_gb():.1f} GB)")


def mode_b(n_train, n_val):
    from io_utils import read_full_tsv
    from preprocess import preprocess_df
    from blocking import generate_candidates
    from features import extract_pairwise_features_vectorized as feats
    from model import build_training_pairs, train_model, predict_match_probabilities
    from decision import generate_matching_results, optimize_decision_threshold
    from evaluate import compute_macro_f05, compute_candidate_recall_ceiling
    from config import MODEL_CONFIG

    t0 = time.time()
    s1 = read_full_tsv(config.TRAIN_S1_PATH)
    gt = read_full_tsv(config.TRAIN_GT_PATH)
    # his exact 80/20 entity split (same ids, same order, same seed)
    ids = s1["entity_id"].unique()
    n_v = int(len(ids) * MODEL_CONFIG["val_entity_split"])
    shuffled = pd.Series(ids).sample(frac=1.0, random_state=MODEL_CONFIG["random_state"]).values
    val_ids, tr_ids = list(shuffled[:n_v][:n_val]), list(shuffled[n_v:][:n_train])
    print(f"[B] his split: {len(shuffled)-n_v:,} train / {n_v:,} val entities; sampled {len(tr_ids):,} / {len(val_ids):,}")
    targets = pd.concat([read_full_tsv(config.TRAIN_S2_PATH), read_full_tsv(config.TRAIN_S3_PATH)], ignore_index=True)
    print(f"[B] loaded ({time.time()-t0:.0f}s); preprocessing {len(targets):,} S2/S3 records with his preprocess_df ...")
    tp = preprocess_df(targets)
    del targets
    s1p = preprocess_df(s1[s1["entity_id"].isin(set(tr_ids) | set(val_ids))])
    print(f"[B] preprocessed ({time.time()-t0:.0f}s, peak mem {peak_mem_gb():.1f} GB)")

    tb = time.time()
    cand = generate_candidates(s1p, tp)  # per-S1 independent: one call for train+val sample
    dt = time.time() - tb
    print(f"[B] his blocking on {len(s1p):,} S1 entities vs {len(tp):,} records: {dt:.0f}s "
          f"({dt/len(s1p)*1000:.2f} ms/entity -> ~{dt/len(s1p)*1_732_544/3600:.1f} h for the 1.73M test entities), "
          f"peak mem {peak_mem_gb():.1f} GB")
    s1_tr, s1_v = s1p[s1p["entity_id"].isin(tr_ids)], s1p[s1p["entity_id"].isin(val_ids)]
    gt_tr, gt_v = gt[gt["source1_entity_id"].isin(tr_ids)], gt[gt["source1_entity_id"].isin(val_ids)]
    cand_tr, cand_v = cand[cand["source1_entity_id"].isin(tr_ids)], cand[cand["source1_entity_id"].isin(val_ids)]
    rv = compute_candidate_recall_ceiling(cand_v, gt_v)
    print(f"[B] his VALIDATION short-list recall: {rv['candidate_recall_ceiling']:.4f} "
          f"({rv['captured_true_pairs']:,}/{rv['total_true_pairs']:,})")

    f_tr, f_v = feats(cand_tr, s1_tr, tp), feats(cand_v, s1_v, tp)
    X_tr, y_tr = build_training_pairs(f_tr, gt_tr)
    X_v, y_v = build_training_pairs(f_v.copy(), gt_v)
    model = train_model(X_tr, y_tr, X_v, y_v)
    prob_v = predict_match_probabilities(model, f_v)
    thr, _ = optimize_decision_threshold(prob_v, gt_v, list(s1_v["entity_id"]))
    val_match = generate_matching_results(prob_v, list(s1_v["entity_id"]), threshold=thr)
    his = compute_macro_f05(val_match, gt_v)
    print(f"[B] model trained + threshold chosen ({time.time()-t0:.0f}s, peak mem {peak_mem_gb():.1f} GB)")

    gt_d = lists_to_dict(gt_v, "source1_entity_id", "matched_entity_ids")
    pred_d = lists_to_dict(val_match, "source1_entity_id", "matched_entity_ids")
    cand_d = lists_to_dict(cand_v, "source1_entity_id", "candidate_entity_ids")
    r_his = our_score(pred_d, gt_d)
    r_orc = our_score({k: cand_d.get(k, set()) & v for k, v in gt_d.items()}, gt_d)

    base_pred = parse_id_list_file(str(ROOT / "runs/baseline_v1/train_matching_results.tsv"), "matched_entity_ids")
    base_cand = parse_id_list_file(str(ROOT / "runs/baseline_v1/train_candidate_pairs.tsv"), "candidate_entity_ids")
    r_base = our_score({k: base_pred.get(k, set()) for k in gt_d}, gt_d)
    r_base_orc = our_score({k: base_cand.get(k, set()) & v for k, v in gt_d.items()}, gt_d)
    base_rec = sum(len(base_cand.get(k, set()) & v) for k, v in gt_d.items()) / sum(len(v) for v in gt_d.values())

    print("\n" + "=" * 78)
    print(f"RESULTS on the same {len(gt_d):,} validation entities (full 10.3M-record pool)")
    print("=" * 78)
    print(f"{'':32}{'Varun v1':>14}{'our baseline v1':>18}")
    print(f"{'macro F0.5 (our scorer)':32}{r_his['macro_f05']:>14.4f}{r_base['macro_f05']:>18.4f}")
    print(f"{'  (his own evaluator)':32}{his['f05_macro']:>14.4f}{'':>18}")
    print(f"{'best-possible judge (oracle)':32}{r_orc['macro_f05']:>14.4f}{r_base_orc['macro_f05']:>18.4f}")
    print(f"{'short-list recall':32}{rv['candidate_recall_ceiling']:>14.4f}{base_rec:>18.4f}")
    print(f"{'micro precision':32}{r_his['micro_precision']:>14.4f}{r_base['micro_precision']:>18.4f}")
    print(f"{'micro recall':32}{r_his['micro_recall']:>14.4f}{r_base['micro_recall']:>18.4f}")
    print(f"{'singleton F0.5':32}{r_his['singleton_f05']:>14.4f}{r_base['singleton_f05']:>18.4f}")
    print(f"{'non-singleton F0.5':32}{r_his['nonsingleton_f05']:>14.4f}{r_base['nonsingleton_f05']:>18.4f}")
    print(f"{'mean predicted list size':32}{r_his['mean_pred_list_size']:>14.2f}{r_base['mean_pred_list_size']:>18.2f}")
    print(f"his chosen threshold: {thr} | feature importance: "
          + ", ".join(f"{n}={v}" for n, v in sorted(zip(model.feature_name_, model.feature_importances_), key=lambda x: -x[1])))
    print(f"total {time.time()-t0:.0f}s, peak mem {peak_mem_gb():.1f} GB")


ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["A", "B"], required=True)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--n-train", type=int, default=40000)
ap.add_argument("--n-val", type=int, default=20000)
a = ap.parse_args()
mode_a(a.n) if a.mode == "A" else mode_b(a.n_train, a.n_val)
