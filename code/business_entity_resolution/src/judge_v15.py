"""
Judge v15 = judge v13 + WORD-ALIGNMENT features (typo vs replaced word).

Why (27 Sep): the judge sees name similarity as whole-name scores and exact word add/drop/swap counts, so
"Tecnoloogy" vs "Technology" (a typo — how TRUE copies are corrupted) and "Printing" vs "Ventures" (a
replaced word — how DECOYS are made) both look like "one word differs". Aligning words one by one separates
them: for every content word of the S1 name, the best fuzzy match among the record's words (100 identical,
~85 typo, ~20 different word), and the same from the record's side.

New features (content words = name tokens minus legal forms / generic words / fillers, as in judge_v12):
  wa_s1_min, wa_s1_mean   worst / average best-match score of the S1 content words in the record
  wa_s1_nbad              S1 content words with no match >= 60 (replaced or dropped)
  wa_s1_ntypo             S1 content words matched at 70-99 (typo-like)
  wa_r_min, wa_r_nbad     worst best-match / unmatched count of the record's content words against the S1 name

  python judge_v15.py train --tag v15
  python judge_v15.py test --tag v15 --v8-dir ../../../runs/v11fr --blocking-dir ../../../runs/blocking_v11 --out-dir <new folder>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, process

import judge_v12
import judge_v13
import pipeline_v8 as p8
import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region
from judge_v11 import CNT, copy_count_logratio, count_features
from judge_v12 import SWF, swap_features
from judge_v13 import RWF, assemble, raw_features
from translit import latin_tokens

RUN15 = PROJECT / "runs" / "v15"
WAF = ["wa_s1_min", "wa_s1_mean", "wa_s1_nbad", "wa_s1_ntypo", "wa_r_min", "wa_r_nbad"]
judge_v12.STOP_VARIANT = "b"


def align_features(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """pairs (i1, i2) -> (i1, i2, WAF)."""
    stop = judge_v12.STOP | judge_v12.STOP_FR
    s1raw, _, oth, _ = p8.load_raw_v8(split)
    t1 = s1raw.select(latin_tokens("business_name").list.unique()).to_series().to_list()
    t2 = oth.select(latin_tokens("business_name").list.unique()).to_series().to_list()
    del s1raw, oth
    content = lambda ws: [w for w in ws if w not in stop and len(w) > 1]
    i1, i2 = pairs["i1"].to_numpy(), pairs["i2"].to_numpy()
    out = np.full((len(i1), len(WAF)), np.nan, dtype=np.float32)
    for k in range(len(i1)):
        a, b = t1[i1[k]], t2[i2[k]]
        ca, cb = content(a), content(b)
        if ca and b:
            m = process.cdist(ca, b, scorer=fuzz.ratio).max(axis=1)
            out[k, 0], out[k, 1] = m.min(), m.mean()
            out[k, 2], out[k, 3] = (m < 60).sum(), ((m >= 70) & (m < 100)).sum()
        if cb and a:
            m = process.cdist(cb, a, scorer=fuzz.ratio).max(axis=1)
            out[k, 4], out[k, 5] = m.min(), (m < 60).sum()
    return pairs.select("i1", "i2").with_columns(*[pl.Series(c, out[:, j]) for j, c in enumerate(WAF)])


def cmd_train(a):
    t0 = time.time()
    d = RUN15 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    lr = copy_count_logratio()
    p1 = pl.read_parquet(src / "train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005).join(kept, on="i1", how="semi")
    cf = count_features(p1, lr)
    sw = swap_features(p1.select("i1", "i2"), "train")
    rw = raw_features(p1, "train")
    wa = align_features(p1.select("i1", "i2"), "train")
    print(f"count + swap + raw + alignment features ({time.time()-t0:.0f}s)", flush=True)
    df = load_region(RUN8 / "train_features", p1.select("i1", "i2"), kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
    df = assemble(df, cf, sw, rw).join(wa, on=["i1", "i2"], how="left") \
        .join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
        .with_columns(pl.col("y").fill_null(0), fold_of().alias("fold"))
    del cf, sw, rw, wa
    F = FEATS + CNT + SWF + RWF + WAF
    oof = []
    for fold in (0, 1):
        tr = df.filter(pl.col("fold") != fold)
        va_e = tr.select("i1").unique().sample(fraction=0.05, seed=99)
        va, tr = tr.join(va_e, on="i1", how="semi"), tr.join(va_e, on="i1", how="anti")
        dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["y"].to_numpy(), max_bin=XPARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(va.select(F).to_numpy(), va["y"].to_numpy(), ref=dtr)
        m = xgb.train(XPARAMS, dtr, num_boost_round=6000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=1000)
        m.save_model(str(d / f"stage2_fold{fold}.ubj"))
        held = df.filter(pl.col("fold") == fold)
        oof.append(held.select("i1", "i2", "rank").with_columns(
            pl.Series("p", m.inplace_predict(held.select(F).to_numpy(), iteration_range=(0, m.best_iteration + 1)))))
        print(f"[fold {fold}] {m.best_iteration + 1} trees ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, tr, va
    oof = pl.concat(oof)
    oof.write_parquet(d / "train_p2.parquet")
    ctry = pl.read_parquet(PROJECT / "data_cache" / "train_source1.parquet").with_row_index("i1").select("i1", "country")

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(pairs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    cut = region(src / "train_p1.parquet", kept, 0.01)
    mine = owned(oof.join(cut, on=["i1", "i2"], how="semi"), 1000)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.6, 0.7)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.1, 1.25)])
    res = {json.dumps(r): score(decide(mine, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  v15 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "lr": lr.tolist(), "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v15 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} (v13: 0.98468) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d, bl = RUN15 / a.tag, a.v8_dir.resolve(), a.blocking_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    dec = json.loads((d / "decision.json").read_text())
    rule = json.loads(a.rule) if a.rule else dec["best_stage2"]["rule"]
    n = pl.read_parquet(bl / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    p1 = pl.read_parquet(v8d / "gpu_slot" / "test_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005)
    cf = count_features(p1, np.array(dec["lr"]))
    sw = swap_features(p1.select("i1", "i2"), "test")
    rw = raw_features(p1, "test")
    cand = p1.filter(pl.col("p") >= a.cand_p1).select("i1", "i2")
    wa = align_features(cand, "test")
    x = assemble(load_region(v8d / "test_features", p1.select("i1", "i2"), kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet")), cf, sw, rw)
    x = x.join(cand, on=["i1", "i2"], how="semi").join(wa, on=["i1", "i2"], how="left")
    X = x.select(FEATS + CNT + SWF + RWF + WAF).to_numpy()
    ms = [xgb.Booster(model_file=str(d / f"stage2_fold{f}.ubj")) for f in (0, 1)]
    p2 = x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean([m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1)) for m in ms], axis=0)))
    p2.write_parquet(d / f"test_p2_{v8d.name}.parquet")
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(bl / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(bl / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    meta = {"tag": a.tag, "features_from": str(v8d), "blocking": str(bl), "cand_p1": a.cand_p1, "rule": rule,
            "candidate_pairs": p2.height, "candidates_per_s1": p2.height / n, "matches": chosen.height}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), f"({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="v15")
    ap.add_argument("--v8-dir", type=Path)
    ap.add_argument("--blocking-dir", type=Path, default=BL8)
    ap.add_argument("--cand-p1", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "test" and not (a.out_dir and a.v8_dir):
        ap.error("test needs --out-dir and --v8-dir")
    {"train": cmd_train, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
