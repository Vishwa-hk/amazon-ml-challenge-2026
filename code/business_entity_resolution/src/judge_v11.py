"""
Judge v11 = judge v9 + RIVAL-COUNT features (the "copy-count prior").

Why (v9 error analysis, 27 Sep): half of the remaining judge loss (~0.0056 F0.5) is records with NO
address whose name fits several S1 entities with the same name. The record itself cannot decide,
but the generator gives each S1 entity 0-11 copies with a known distribution (mostly 2-5). If one
same-named S1 already has 5 confident copies and the other has 1, the extra copy is several times
more likely to be the second one's: pmf(n+1)/pmf(n) is 0.51 at n=5 but 3.15 at n=1. The v9 judge sees
its own S1's count, not the rival's.

New features (from stage-1 out-of-fold probabilities p1, same definition on train and test):
  c_own_n5 / c_own_sum    confident (p1 > 0.5) copies / probability mass of this S1, this record excluded
  c_riv_p                 p1 of the record's best OTHER S1
  c_riv_n5 / c_riv_sum    the same counts for that rival S1 (this record excluded)
  c_n_riv                 how many other S1 entities hold the record with p1 > 0.05
  c_lr_own / c_lr_riv     log pmf(n+1) - log pmf(n) at n = c_own_n5 / c_riv_n5 (train copy-count distribution)
  c_lr_diff               c_lr_own - c_lr_riv
Everything else (region, universe, folds, XGBoost settings) is judge v9's.

  python judge_v11.py train --tag v11
  python judge_v11.py test --tag v11 --v8-dir ../../../runs/v11fr --out-dir <new folder>
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

import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region

RUN11 = PROJECT / "runs" / "v11"
CNT = ["c_own_n5", "c_own_sum", "c_riv_p", "c_riv_n5", "c_riv_sum", "c_n_riv", "c_lr_own", "c_lr_riv", "c_lr_diff"]


def copy_count_logratio() -> np.ndarray:
    """lr[n] = log pmf(n+1) - log pmf(n), n = 0..11, from the train ground truth (all S1)."""
    v6.BL6 = BL8
    n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    k = (pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
         .join(v6.truth("train").group_by("i1").len("k"), on="i1", how="left").fill_null(0)["k"].to_numpy())
    pmf = (np.bincount(k, minlength=13)[:13] + 1.0) / (len(k) + 13.0)
    return np.log(pmf[1:]) - np.log(pmf[:-1])


def count_features(p1: pl.DataFrame, lr: np.ndarray) -> pl.DataFrame:
    """p1: (i1, i2, p) stage-1 probabilities (p >= 0.005 region) -> (i1, i2, CNT)."""
    p = p1.select("i1", "i2", pl.col("p").cast(pl.Float32))
    own = p.group_by("i1").agg((pl.col("p") > 0.5).sum().alias("n5"), pl.col("p").sum().alias("s"))
    top = p.group_by("i2").agg(pl.col("i1").sort_by("p", descending=True).head(2).alias("a"),
                               pl.col("p").sort(descending=True).head(2).alias("t"),
                               (pl.col("p") > 0.05).sum().alias("m05"))
    x = (p.join(own, on="i1").join(top, on="i2")
         .with_columns(pl.when(pl.col("a").list.get(0) == pl.col("i1")).then(pl.col("a").list.get(1, null_on_oob=True))
                       .otherwise(pl.col("a").list.get(0)).alias("riv"),
                       pl.when(pl.col("a").list.get(0) == pl.col("i1")).then(pl.col("t").list.get(1, null_on_oob=True))
                       .otherwise(pl.col("t").list.get(0)).fill_null(0.0).alias("c_riv_p"))
         .with_columns((pl.col("n5") - (pl.col("p") > 0.5).cast(pl.UInt32)).alias("c_own_n5"),
                       (pl.col("s") - pl.col("p")).alias("c_own_sum"),
                       (pl.col("m05") - (pl.col("p") > 0.05).cast(pl.UInt32)).alias("c_n_riv"))
         .join(own.rename({"i1": "riv", "n5": "rn5", "s": "rs"}), on="riv", how="left")
         .with_columns((pl.col("rn5") - (pl.col("c_riv_p") > 0.5).cast(pl.UInt32)).alias("c_riv_n5"),
                       (pl.col("rs") - pl.col("c_riv_p")).alias("c_riv_sum")))
    table = {i: float(v) for i, v in enumerate(lr)}
    look = lambda c: pl.col(c).clip(0, len(lr) - 1).cast(pl.Int64).replace_strict(table, return_dtype=pl.Float32)
    x = x.with_columns(look("c_own_n5").alias("c_lr_own"),
                       pl.when(pl.col("c_riv_n5").is_null()).then(None).otherwise(look("c_riv_n5")).alias("c_lr_riv"))
    x = x.with_columns((pl.col("c_lr_own") - pl.col("c_lr_riv")).alias("c_lr_diff"))
    return x.select("i1", "i2", *[pl.col(c).cast(pl.Float32) for c in CNT])


def cmd_train(a):
    t0 = time.time()
    d = RUN11 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    lr = copy_count_logratio()
    print("copy-count log-ratios lr[n] = log pmf(n+1)/pmf(n):", np.round(lr, 2).tolist(), flush=True)
    p1 = pl.read_parquet(src / "train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005).join(kept, on="i1", how="semi")
    cf = count_features(p1, lr)
    reg = p1.select("i1", "i2")
    df = load_region(RUN8 / "train_features", reg, kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
    df = df.join(cf, on=["i1", "i2"], how="left").join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
        .with_columns(pl.col("y").fill_null(0), fold_of().alias("fold"))
    F = FEATS + CNT
    print(f"region {df.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
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
        print(f"  v11 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "lr": lr.tolist(), "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v11 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} (v9: 0.98418) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d = RUN11 / a.tag, a.v8_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    dec = json.loads((d / "decision.json").read_text())
    rule = json.loads(a.rule) if a.rule else dec["best_stage2"]["rule"]
    lr = np.array(dec["lr"])
    bl = a.blocking_dir.resolve()
    n = pl.read_parquet(bl / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    p1 = pl.read_parquet(v8d / "gpu_slot" / "test_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005)
    cf = count_features(p1, lr)
    cand = p1.filter(pl.col("p") >= a.cand_p1).select("i1", "i2")
    x = load_region(v8d / "test_features", cand, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet")).join(cf, on=["i1", "i2"], how="left")
    X = x.select(FEATS + CNT).to_numpy()
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
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--v8-dir", type=Path, help="test: folder with test_features/ and gpu_slot/ (runs/v8fr or runs/v11fr)")
    ap.add_argument("--blocking-dir", type=Path, default=BL8, help="test: folder with the test id maps")
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
