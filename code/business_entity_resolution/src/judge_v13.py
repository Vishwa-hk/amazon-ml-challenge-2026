"""
Judge v13 = judge v12b + RAW-NAME RIVALRY and SOURCE-BALANCE features.

Why (27 Sep, reverse-engineering the no-address records): for a no-address record whose true S1 (A)
has a same-named twin (B) — same name after normalisation — the record's RAW name (legal form and
punctuation kept) is closer to A than to B in 83.8% of the cases where they differ (19,013 vs 3,689;
17,298 ties = identical raw names, truly undecidable). The generator keeps the raw form of the
source name; our normalisation strips exactly those details, and the judge never compared a pair's raw
similarity with the record's OTHER candidates. Also, a record tends to come from the source (S2/S3)
where its S1 has FEWER other copies (42.3% vs 57.7%): the generator balances sources per entity.

New features (candidate region = stage-1 p >= 0.005, same on train and test):
  rw_ratio    fuzz.ratio of the accent-folded lower-case raw names, punctuation kept
  rw_margin   rw_ratio minus the best rw_ratio of the record's OTHER candidate S1 entities
  rw_rank     rank of rw_ratio among the record's candidates (1 = closest)
  raw_margin  the same margin on the existing raw_ratio feature
  rw_exact    raw names equal
  src_same / src_other   this S1's other confident (stage-1 p > 0.5) records from the record's source / the other source

  python judge_v13.py train --tag v13
  python judge_v13.py test --tag v13 --v8-dir ../../../runs/v11fr --blocking-dir ../../../runs/blocking_v11 --out-dir <new folder>
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
import pipeline_v8 as p8
import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region
from judge_v11 import CNT, copy_count_logratio, count_features
from judge_v12 import SWF, swap_features
from normalize import clean_text

RUN13 = PROJECT / "runs" / "v13"
RWF = ["rw_ratio", "rw_margin", "rw_rank", "raw_margin", "rw_exact", "src_same", "src_other"]
judge_v12.STOP_VARIANT = "b"


def margin(df: pl.DataFrame, col: str, name: str) -> pl.Expr:
    """col minus the best col among the record's OTHER candidates (NaN when it has no other)."""
    best = pl.col(col).max().over("i2")
    second = pl.col(col).sort(descending=True).slice(1, 1).first().over("i2")
    return (pl.when(pl.len().over("i2") == 1).then(None)
            .when(pl.col(col) >= best).then(pl.col(col) - second)
            .otherwise(pl.col(col) - best)).alias(name)


def raw_features(p1: pl.DataFrame, split: str) -> pl.DataFrame:
    """p1: (i1, i2, p) region pairs -> (i1, i2, RWF without raw_margin)."""
    s1raw, n_s2, oth, _ = p8.load_raw_v8(split)
    n1 = s1raw.select(clean_text(pl.col("business_name")).str.strip_chars()).to_series().to_list()
    n2 = oth.select(clean_text(pl.col("business_name")).str.strip_chars()).to_series().to_list()
    del s1raw, oth
    i1, i2 = p1["i1"].to_numpy(), p1["i2"].to_numpy()
    a, b = [n1[i] for i in i1], [n2[i] for i in i2]
    r = process.cpdist(a, b, scorer=fuzz.ratio, workers=-1)
    exact = np.fromiter((x == y for x, y in zip(a, b)), dtype=bool, count=len(a))
    del a, b, n1, n2
    x = p1.select("i1", "i2", "p").with_columns(pl.Series("rw_ratio", r.astype(np.float32)), pl.Series("rw_exact", exact),
                                                (pl.col("i2") >= n_s2).alias("s3"))
    x = x.with_columns(margin(x, "rw_ratio", "rw_margin"),
                       pl.col("rw_ratio").rank("min", descending=True).over("i2").alias("rw_rank"))
    conf = x.filter(pl.col("p") > 0.5).group_by("i1").agg(pl.col("s3").sum().alias("n3"), (~pl.col("s3")).sum().alias("n2c"))
    x = x.join(conf, on="i1", how="left").with_columns(pl.col("n3").fill_null(0), pl.col("n2c").fill_null(0))
    self_conf = (pl.col("p") > 0.5).cast(pl.Int64)
    x = x.with_columns(
        pl.when(pl.col("s3")).then(pl.col("n3") - self_conf).otherwise(pl.col("n2c") - self_conf).alias("src_same"),
        pl.when(pl.col("s3")).then(pl.col("n2c")).otherwise(pl.col("n3")).alias("src_other"))
    return x.select("i1", "i2", *[pl.col(c).cast(pl.Float32) for c in ["rw_ratio", "rw_margin", "rw_rank", "rw_exact", "src_same", "src_other"]])


def assemble(df: pl.DataFrame, cf, sw, rw) -> pl.DataFrame:
    df = df.join(cf, on=["i1", "i2"], how="left").join(sw, on=["i1", "i2"], how="left").join(rw, on=["i1", "i2"], how="left")
    return df.with_columns(margin(df, "raw_ratio", "raw_margin").cast(pl.Float32))


def cmd_train(a):
    t0 = time.time()
    d = RUN13 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / a.slot_tag            # v14: stage-1/2 trained on the FULL universe (--slot-tag gpu_slot_full --drop-frac 0)
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, a.drop_frac)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    lr = copy_count_logratio()
    p1 = pl.read_parquet(src / "train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005).join(kept, on="i1", how="semi")
    cf = count_features(p1, lr)
    sw = swap_features(p1.select("i1", "i2"), "train")
    rw = raw_features(p1, "train")
    print(f"count + swap + raw features ({time.time()-t0:.0f}s)", flush=True)
    ctx = str(src / "train_ctx.parquet") if a.drop_frac > 0 else None   # full universe: the parts' own context is exact
    df = load_region(RUN8 / "train_features", p1.select("i1", "i2"), kept, ctx, str(src / "train_s2" / "part-*.parquet"))
    df = assemble(df, cf, sw, rw).join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
        .with_columns(pl.col("y").fill_null(0), fold_of().alias("fold"))
    del cf, sw, rw
    F = FEATS + CNT + SWF + RWF
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
        print(f"  v13 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "lr": lr.tolist(), "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v13 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} (v12b: 0.98434) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d, bl = RUN13 / a.tag, a.v8_dir.resolve(), a.blocking_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    dec = json.loads((d / "decision.json").read_text())
    rule = json.loads(a.rule) if a.rule else dec["best_stage2"]["rule"]
    n = pl.read_parquet(bl / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    p1 = pl.read_parquet(v8d / a.slot_tag / "test_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005)
    cf = count_features(p1, np.array(dec["lr"]))
    sw = swap_features(p1.select("i1", "i2"), "test")
    rw = raw_features(p1, "test")
    x = assemble(load_region(v8d / "test_features", p1.select("i1", "i2"), kept, None, str(v8d / a.slot_tag / "test_s2" / "part-*.parquet")), cf, sw, rw)
    x = x.join(p1.filter(pl.col("p") >= a.cand_p1).select("i1", "i2"), on=["i1", "i2"], how="semi")   # submitted candidates
    X = x.select(FEATS + CNT + SWF + RWF).to_numpy()
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
    ap.add_argument("--tag", default="v13")
    ap.add_argument("--slot-tag", default="gpu_slot", help="stage-1/2 run folder under runs/v8 (train) or --v8-dir (test)")
    ap.add_argument("--drop-frac", type=float, default=0.19, help="test-like universe; 0 = full universe (v14)")
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
