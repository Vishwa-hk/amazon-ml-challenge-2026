"""
Judge v10 = STAGE 3: the rival / cluster evidence recomputed from v9's (better) stage-2
probabilities, then another judge trained on top — a second round of collective reasoning.

Same region (stage-1 OOF p >= 0.005), same universe and folds as judge_v9. Evidence builder:
pipeline_v6.stage2_features on the v8 inputs (importing pipeline_v8 patches the raw loading),
fed with v9 probabilities instead of stage-1 ones; its columns get the suffix "_3".

  python judge_v10.py train --tag v10
  python judge_v10.py test --tag v10 --v8-dir ../../../runs/v8fr --v9-pred ../../../runs/v9/v9_full/test_p2_v8fr.parquet --out-dir <new>
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

import pipeline_v8  # noqa: F401  (patches pipeline_v6.load_raw to the v8 inputs)
import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import S2F, kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region

RUN10 = PROJECT / "runs" / "v10"
S3F = [f"{c}_3" for c in S2F]


def evidence(p: pl.DataFrame, split: str, out_dir: Path) -> str:
    v6.stage2_features(p.select("i1", "i2", "p"), split, out_dir)
    return str(out_dir / "part-*.parquet")


def with_s3(df: pl.DataFrame, s3: str) -> pl.DataFrame:
    e = pl.scan_parquet(s3).rename(dict(zip(S2F, S3F))).join(df.select("i1", "i2").lazy(), on=["i1", "i2"], how="semi").collect()
    return df.join(e, on=["i1", "i2"], how="left")


def cmd_train(a):
    t0 = time.time()
    d = RUN10 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    v9 = pl.read_parquet(PROJECT / "runs" / "v9" / a.v9_tag / "train_p2.parquet")
    s3 = evidence(v9, "train", d / "train_s3")
    print(f"stage-3 evidence built ({time.time()-t0:.0f}s)", flush=True)
    reg = region(src / "train_p1.parquet", kept, 0.005)
    df = load_region(RUN8 / "train_features", reg, kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
    df = with_s3(df, s3).join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
        .with_columns(pl.col("y").fill_null(0), fold_of().alias("fold"))
    F = FEATS + S3F
    oof = []
    for fold in (0, 1):
        tr = df.filter(pl.col("fold") != fold)
        va_e = tr.select("i1").unique().sample(fraction=0.05, seed=99)
        va, tr = tr.join(va_e, on="i1", how="semi"), tr.join(va_e, on="i1", how="anti")
        dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["y"].to_numpy(), max_bin=XPARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(va.select(F).to_numpy(), va["y"].to_numpy(), ref=dtr)
        m = xgb.train(XPARAMS, dtr, num_boost_round=6000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=1000)
        m.save_model(str(d / f"stage3_fold{fold}.ubj"))
        held = df.filter(pl.col("fold") == fold)
        oof.append(held.select("i1", "i2", "rank").with_columns(
            pl.Series("p", m.inplace_predict(held.select(F).to_numpy(), iteration_range=(0, m.best_iteration + 1)))))
        print(f"[fold {fold}] {m.best_iteration + 1} trees ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, tr, va
    oof = pl.concat(oof)
    oof.write_parquet(d / "train_p3.parquet")
    ctry = pl.read_parquet(PROJECT / "data_cache" / "train_source1.parquet").with_row_index("i1").select("i1", "country")

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(pairs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    cut = region(src / "train_p1.parquet", kept, 0.01)
    base9 = score(decide(owned(v9.join(cut, on=["i1", "i2"], how="semi"), 1000),
                         json.loads((PROJECT / "runs" / "v9" / a.v9_tag / "decision.json").read_text())["best_stage2"]["rule"], already_owned=True))
    mine = owned(oof.join(cut, on=["i1", "i2"], how="semi"), 1000)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.6, 0.7)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.1, 1.25)])
    res = {json.dumps(r): score(decide(mine, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  v10 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "v9_tag": a.v9_tag, "v9_same_region": base9,
                                                 "best_stage3": {"rule": json.loads(best), **res[best]}, "by_country": by}, indent=2))
    print(f"\nBEST v10 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} "
          f"(v9 same region {base9['f05']:.5f}) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d = RUN10 / a.tag, a.v8_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((d / "decision.json").read_text())["best_stage3"]["rule"]
    n = pl.read_parquet(BL8 / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    v9 = pl.read_parquet(a.v9_pred)
    s3dir = d / f"test_s3_{v8d.name}"
    v6.refuse(s3dir)
    s3 = evidence(v9, "test", s3dir)
    cand = region(v8d / "gpu_slot" / "test_p1.parquet", kept, a.cand_p1)
    assert cand.join(v9, on=["i1", "i2"], how="anti").height == 0, "v9 predictions must cover the candidate set"
    x = with_s3(load_region(v8d / "test_features", cand, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet")), s3)
    X = x.select(FEATS + S3F).to_numpy()
    ms = [xgb.Booster(model_file=str(d / f"stage3_fold{f}.ubj")) for f in (0, 1)]
    p3 = x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean(
        [m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1)) for m in ms], axis=0)))
    p3.write_parquet(d / f"test_p3_{v8d.name}.parquet")
    chosen = decide_by_country(p3, rule, "test")
    s1_ids = pl.read_parquet(BL8 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL8 / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p3.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    meta = {"tag": a.tag, "features_from": str(v8d), "cand_p1": a.cand_p1, "rule": rule,
            "candidate_pairs": p3.height, "candidates_per_s1": p3.height / n, "matches": chosen.height}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), f"({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--v9-tag", default="v9_full")
    ap.add_argument("--v8-dir", type=Path)
    ap.add_argument("--v9-pred", type=Path)
    ap.add_argument("--cand-p1", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "test" and not (a.out_dir and a.v8_dir and a.v9_pred):
        ap.error("test needs --out-dir, --v8-dir and --v9-pred")
    {"train": cmd_train, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
