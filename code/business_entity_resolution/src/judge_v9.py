"""
Judge v9 = a stronger STAGE 2, trained only where it decides.

Since the 27 Sep rule update the submitted candidate set is the pairs with stage-1 probability
>= 0.01 (export_submission.py); the stage-2 judge only chooses among them. So stage 2 is retrained
on the pairs with stage-1 OOF probability >= P1_MIN (0.005, a margin below the 0.01 cut) instead of
the whole ~97-per-S1 short-list. That region is ~20x smaller, so ALL kept train entities fit
(v5/v8: 200K entities per fold), with a slower learning rate and more trees.

Everything upstream is v8's, unchanged: short-lists v8, v8 features (+NEW7), the v8 stage-1 models
and OOF, the v8 stage-2 evidence features (runs/v8/<tag>/train_s2, test_s2). Same test-like universe,
same 2 folds (fold_of), so the held-out score is directly comparable with v8 restricted the same way.

  python judge_v9.py train --tag v9_full                       # -> runs/v9/<tag>/
  python judge_v9.py test  --tag v9_full --v8-dir ../../../runs/v8fr --out-dir <new folder> [--cand-p1 0.01]
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

from baseline import to_ids, write_id_lists
from features_v7 import NEW7
from features_v4 import FEATURES_V4
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import S2F, kept_entities, load_part
import pipeline_v6 as v6

PROJECT = Path(__file__).resolve().parents[3]
BL8, RUN8, RUN9 = PROJECT / "runs" / "blocking_v8", PROJECT / "runs" / "v8", PROJECT / "runs" / "v9"
FEATS = FEATURES_V4 + NEW7 + S2F
P1_MIN = 0.005
XPARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
    "learning_rate": 0.03, "max_depth": 0, "max_leaves": 255, "grow_policy": "lossguide", "min_child_weight": 2,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "max_bin": 256, "seed": 42,
}


def region(p1_path: Path, kept: pl.DataFrame, p1_min: float) -> pl.DataFrame:
    return (pl.scan_parquet(p1_path).filter(pl.col("p") >= p1_min).select("i1", "i2").collect()
            .join(kept, on="i1", how="semi"))


def load_region(feat_dir: Path, reg: pl.DataFrame, kept: pl.DataFrame, ctx, s2) -> pl.DataFrame:
    rows = []
    for p in sorted(feat_dir.glob("part-*.parquet")):
        x = load_part(p, kept, ctx, s2)
        if x.height:
            rows.append(x.join(reg, on=["i1", "i2"], how="semi").select(["i1", "i2"] + FEATS))
    return pl.concat(rows)


def cmd_train(a):
    t0 = time.time()
    params = {**XPARAMS, **(json.loads(a.params) if a.params else {})}
    d = RUN9 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    reg = region(src / "train_p1.parquet", kept, a.p1_min)
    df = load_region(RUN8 / "train_features", reg, kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
    df = df.join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left").with_columns(
        pl.col("y").fill_null(0), fold_of().alias("fold"))
    print(f"region p1 >= {a.p1_min}: {df.height:,} pairs, {int(df['y'].sum()):,} true, {df['i1'].n_unique():,} entities "
          f"(true pairs in region: {df['y'].sum() / truth.height:.2%} of all) ({time.time()-t0:.0f}s)", flush=True)
    oof = []
    for fold in (0, 1):
        tr = df.filter(pl.col("fold") != fold)
        ents = tr.select("i1").unique()
        va_e = ents.sample(fraction=0.05, seed=99)
        va, tr = tr.join(va_e, on="i1", how="semi"), tr.join(va_e, on="i1", how="anti")
        dtr = xgb.QuantileDMatrix(tr.select(FEATS).to_numpy(), tr["y"].to_numpy(), max_bin=params["max_bin"])
        dva = xgb.QuantileDMatrix(va.select(FEATS).to_numpy(), va["y"].to_numpy(), ref=dtr)
        m = xgb.train(params, dtr, num_boost_round=a.rounds, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=500)
        m.save_model(str(d / f"stage2_fold{fold}.ubj"))
        held = df.filter(pl.col("fold") == fold)
        p = m.inplace_predict(held.select(FEATS).to_numpy(), iteration_range=(0, m.best_iteration + 1))
        oof.append(held.select("i1", "i2", "rank").with_columns(pl.Series("p", p)))
        print(f"[fold {fold}] train {tr.height:,} pairs | {m.best_iteration + 1} trees ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, tr, va
    oof = pl.concat(oof)
    oof.write_parquet(d / "train_p2.parquet")
    rm = kept.with_row_index("j")
    ctry = pl.read_parquet(PROJECT / "data_cache" / "train_source1.parquet").with_row_index("i1").select("i1", "country")

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(pairs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    # baseline: v8 stage 2 on the same submitted region (p1 >= 0.01)
    cut = region(src / "train_p1.parquet", kept, 0.01)
    v8p2 = pl.read_parquet(src / "train_p2.parquet").join(cut, on=["i1", "i2"], how="semi")
    v8rule = json.loads((src / "decision.json").read_text())["best_stage2"]["rule"]
    base = score(decide(owned(v8p2, 1000), v8rule, already_owned=True))
    print(f"\nv8 stage 2 on candidates p1>=0.01: F0.5 {base['f05']:.5f} | P {base['precision']:.4f} R {base['recall']:.4f}")
    mine = owned(oof.join(cut, on=["i1", "i2"], how="semi"), 1000)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.4, 0.5, 0.6, 0.7)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.8, 0.9, 1.0, 1.1, 1.25)])
    res = {json.dumps(r): score(decide(mine, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  v9 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "p1_min": a.p1_min, "params": params,
                                                 "v8_same_region": base, "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v9 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} "
          f"(v8 same region {base['f05']:.5f}) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d = RUN9 / a.tag, a.v8_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((d / "decision.json").read_text())["best_stage2"]["rule"]
    n = pl.read_parquet(BL8 / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    cand = region(v8d / "gpu_slot" / "test_p1.parquet", kept, a.cand_p1)
    x = load_region(v8d / "test_features", cand, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet"))
    X = x.select(FEATS).to_numpy()
    ms = [xgb.Booster(model_file=str(d / f"stage2_fold{f}.ubj")) for f in (0, 1)]
    p = np.mean([m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1)) for m in ms], axis=0)
    p2 = x.select("i1", "i2", "rank").with_columns(pl.Series("p", p))
    p2.write_parquet(d / f"test_p2_{v8d.name}.parquet")
    chosen = decide_by_country(p2, rule, "test")
    assert chosen.join(p2, on=["i1", "i2"], how="anti").height == 0
    s1_ids = pl.read_parquet(BL8 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL8 / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    meta = {"tag": a.tag, "features_from": str(v8d), "cand_p1": a.cand_p1, "rule": rule,
            "candidate_pairs": p2.height, "candidates_per_s1": p2.height / n, "matches": chosen.height}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), f"({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="v9_full")
    ap.add_argument("--p1-min", type=float, default=P1_MIN)
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--params", help="JSON overrides of XPARAMS (ensemble members)")
    ap.add_argument("--v8-dir", type=Path, help="test: runs/v8 or runs/v8fr (test features + stage-1 outputs)")
    ap.add_argument("--cand-p1", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "test" and not (a.out_dir and a.v8_dir):
        ap.error("test needs --out-dir (a new folder) and --v8-dir")
    {"train": cmd_train, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
