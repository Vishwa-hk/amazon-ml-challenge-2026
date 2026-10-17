"""
v5 judge = the v4 two-stage judge with the MODEL swapped to XGBoost on the GPU.

Everything else (short-lists v3, v4 features, test-like universe, stage-2 evidence,
decision rules, one owner per record) is imported unchanged from judge_v4.py —
judge_v4.py itself is not modified. Outputs go to runs/v5/<tag>/ (never overwritten).

Why: on CPU, LightGBM prediction over ~160-210M pairs took ~33 min of each test
run and most of each training run. XGBoost (CUDA build, verified on the
RTX 5060 Ti) trains and predicts on the GPU.

Usage:
  python judge_v5.py train [--tag gpu_test_like] [--drop-frac 0.19] [--max-parts N]
  python judge_v5.py test  --tag gpu_test_like --out-dir <new folder> [--rule '<json>']
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

import judge_v4
from features_v4 import FEATURES_V4
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import BL3, S2F, context_table, kept_entities, load_part, parts, stage2_features, truth_pairs

PROJECT = Path(__file__).resolve().parents[3]
RUN5 = PROJECT / "runs" / "v5"
XPARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
    "learning_rate": 0.08, "max_depth": 0, "max_leaves": 255, "grow_policy": "lossguide",
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "max_bin": 256, "seed": 42,
}


def fit(X, y, Xv, yv, rounds) -> xgb.Booster:
    dtr = xgb.QuantileDMatrix(X, y, max_bin=XPARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(Xv, yv, ref=dtr)
    return xgb.train(XPARAMS, dtr, num_boost_round=rounds, evals=[(dva, "valid")],
                     early_stopping_rounds=50, verbose_eval=200)


def predict(m: xgb.Booster, X: np.ndarray) -> np.ndarray:
    return m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1))


def cv_stage(name, split_parts, kept, ctx, truth, feats, extra, n_train, rounds, out_dir, t0):
    """2-fold OOF by S1 entity (as judge_v4.cv_stage, XGBoost on GPU)."""
    ent = kept.with_columns(fold_of().alias("fold"))
    oof = []
    for fold in (0, 1):
        pool = ent.filter(pl.col("fold") != fold)
        tr_e = pool.sample(min(n_train, pool.height), seed=fold).select("i1")
        st_e = tr_e.sample(fraction=0.1, seed=99)
        rows = [x for p in split_parts if (x := load_part(p, tr_e, ctx, extra)).height]
        df = pl.concat(rows).join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
            .with_columns(pl.col("y").fill_null(0))
        del rows
        va = df.join(st_e, on="i1", how="semi")
        Xv, yv = va.select(feats).to_numpy(), va["y"].to_numpy()
        del va
        tr = df.join(st_e, on="i1", how="anti")
        del df
        n_tr, n_pos = tr.height, int(tr["y"].sum())
        X, y = tr.select(feats).to_numpy(), tr["y"].to_numpy()
        del tr
        t_fit = time.time()
        print(f"[{name} fold {fold}] train {n_tr:,} pairs ({n_pos:,} true) ({t_fit-t0:.0f}s)", flush=True)
        m = fit(X, y, Xv, yv, rounds)
        del X, y, Xv, yv
        m.save_model(str(out_dir / f"{name}_fold{fold}.ubj"))
        t_pred = time.time()
        held = ent.filter(pl.col("fold") == fold).select("i1")
        for p in split_parts:
            x = load_part(p, held, ctx, extra)
            if x.height:
                oof.append(x.select("i1", "i2", "rank").with_columns(pl.Series("p", predict(m, x.select(feats).to_numpy()))))
        print(f"[{name} fold {fold}] {m.best_iteration + 1} trees | fit {t_pred-t_fit:.0f}s | "
              f"held-out predict {time.time()-t_pred:.0f}s ({time.time()-t0:.0f}s)", flush=True)
    return pl.concat(oof)


def train(a) -> None:
    t0 = time.time()
    d = RUN5 / a.tag
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"refusing to overwrite existing {d}")
    d.mkdir(parents=True, exist_ok=True)
    n_s1 = pl.read_parquet(BL3 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, a.drop_frac)
    truth = truth_pairs().join(kept, on="i1", how="semi")
    print(f"universe '{a.tag}': {kept.height:,} of {n_s1:,} S1 kept (drop {a.drop_frac:.0%})", flush=True)
    ctx = context_table("train", kept, d / "train_ctx.parquet") if a.drop_frac > 0 else None
    P = parts("train")
    oof1 = cv_stage("stage1", P, kept, ctx, truth, FEATURES_V4, None, a.n_train_entities, a.rounds, d, t0)
    oof1.write_parquet(d / "train_p1.parquet")
    s2 = stage2_features(oof1, "train", d / "train_s2")
    del oof1
    print(f"stage-2 features built ({time.time()-t0:.0f}s)", flush=True)
    oof2 = cv_stage("stage2", P, kept, ctx, truth, FEATURES_V4 + S2F, s2, a.n_train_entities, a.rounds, d, t0)
    oof2.write_parquet(d / "train_p2.parquet")

    remap = kept.with_row_index("j")
    def score(pairs):
        return macro_f05(pairs.join(remap, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(remap, on="i1").select(pl.col("j").alias("i1"), "i2"), kept.height)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.5, 0.6, 0.7, 0.8)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.25, 1.5)])
    res = {}
    for stage, oof in (("stage1", pl.read_parquet(d / "train_p1.parquet")), ("stage2", oof2)):
        base = owned(oof, None)
        for r in rules:
            f = score(decide(base, r, already_owned=True))
            res[(stage, json.dumps(r))] = f
            print(f"  {stage} {json.dumps(r):<70} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max((k for k in res if k[0] == "stage2"), key=lambda k: res[k]["f05"])
    best1 = max((k for k in res if k[0] == "stage1"), key=lambda k: res[k]["f05"])
    (d / "decision.json").write_text(json.dumps({
        "tag": a.tag, "drop_frac": a.drop_frac, "model": "xgboost-cuda",
        "best_stage2": {"rule": json.loads(best[1]), **res[best]},
        "best_stage1": {"rule": json.loads(best1[1]), **res[best1]}}, indent=2))
    print(f"\nBEST stage1 {res[best1]['f05']:.5f} {best1[1]}\nBEST stage2 {res[best]['f05']:.5f} {best[1]}\n"
          f"done ({time.time()-t0:.0f}s)")


def test(a) -> None:
    t0 = time.time()
    d = RUN5 / a.tag
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((d / "decision.json").read_text())["best_stage2"]["rule"]
    P = parts("test")
    kept = pl.DataFrame({"i1": pl.arange(0, pl.read_parquet(BL3 / "test_s1_ids.parquet").height, eager=True, dtype=pl.UInt32)})
    pred_path = d / "test_p2.parquet"
    if pred_path.exists():
        p2 = pl.read_parquet(pred_path)
    else:
        m1 = [xgb.Booster(model_file=str(d / f"stage1_fold{f}.ubj")) for f in (0, 1)]
        for m in m1:
            m.set_param({"device": "cuda"})
        p1 = []
        for p in P:
            x = pl.read_parquet(p)
            X = x.select(FEATURES_V4).to_numpy()
            p1.append(x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean([predict(m, X) for m in m1], axis=0))))
        p1 = pl.concat(p1)
        p1.write_parquet(d / "test_p1.parquet")
        print(f"[test] stage 1 scored {p1.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
        s2 = stage2_features(p1, "test", d / "test_s2")
        del p1
        m2 = [xgb.Booster(model_file=str(d / f"stage2_fold{f}.ubj")) for f in (0, 1)]
        for m in m2:
            m.set_param({"device": "cuda"})
        p2 = []
        for p in P:
            x = load_part(p, kept, None, s2)
            X = x.select(FEATURES_V4 + S2F).to_numpy()
            p2.append(x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean([predict(m, X) for m in m2], axis=0))))
        p2 = pl.concat(p2)
        p2.write_parquet(pred_path)
        print(f"[test] stage 2 scored ({time.time()-t0:.0f}s)", flush=True)
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(BL3 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL3 / "test_oth_ids.parquet")["entity_id"]
    from baseline import to_ids, write_id_lists
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    (out / "decision.json").write_text(json.dumps({"tag": a.tag, "model": "xgboost-cuda", "rule": rule}, indent=2))
    print(f"[test] {chosen.height:,} matches -> {out} ({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="gpu_test_like")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--n-train-entities", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--max-parts", type=int, default=None)
    a = ap.parse_args()
    judge_v4.MAX_PARTS = a.max_parts  # smoke tests only
    if a.cmd == "test" and not a.out_dir:
        ap.error("test needs --out-dir (a new folder)")
    train(a) if a.cmd == "train" else test(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
