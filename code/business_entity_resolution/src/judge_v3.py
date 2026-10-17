"""
v3 ML judge: LightGBM pair classifier + decision layer.

train (python judge_v3.py train)
  * 2-fold cross-validation by Source-1 entity: the model trained on one fold
    predicts the other, so every train entity gets an out-of-fold (OOF)
    probability from a model that never saw it — a held-out estimate over all
    2.2M entities, with every entity competing for records exactly as in test.
  * Each fold model trains on ALL candidates of a random sample of entities (the
    real candidate mix, so probabilities stay calibrated), early-stopping on a
    held-back slice of that fold.
  * Decision rules are compared on the OOF predictions; the best is saved.

test (python judge_v3.py test)
  * averages the fold models, applies the saved decision rule, writes
    output/matching_results.tsv + output/candidate_pairs.tsv (the exact pairs
    the model scored), then runs the official validator.

Decision layer
  1. one owner per record: each S2/S3 record goes only to its most probable S1
     entity (the data guarantees at most one owner); exact ties are dropped.
  2. then either a global probability threshold, or per S1 entity the top-k set
     maximising expected F0.5 (k = 0 means "predict no match"):
        E[F_k] ~= 1.25*S_k / (1.25*S_k + 0.25*(T - S_k) + (k - S_k)),  S_k = sum of the top-k p,
        T = sum of p over the entity's candidates,  E[F_0] = prod(1 - p).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from features_v3 import FEATURES

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]
RUN = PROJECT / "runs" / "v3"
PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "num_threads": 12, "verbose": -1, "seed": 42,
}


def gt_pairs(split_dir: Path) -> pl.DataFrame:
    """True pairs as (i1, i2) row indices, plus number of true matches per S1."""
    s1_ids = pl.read_parquet(split_dir / "train_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(split_dir / "train_oth_ids.parquet")["entity_id"]
    s1_map = pl.DataFrame({"s1": s1_ids, "i1": pl.arange(0, len(s1_ids), eager=True, dtype=pl.UInt32)})
    o_map = pl.DataFrame({"r": o_ids, "i2": pl.arange(0, len(o_ids), eager=True, dtype=pl.UInt32)})
    gt = pl.read_parquet(PROJECT / "data_cache" / "train_ground_truth.parquet")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
            .join(s1_map, on="s1").join(o_map, on="r").select("i1", "i2"))


def fold_of(col="i1") -> pl.Expr:
    return (pl.col(col).hash(seed=7) % 2).cast(pl.UInt8)


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, n_s1: int) -> dict:
    """Same metric as scorer.py, vectorised on (i1, i2) pairs over ALL S1 entities."""
    nt = truth.group_by("i1").len("nt")
    npd = pred.group_by("i1").len("np")
    tp = pred.join(truth, on=["i1", "i2"], how="semi").group_by("i1").len("tp")
    t = (pl.DataFrame({"i1": pl.arange(0, n_s1, eager=True, dtype=pl.UInt32)})
         .join(nt, on="i1", how="left").join(npd, on="i1", how="left").join(tp, on="i1", how="left")
         .fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp"))))
    r = t.select(f.mean().alias("f"), pl.col("tp").sum(), pl.col("np").sum(), pl.col("nt").sum()).row(0, named=True)
    return {"f05": r["f"], "precision": r["tp"] / max(r["np"], 1), "recall": r["tp"] / r["nt"]}


def one_owner(p: pl.DataFrame) -> pl.DataFrame:
    """Keep each record only for its most probable S1 entity (drop exact ties)."""
    best = p.filter(pl.col("p") == pl.col("p").max().over("i2"))
    return best.filter(pl.len().over("i2") == 1)


def owned(p: pl.DataFrame, max_rank=None) -> pl.DataFrame:
    """Rank cut-off (the model's candidate set) + one owner per record."""
    if max_rank:
        p = p.filter(pl.col("rank") <= max_rank)
    return one_owner(p)


def decide(p: pl.DataFrame, rule: dict, already_owned: bool = False) -> pl.DataFrame:
    """Apply a decision rule to (i1, i2, p[, rank]) → chosen (i1, i2)."""
    if not already_owned:
        p = owned(p, rule.get("max_rank"))
    if rule["kind"] == "threshold":
        return p.filter(pl.col("p") >= rule["t"]).select("i1", "i2")
    # expected-F0.5 top-k per S1 entity
    p = p.filter(pl.col("p") >= rule.get("floor", 0.01)).sort(["i1", "p"], descending=[False, True])
    p = p.with_columns(
        pl.col("p").cum_sum().over("i1").alias("S"),
        pl.int_range(1, pl.len() + 1).over("i1").cast(pl.Float64).alias("k"),
        (pl.col("p").sum().over("i1") * rule.get("t_scale", 1.0)).alias("T"),
        (1 - pl.col("p")).clip(1e-9, 1.0).log().sum().over("i1").exp().alias("F0"),
    )
    p = p.with_columns((1.25 * pl.col("S") / (1.25 * pl.col("S") + 0.25 * (pl.col("T") - pl.col("S")).clip(0)
                                               + (pl.col("k") - pl.col("S")))).alias("Fk"))
    p = p.with_columns(pl.col("Fk").max().over("i1").alias("Fmax"))
    kbest = p.filter(pl.col("Fk") == pl.col("Fmax")).group_by("i1").agg(pl.col("k").min().alias("kb"), pl.col("F0").first())
    p = p.join(kbest, on="i1").filter((pl.col("Fmax") > pl.col("F0")) & (pl.col("k") <= pl.col("kb")))
    return p.select("i1", "i2")


def load_features(split: str, cols=None) -> pl.LazyFrame:
    return pl.scan_parquet(RUN / f"{split}_features" / "part-*.parquet").select(cols or (["i1", "i2"] + FEATURES))


def train(a) -> None:
    t0 = time.time()
    truth = gt_pairs(PROJECT / "runs" / "blocking_v2")
    n_s1 = pl.read_parquet(PROJECT / "runs" / "blocking_v2" / "train_s1_ids.parquet").height
    ent = pl.DataFrame({"i1": pl.arange(0, n_s1, eager=True, dtype=pl.UInt32)}).with_columns(fold_of().alias("fold"))
    oof, models, importance = [], [], {}
    for fold in (0, 1):
        train_ents = ent.filter(pl.col("fold") != fold).sample(a.n_train_entities, seed=fold).select("i1")
        stop_ents = train_ents.sample(fraction=0.1, seed=99)
        df = (load_features("train").join(train_ents.lazy(), on="i1", how="semi").collect()
              .join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left")
              .with_columns(pl.col("y").fill_null(0)))
        tr, va = df.join(stop_ents, on="i1", how="anti"), df.join(stop_ents, on="i1", how="semi")
        print(f"[fold {fold}] train {tr.height:,} pairs ({tr['y'].sum():,} true) | early-stop {va.height:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
        dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
        dva = lgb.Dataset(va.select(FEATURES).to_numpy(), va["y"].to_numpy(), reference=dtr)
        del df, tr
        m = lgb.train(PARAMS, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        m.save_model(str(RUN / f"model_fold{fold}.txt"))
        models.append(m)
        for n, v in zip(FEATURES, m.feature_importance("gain")):
            importance[n] = importance.get(n, 0.0) + float(v)
        print(f"[fold {fold}] {m.best_iteration} trees ({time.time()-t0:.0f}s); predicting held-out fold ...", flush=True)
        held = ent.filter(pl.col("fold") == fold).select("i1")
        for part in sorted((RUN / "train_features").glob("part-*.parquet")):
            x = pl.read_parquet(part, columns=["i1", "i2"] + FEATURES).join(held, on="i1", how="semi")
            if x.height:
                oof.append(x.select("i1", "i2", "rank").with_columns(
                    pl.Series("p", m.predict(x.select(FEATURES).to_numpy(), num_iteration=m.best_iteration))))
    oof = pl.concat(oof)
    oof.write_parquet(RUN / "train_oof.parquet")
    print(f"OOF predictions: {oof.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
    top = sorted(importance.items(), key=lambda x: -x[1])
    print("feature importance (gain): " + ", ".join(f"{n}={v/1e6:.1f}M" for n, v in top))

    rules = []
    for mr in (30, 100):
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            rules.append({"kind": "threshold", "t": t, "max_rank": mr})
        for ts in (0.9, 1.0, 1.1, 1.25):
            rules.append({"kind": "expected_f05", "t_scale": ts, "max_rank": mr, "floor": 0.01})
    best = None
    base = {mr: owned(oof, mr) for mr in (30, 100)}
    print("\ndecision rules on OOF (all 2,206,821 train entities):")
    for r in rules:
        res = macro_f05(decide(base[r["max_rank"]], r, already_owned=True), truth, n_s1)
        print(f"  {json.dumps(r):<75} F0.5 {res['f05']:.5f} | P {res['precision']:.4f} R {res['recall']:.4f}", flush=True)
        if best is None or res["f05"] > best[1]["f05"]:
            best = (r, res)
    print(f"\nBEST: {best[0]} -> OOF macro F0.5 {best[1]['f05']:.5f}")
    (RUN / "decision.json").write_text(json.dumps({"rule": best[0], "oof": best[1],
                                                    "importance": dict(top)}, indent=2))
    print(f"done ({time.time()-t0:.0f}s)")


def predict_test() -> pl.DataFrame:
    """Average both fold models over all test candidates; save to runs/v3/test_pred.parquet."""
    t0 = time.time()
    models = [lgb.Booster(model_file=str(RUN / f"model_fold{f}.txt")) for f in (0, 1)]
    preds = []
    for part in sorted((RUN / "test_features").glob("part-*.parquet")):
        x = pl.read_parquet(part, columns=["i1", "i2"] + FEATURES)
        X = x.select(FEATURES).to_numpy()
        p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
        preds.append(x.select("i1", "i2", "rank").with_columns(pl.Series("p", p)))
    p = pl.concat(preds)
    p.write_parquet(RUN / "test_pred.parquet")
    print(f"[test] scored {p.height:,} pairs ({time.time()-t0:.0f}s) -> {RUN / 'test_pred.parquet'}", flush=True)
    return p


def decide_by_country(p: pl.DataFrame, rule: dict, split: str) -> pl.DataFrame:
    """Rule may hold per-country overrides: {"default": {...}, "by_country": {"France": {...}}}.
    Records only ever compete within one country (blocking is within country), so
    deciding each country separately is exact."""
    if "by_country" not in rule:
        return decide(p, rule)
    ctry = pl.read_parquet(PROJECT / "data_cache" / f"{split}_source1.parquet").with_row_index("i1").select("i1", "country")
    p = p.join(ctry, on="i1")
    out = []
    for c in p["country"].unique().to_list():
        out.append(decide(p.filter(pl.col("country") == c), rule["by_country"].get(c, rule["default"])))
    return pl.concat(out)


def test(a) -> None:
    rule = json.loads(a.rule) if a.rule else json.loads((RUN / "decision.json").read_text())["rule"]
    t0 = time.time()
    p = pl.read_parquet(RUN / "test_pred.parquet") if a.reuse_pred else predict_test()
    print(f"[test] rule {rule}", flush=True)
    chosen = decide_by_country(p, rule, "test")
    mr = rule.get("max_rank") or rule.get("default", {}).get("max_rank")
    cands = p.filter(pl.col("rank") <= mr) if mr else p
    s1_ids = pl.read_parquet(PROJECT / "runs" / "blocking_v2" / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(PROJECT / "runs" / "blocking_v2" / "test_oth_ids.parquet")["entity_id"]
    from baseline import to_ids, write_id_lists
    # Never write into output/ (it holds whatever was last uploaded) and never
    # overwrite an existing submission: each version gets its own new folder.
    out = a.out_dir.resolve()  # absolute: the validator runs from student_resource/
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f} — pass a new --out-dir")
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(cands.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    (out / "decision.json").write_text(json.dumps({"rule": rule}, indent=2))
    print(f"[test] {chosen.height:,} matches written to {out} ({time.time()-t0:.0f}s)", flush=True)
    sr = PROJECT / "student_resource"
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"], cwd=sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test", "predict"])
    ap.add_argument("--rule", help="JSON decision rule (default: runs/v3/decision.json)")
    ap.add_argument("--reuse-pred", action="store_true", help="use saved runs/v3/test_pred.parquet")
    ap.add_argument("--n-train-entities", type=int, default=400_000)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "submissions" / "v3_expf05_top100",
                    help="NEW folder for this version's output files (never output/)")
    a = ap.parse_args()
    {"train": train, "test": test, "predict": lambda _: predict_test()}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
