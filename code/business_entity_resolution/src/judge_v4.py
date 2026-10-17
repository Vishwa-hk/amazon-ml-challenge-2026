"""
v4 judge: two-stage LightGBM over short-lists v3, trained and validated in a
TEST-LIKE universe.

Test-like universe (train only): the test set holds ~40% decoy records vs 26% in
train (label-free check, 26 Sep). Decoys look like variants of businesses that
have no Source-1 record ("hidden twins"). Dropping a random `drop_frac` of train
S1 entities turns their records into exactly such decoys: 0.19 lifts the decoy
share to ~40%. Records stay in the pool; only the entity (and its truth) goes.
The record-competition features (rank_r, r_cands, gap_r) are recomputed without
the dropped entities.

Stage 1: LightGBM on the v4 pair features (2-fold OOF by S1 entity).
Stage 2: LightGBM on v4 features + evidence derived from stage-1 probabilities:
    p1, rank / gap / sum / confident-count of p1 within the S1 entity,
    rank of p1 within the record, best rival p1 for the record, strong-rival
    count, margin over the best rival, and cluster consensus:
      num_support     other confident (p1>0.7) candidates of this S1 sharing a
                      house number with this record
      devnum_support  same, counting only numbers the S1 itself does NOT have
                      (a twin's deviant number is shared by its siblings)
      name_support    other confident candidates with the same word-set name
      s1_cluster      confident candidates sharing a number with the S1 itself
Decision layer as v3 (one owner per record, then expected-F0.5 or threshold),
compared on stage-2 OOF.

Usage:
  python judge_v4.py train [--drop-frac 0.19] [--tag test_like]
  python judge_v4.py test  --tag test_like --out-dir <new folder> [--rule '<json>']
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

from features_v4 import FEATURES_V4
from judge_v3 import PARAMS, decide, fold_of, macro_f05, owned

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]
RUN4, BL3 = PROJECT / "runs" / "v4", PROJECT / "runs" / "blocking_v3"
CTX = ["rank_r", "r_cands", "gap_r"]
S2F = ["p1", "p1_rank_i1", "p1_gap_i1", "sum_p1_i1", "n_conf_i1", "p1_rank_i2", "best_rival", "n_rivals",
       "margin_rival", "num_support", "devnum_support", "name_support", "s1_cluster"]
CONF = 0.7
MAX_PARTS = None  # smoke tests: only the first N feature parts (set from --max-parts)


def parts(split: str):
    p = sorted((RUN4 / f"{split}_features").glob("part-*.parquet"))
    return p[:MAX_PARTS] if MAX_PARTS else p


def truth_pairs() -> pl.DataFrame:
    s1 = pl.read_parquet(BL3 / "train_s1_ids.parquet")["entity_id"]
    o = pl.read_parquet(BL3 / "train_oth_ids.parquet")["entity_id"]
    s1m = pl.DataFrame({"s1": s1, "i1": pl.arange(0, len(s1), eager=True, dtype=pl.UInt32)})
    om = pl.DataFrame({"r": o, "i2": pl.arange(0, len(o), eager=True, dtype=pl.UInt32)})
    gt = pl.read_parquet(PROJECT / "data_cache" / "train_ground_truth.parquet")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
            .join(s1m, on="s1").join(om, on="r").select("i1", "i2"))


def kept_entities(n_s1: int, drop_frac: float, seed: int = 11) -> pl.DataFrame:
    ent = pl.DataFrame({"i1": pl.arange(0, n_s1, eager=True, dtype=pl.UInt32)})
    return ent if drop_frac <= 0 else ent.sample(fraction=1 - drop_frac, seed=seed).sort("i1")


def context_table(split: str, kept: pl.DataFrame, path: Path) -> Path:
    """Record-competition features recomputed over the kept universe, written to disk
    (held in memory it would sit alongside everything else for the whole run)."""
    base = pl.concat([pl.read_parquet(p, columns=["i1", "i2", "score"]) for p in parts(split)])
    base.join(kept, on="i1", how="semi").select(
        "i1", "i2",
        (pl.col("score").max().over("i2") - pl.col("score")).cast(pl.Float32).alias("gap_r"),
        pl.col("score").rank("min", descending=True).over("i2").cast(pl.Float32).alias("rank_r"),
        pl.len().over("i2").cast(pl.Float32).alias("r_cands"),
    ).sort("i1").write_parquet(path, row_group_size=2_000_000)
    return path


def _range(src, lo, hi) -> pl.DataFrame:
    """Rows of a parquet file / glob with lo <= i1 <= hi (row-group pushdown)."""
    return pl.scan_parquet(src).filter((pl.col("i1") >= lo) & (pl.col("i1") <= hi)).collect()


def load_part(path, kept, ctx=None, extra=None, cols=None) -> pl.DataFrame:
    """One feature part, restricted to kept entities, with context / stage-2 columns
    swapped in from on-disk tables."""
    x = pl.read_parquet(path, columns=cols)
    x = x.join(kept, on="i1", how="semi")
    if x.height and (ctx is not None or extra is not None):
        lo, hi = x["i1"].min(), x["i1"].max()
        if ctx is not None:
            x = x.drop(CTX).join(_range(ctx, lo, hi), on=["i1", "i2"])
        if extra is not None:
            x = x.join(_range(extra, lo, hi), on=["i1", "i2"])
    return x


def fit(X, y, Xv, yv, rounds):
    dtr = lgb.Dataset(X, y, free_raw_data=True).construct()
    dva = lgb.Dataset(Xv, yv, reference=dtr).construct()
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])


def cv_stage(name, split_parts, kept, ctx, truth, feats, extra, n_train, rounds, out_dir, t0):
    """2-fold OOF by S1 entity; returns OOF (i1, i2, rank, p) and saves both fold models."""
    ent = kept.with_columns(fold_of().alias("fold"))
    oof = []
    for fold in (0, 1):
        tr_e = ent.filter(pl.col("fold") != fold).sample(min(n_train, ent.filter(pl.col("fold") != fold).height), seed=fold).select("i1")
        st_e = tr_e.sample(fraction=0.1, seed=99)
        rows = []
        for p in split_parts:
            x = load_part(p, tr_e, ctx, extra)
            if x.height:
                rows.append(x)
        df = pl.concat(rows).join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
            .with_columns(pl.col("y").fill_null(0))
        del rows
        # build the arrays one piece at a time, freeing each table copy as soon as it's used
        va = df.join(st_e, on="i1", how="semi")
        Xv, yv = va.select(feats).to_numpy(), va["y"].to_numpy()
        del va
        tr = df.join(st_e, on="i1", how="anti")
        del df
        n_tr, n_pos = tr.height, int(tr["y"].sum())
        X, y = tr.select(feats).to_numpy(), tr["y"].to_numpy()
        del tr
        print(f"[{name} fold {fold}] train {n_tr:,} pairs ({n_pos:,} true) ({time.time()-t0:.0f}s)", flush=True)
        m = fit(X, y, Xv, yv, rounds)
        del X, y, Xv, yv
        m.save_model(str(out_dir / f"{name}_fold{fold}.txt"))
        held = ent.filter(pl.col("fold") == fold).select("i1")
        for p in split_parts:
            x = load_part(p, held, ctx, extra)
            if x.height:
                oof.append(x.select("i1", "i2", "rank").with_columns(
                    pl.Series("p", m.predict(x.select(feats).to_numpy(), num_iteration=m.best_iteration))))
        print(f"[{name} fold {fold}] {m.best_iteration} trees, held-out predicted ({time.time()-t0:.0f}s)", flush=True)
    return pl.concat(oof)


def stage2_features(p1: pl.DataFrame, split: str, out_dir: Path, chunk: int = 100_000) -> Path:
    """Evidence from stage-1 probabilities (see module docstring), written chunk by
    chunk to out_dir/part-*.parquet (the full table would be ~10 GB in memory).
    Returns the glob to scan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from baseline import load_source
    from blocking_v2 import prepare
    dd, cd = PROJECT / "student_resource" / "dataset", PROJECT / "data_cache"
    s1p = prepare(load_source(dd, cd, split, "source1")).select(pl.col("i").alias("i1"), pl.col("nums").alias("n1"))
    othp = prepare(pl.concat([load_source(dd, cd, split, "source2"), load_source(dd, cd, split, "source3")])).select(
        pl.col("i").alias("i2"), pl.col("nums").alias("n2"), pl.col("key_ts").alias("kts"))
    p1 = p1.select("i1", "i2", pl.col("p").alias("p1"))
    # record-level quantities need ALL competing S1 entities, so compute them globally
    p1 = p1.with_columns(pl.col("p1").rank("min", descending=True).over("i2").cast(pl.Float32).alias("p1_rank_i2"))
    top = p1.group_by("i2").agg(
        pl.col("p1").max().alias("t1"),
        pl.col("i1").sort_by("p1", descending=True).first().alias("a1"),
        pl.col("p1").sort(descending=True).slice(1, 1).first().alias("t2"),
        (pl.col("p1") > 0.5).sum().alias("c05"))
    n = int(p1["i1"].max()) + 1
    for part, start in enumerate(range(0, n, chunk)):
        c = p1.filter((pl.col("i1") >= start) & (pl.col("i1") < start + chunk))
        if not c.height:
            continue
        c = c.join(top, on="i2").with_columns(
            pl.when(pl.col("a1") == pl.col("i1")).then(pl.col("t2")).otherwise(pl.col("t1")).fill_null(0.0).alias("best_rival"),
            (pl.col("c05") - (pl.col("p1") > 0.5).cast(pl.UInt32)).cast(pl.Float32).alias("n_rivals"),
        ).drop("t1", "a1", "t2", "c05")
        c = c.with_columns(
            pl.col("p1").rank("min", descending=True).over("i1").cast(pl.Float32).alias("p1_rank_i1"),
            (pl.col("p1").max().over("i1") - pl.col("p1")).alias("p1_gap_i1"),
            pl.col("p1").sum().over("i1").alias("sum_p1_i1"),
            (pl.col("p1") > 0.5).sum().over("i1").cast(pl.Float32).alias("n_conf_i1"),
            (pl.col("p1") - pl.col("best_rival")).alias("margin_rival"),
        )
        c = c.join(othp, on="i2").join(s1p, on="i1").with_columns((pl.col("p1") > CONF).alias("conf"))
        conf = c.filter(pl.col("conf"))
        num_cnt = conf.select("i1", pl.col("n2").alias("num")).explode("num").drop_nulls().group_by("i1", "num").len("nc")
        name_cnt = conf.filter(pl.col("kts") != "").group_by("i1", "kts").len("kc")
        s1num = c.select("i1", pl.col("n1").alias("num")).unique("i1").explode("num").drop_nulls().with_columns(pl.lit(True).alias("in_s1"))
        ex = (c.select("i1", "i2", "conf", pl.col("n2").alias("num")).explode("num").drop_nulls()
              .join(num_cnt, on=["i1", "num"], how="left").join(s1num, on=["i1", "num"], how="left")
              .with_columns((pl.col("nc").fill_null(0) - pl.col("conf").cast(pl.UInt32)).clip(0).alias("sup"),
                            pl.col("in_s1").fill_null(False)))
        sup = ex.group_by("i1", "i2").agg(
            pl.col("sup").max().cast(pl.Float32).alias("num_support"),
            pl.col("sup").filter(~pl.col("in_s1")).max().fill_null(0).cast(pl.Float32).alias("devnum_support"))
        s1c = (conf.select("i1", "i2", pl.col("n2"), pl.col("n1")).with_columns(
            (pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("sh")).group_by("i1")
            .agg(pl.col("sh").sum().cast(pl.Float32).alias("s1_cluster")))
        c = (c.join(sup, on=["i1", "i2"], how="left").join(name_cnt, on=["i1", "kts"], how="left")
             .join(s1c, on="i1", how="left")
             .with_columns(pl.col("num_support").fill_null(0), pl.col("devnum_support").fill_null(0),
                           (pl.col("kc").fill_null(0) - pl.col("conf").cast(pl.UInt32)).clip(0).cast(pl.Float32).alias("name_support"),
                           pl.col("s1_cluster").fill_null(0)))
        c.select(["i1", "i2"] + [pl.col(f).cast(pl.Float32) for f in S2F]).write_parquet(out_dir / f"part-{part:03d}.parquet")
    return out_dir / "part-*.parquet"


def train(a) -> None:
    t0 = time.time()
    d = RUN4 / a.tag
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"refusing to overwrite existing {d}")
    d.mkdir(parents=True, exist_ok=True)
    n_s1 = pl.read_parquet(BL3 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, a.drop_frac)
    truth_all = truth_pairs()
    truth = truth_all.join(kept, on="i1", how="semi")
    n_rec = pl.read_parquet(BL3 / "train_oth_ids.parquet").height
    print(f"universe '{a.tag}': {kept.height:,} of {n_s1:,} S1 kept (drop {a.drop_frac:.0%}); decoy share of records "
          f"{1 - truth['i2'].n_unique()/n_rec:.1%} (full train {1 - truth_all['i2'].n_unique()/n_rec:.1%})", flush=True)
    ctx = context_table("train", kept, d / "train_ctx.parquet") if a.drop_frac > 0 else None
    P = parts("train")
    oof1 = cv_stage("stage1", P, kept, ctx, truth, FEATURES_V4, None, a.n_train_entities, a.rounds, d, t0)
    oof1.write_parquet(d / "train_p1.parquet")
    s2 = stage2_features(oof1, "train", d / "train_s2")
    del oof1  # reloaded from disk for the rule comparison; frees ~3 GB during stage 2
    print(f"stage-2 features built ({time.time()-t0:.0f}s)", flush=True)
    oof2 = cv_stage("stage2", P, kept, ctx, truth, FEATURES_V4 + S2F, s2, a.n_train_entities, a.rounds, d, t0)
    oof2.write_parquet(d / "train_p2.parquet")

    n_eval = n_s1
    remap = kept.with_row_index("j")  # score over kept entities only: dropped ones aren't part of this universe
    def score(pairs):
        return macro_f05(pairs.join(remap, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(remap, on="i1").select(pl.col("j").alias("i1"), "i2"), kept.height)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.8, 1.0, 1.25, 1.5, 2.0)])
    res = {}
    for stage, oof in (("stage1", pl.read_parquet(d / "train_p1.parquet")), ("stage2", oof2)):
        base = owned(oof, None)
        for r in rules:
            res[(stage, json.dumps(r))] = score(decide(base, r, already_owned=True))
            f = res[(stage, json.dumps(r))]
            print(f"  {stage} {json.dumps(r):<70} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max((k for k in res if k[0] == "stage2"), key=lambda k: res[k]["f05"])
    best1 = max((k for k in res if k[0] == "stage1"), key=lambda k: res[k]["f05"])
    summary = {"tag": a.tag, "drop_frac": a.drop_frac, "best_stage2": {"rule": json.loads(best[1]), **res[best]},
               "best_stage1": {"rule": json.loads(best1[1]), **res[best1]}}
    (d / "decision.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBEST stage1 {res[best1]['f05']:.5f} {best1[1]}\nBEST stage2 {res[best]['f05']:.5f} {best[1]}\ndone ({time.time()-t0:.0f}s)")


def test(a) -> None:
    t0 = time.time()
    d = RUN4 / a.tag
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
        m1 = [lgb.Booster(model_file=str(d / f"stage1_fold{f}.txt")) for f in (0, 1)]
        p1 = []
        for p in P:
            x = pl.read_parquet(p)
            X = x.select(FEATURES_V4).to_numpy()
            p1.append(x.select("i1", "i2", "rank").with_columns(
                pl.Series("p", np.mean([m.predict(X, num_iteration=m.best_iteration) for m in m1], axis=0))))
        p1 = pl.concat(p1)
        p1.write_parquet(d / "test_p1.parquet")
        print(f"[test] stage 1 scored {p1.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
        s2 = stage2_features(p1, "test", d / "test_s2")
        del p1
        m2 = [lgb.Booster(model_file=str(d / f"stage2_fold{f}.txt")) for f in (0, 1)]
        p2 = []
        for p in P:
            x = load_part(p, kept, None, s2)
            X = x.select(FEATURES_V4 + S2F).to_numpy()
            p2.append(x.select("i1", "i2", "rank").with_columns(
                pl.Series("p", np.mean([m.predict(X, num_iteration=m.best_iteration) for m in m2], axis=0))))
        p2 = pl.concat(p2)
        p2.write_parquet(pred_path)
        print(f"[test] stage 2 scored ({time.time()-t0:.0f}s)", flush=True)
    from judge_v3 import decide_by_country
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(BL3 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL3 / "test_oth_ids.parquet")["entity_id"]
    from baseline import to_ids, write_id_lists
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    (out / "decision.json").write_text(json.dumps({"tag": a.tag, "rule": rule}, indent=2))
    print(f"[test] {chosen.height:,} matches -> {out} ({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="test_like")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--n-train-entities", type=int, default=400_000)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--rule", help="JSON decision rule for test (default: best stage-2 rule)")
    ap.add_argument("--out-dir", type=Path, help="NEW folder for test outputs")
    ap.add_argument("--max-parts", type=int, default=None, help="smoke test: only the first N feature parts")
    a = ap.parse_args()
    global MAX_PARTS
    MAX_PARTS = a.max_parts
    if a.cmd == "test" and not a.out_dir:
        ap.error("test needs --out-dir (a new folder)")
    train(a) if a.cmd == "train" else test(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
