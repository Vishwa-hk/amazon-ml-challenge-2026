"""
Judge v17 = judge v16 (same features) trained with 4 NESTED folds instead of 2 (27 Sep evening).

Why: with 2 folds each stage-2 model learns from only half of the kept train entities. With 4 folds each
model sees 75%, and the test prediction averages 4 models. The folds are nested in the old 2-fold split
(fold4 = 2 * fold_of() + sub-split), so every held-out entity's upstream stage-1/2 features stay
out-of-fold exactly as before - the held-out score stays comparable with v16's 0.98551.
The assembled training table is cached (runs/v17/<tag>/train_frame.parquet) so a re-train skips the
~12 min feature build.

  python judge_v17.py train --tag v17
  python judge_v17.py test --tag v17 --v8-dir ../../../runs/v11fr --blocking-dir ../../../runs/blocking_v11 --out-dir <new folder>
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
from sklearn.feature_extraction.text import TfidfVectorizer

import judge_v12
import pipeline_v8 as p8
import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region
from judge_v11 import CNT, copy_count_logratio, count_features
from judge_v12 import SWF, swap_features
from judge_v13 import RWF, assemble, margin, raw_features
from judge_v15 import WAF, align_features
from normalize import clean_text

RUN16 = PROJECT / "runs" / "v17"
TFF = ["tf_name", "tf_addr", "tf_name_margin"]
judge_v12.STOP_VARIANT = "b"
NF = 4


def fold4() -> pl.Expr:
    """nested in fold_of(): folds 0,1 = old fold 0; folds 2,3 = old fold 1."""
    return (fold_of() * 2 + (pl.col("i1").hash(seed=13) % 2)).cast(pl.UInt8)


def _cos(s1: list, oth: list, i1: np.ndarray, i2: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32, sublinear_tf=True)
    vec.fit(s1 + oth)
    A, B = vec.transform(s1), vec.transform(oth)          # rows are L2-normalised
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        out[s:s + chunk] = np.asarray(A[i1[s:s + chunk]].multiply(B[i2[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def tfidf_features(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1raw, _, oth, _ = p8.load_raw_v8(split)
    txt = lambda df, c: df.select(clean_text(pl.col(c)).str.strip_chars()).to_series().to_list()
    n1, n2, a1, a2 = txt(s1raw, "business_name"), txt(oth, "business_name"), txt(s1raw, "business_address"), txt(oth, "business_address")
    del s1raw, oth
    i1, i2 = pairs["i1"].to_numpy(), pairs["i2"].to_numpy()
    tn = _cos(n1, n2, i1, i2)
    del n1, n2
    ta = _cos(a1, a2, i1, i2)
    del a1, a2
    x = pairs.select("i1", "i2").with_columns(pl.Series("tf_name", tn), pl.Series("tf_addr", ta))
    return x.with_columns(margin(x, "tf_name", "tf_name_margin").cast(pl.Float32))


def cmd_train(a):
    t0 = time.time()
    d = RUN16 / a.tag
    if (d / "decision.json").exists():
        raise SystemExit(f"refusing to overwrite existing {d}")
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    lr = copy_count_logratio()
    frame = d / "train_frame.parquet"
    if frame.exists():
        df = pl.read_parquet(frame)
        print(f"cached training frame {df.height:,} rows ({time.time()-t0:.0f}s)", flush=True)
    else:
        p1 = pl.read_parquet(src / "train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005).join(kept, on="i1", how="semi")
        pairs = p1.select("i1", "i2")
        cf, sw, rw = count_features(p1, lr), swap_features(pairs, "train"), raw_features(p1, "train")
        wa = align_features(pairs, "train")
        tf = tfidf_features(pairs, "train")
        print(f"all extra features ({time.time()-t0:.0f}s)", flush=True)
        df = load_region(RUN8 / "train_features", pairs, kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
        df = assemble(df, cf, sw, rw).join(wa, on=["i1", "i2"], how="left").join(tf, on=["i1", "i2"], how="left") \
            .join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left") \
            .with_columns(pl.col("y").fill_null(0))
        del cf, sw, rw, wa, tf, p1, pairs
        df.write_parquet(frame)
        print(f"training frame cached ({time.time()-t0:.0f}s)", flush=True)
    df = df.with_columns(fold4().alias("fold"))
    F = FEATS + CNT + SWF + RWF + WAF + TFF
    oof = []
    for fold in range(NF):
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

    def score(prs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(prs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    cut = region(src / "train_p1.parquet", kept, 0.01)
    mine = owned(oof.join(cut, on=["i1", "i2"], how="semi"), 1000)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.6, 0.7)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.1, 1.25)])
    res = {json.dumps(r): score(decide(mine, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  v17 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "lr": lr.tolist(), "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v17 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} (v16: 0.98551) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d, bl = RUN16 / a.tag, a.v8_dir.resolve(), a.blocking_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    dec = json.loads((d / "decision.json").read_text())
    rule = json.loads(a.rule) if a.rule else dec["best_stage2"]["rule"]
    n = pl.read_parquet(bl / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    p1 = pl.read_parquet(v8d / "gpu_slot" / "test_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005)
    pairs = p1.select("i1", "i2")
    cf, sw, rw = count_features(p1, np.array(dec["lr"])), swap_features(pairs, "test"), raw_features(p1, "test")
    tf = tfidf_features(pairs, "test")
    cand = p1.filter(pl.col("p") >= a.cand_p1).select("i1", "i2")
    wa = align_features(cand, "test")
    x = assemble(load_region(v8d / "test_features", pairs, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet")), cf, sw, rw)
    x = x.join(tf, on=["i1", "i2"], how="left").join(cand, on=["i1", "i2"], how="semi").join(wa, on=["i1", "i2"], how="left")
    X = x.select(FEATS + CNT + SWF + RWF + WAF + TFF).to_numpy()
    ms = [xgb.Booster(model_file=str(d / f"stage2_fold{f}.ubj")) for f in range(NF)]
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
    ap.add_argument("--tag", default="v17")
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
