"""
v6 = v5 (short-lists v3 + v4 features + GPU two-stage judge, test-like universe)
     + transliteration of native-script S2/S3 names BEFORE short-lists and features.

The dictionary (runs/translit/dictionary.parquet, built by run_translit.py from
training pairs only) maps native-script words to English words; known words are
replaced, unknown words are left as they are. The `non_latin` feature keeps
flagging records whose ORIGINAL name was native script, so the judge still knows
a translated name is less certain.

All earlier code is reused by import and left unmodified. New folders only:
runs/blocking_v6/ (candidates), runs/v6/ (features, models, predictions).

Commands (run in this order):
  python pipeline_v6.py blocking --split train      # + recall / oracle report
  python pipeline_v6.py blocking --split test
  python pipeline_v6.py features --split train
  python pipeline_v6.py features --split test
  python pipeline_v6.py train [--tag gpu_translit]
  python pipeline_v6.py test --tag gpu_translit --out-dir <new folder> [--rule '<json>']
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

from baseline import load_source, to_ids, write_id_lists
from blocking_v2 import _freq, prepare
from blocking_v3 import FLAGS3, build_candidates
from features_v3 import chunk_features, record_table
from features_v4 import FEATURES_V4
from judge_v3 import decide, decide_by_country, macro_f05, owned
from judge_v4 import CONF, CTX, S2F, kept_entities, load_part
from judge_v5 import cv_stage, predict
from run_blocking_v3 import oracle
from translit import LATIN_ONLY, transliterate

PROJECT = Path(__file__).resolve().parents[3]
DD, CD = PROJECT / "student_resource" / "dataset", PROJECT / "data_cache"
BL6, RUN6 = PROJECT / "runs" / "blocking_v6", PROJECT / "runs" / "v6"
DICT = PROJECT / "runs" / "translit" / "dictionary.parquet"


def load_raw(split: str):
    """S1 raw, #S2 rows, S2+S3 raw with transliterated names, original-native flag."""
    s1 = load_source(DD, CD, split, "source1")
    s2 = load_source(DD, CD, split, "source2")
    oth = pl.concat([s2, load_source(DD, CD, split, "source3")])
    orig_native = ~oth["business_name"].fill_null("").str.contains(LATIN_ONLY)
    return s1, s2.height, transliterate(oth, pl.read_parquet(DICT)), orig_native


def refuse(path: Path):
    if path.exists() and (path.is_file() or any(path.iterdir())):
        raise SystemExit(f"refusing to overwrite existing {path}")


def truth(split: str = "train") -> pl.DataFrame:
    s1 = pl.read_parquet(BL6 / f"{split}_s1_ids.parquet")["entity_id"]
    o = pl.read_parquet(BL6 / f"{split}_oth_ids.parquet")["entity_id"]
    s1m = pl.DataFrame({"s1": s1, "i1": pl.arange(0, len(s1), eager=True, dtype=pl.UInt32)})
    om = pl.DataFrame({"r": o, "i2": pl.arange(0, len(o), eager=True, dtype=pl.UInt32)})
    gt = pl.read_parquet(CD / "train_ground_truth.parquet")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
            .join(s1m, on="s1").join(om, on="r").select("i1", "i2"))


# ------------------------------------------------------------------ blocking
def cmd_blocking(a):
    full, cut = BL6 / f"{a.split}_candidates.parquet", BL6 / f"{a.split}_candidates_top100a20.parquet"
    refuse(full)
    BL6.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s1raw, _, oth_t, native = load_raw(a.split)
    print(f"[{a.split}] {int(native.sum()):,} native-script S2/S3 names transliterated ({time.time()-t0:.0f}s)", flush=True)
    cand = build_candidates(prepare(s1raw), prepare(oth_t), top_n=150, addr_n=20)
    cand.write_parquet(full)
    pl.DataFrame({"entity_id": s1raw["entity_id"]}).write_parquet(BL6 / f"{a.split}_s1_ids.parquet")
    pl.DataFrame({"entity_id": oth_t["entity_id"]}).write_parquet(BL6 / f"{a.split}_oth_ids.parquet")
    del cand
    pl.scan_parquet(full).filter((pl.col("rank") <= 100) | (pl.col("arank") <= 20)).sink_parquet(cut, row_group_size=2_000_000)
    n_cut = pl.scan_parquet(cut).select(pl.len()).collect().item()
    print(f"[{a.split}] candidates saved; top-100 + address-20 cut: {n_cut:,} ({time.time()-t0:.0f}s)", flush=True)
    if a.split == "train":
        pairs = truth("train")
        c = pl.scan_parquet(cut).select("i1", "i2").collect()
        r, f = oracle(c, pairs, s1raw.height)
        print(f"[train] top-100 + address-20: recall {r:.2%} | oracle F0.5 {f:.4f}   (v3 without transliteration: 96.15% / 0.9863)")
        ctry = s1raw.with_row_index("i1").select("i1", "country")
        hit = pairs.join(c, on=["i1", "i2"], how="semi").join(ctry, on="i1").group_by("country").len("hit")
        tot = pairs.join(ctry, on="i1").group_by("country").len("tot")
        for row in hit.join(tot, on="country").sort("country").iter_rows(named=True):
            print(f"   {row['country']:<6} recall {row['hit']/row['tot']:.2%}")


# ------------------------------------------------------------------ features
def cmd_features(a):
    out = RUN6 / f"{a.split}_features"
    refuse(out)
    t0 = time.time()
    cand_path = BL6 / f"{a.split}_candidates_top100a20.parquet"
    s1raw, n_s2, oth_t, native = load_raw(a.split)
    s1p, othp = prepare(s1raw), prepare(oth_t)
    out.mkdir(parents=True, exist_ok=True)
    ctx_path = RUN6 / f"{a.split}_features_context.parquet"
    slim = pl.scan_parquet(cand_path).select("i1", "i2", pl.col("score").cast(pl.Float32)).collect()
    slim.select("i1", "i2",
                (pl.col("score").max().over("i2") - pl.col("score")).alias("gap_r"),
                pl.col("score").rank("min", descending=True).over("i2").cast(pl.UInt32).alias("rank_r"),
                pl.len().over("i2").cast(pl.UInt32).alias("r_cands")).write_parquet(ctx_path, row_group_size=2_000_000)
    del slim
    tok_f, aw_f = _freq(s1p, othp, "toks"), _freq(s1p, othp, "awords")
    S = record_table(s1p, s1raw)
    O = record_table(othp, oth_t, n_s2=n_s2).with_columns(pl.Series("non_latin", native))  # ORIGINAL script flag
    n, chunk = s1p.height, 50_000
    for part, start in enumerate(range(0, n, chunk)):
        in_chunk = (pl.col("i1") >= start) & (pl.col("i1") < start + chunk)
        c = (pl.scan_parquet(cand_path).filter(in_chunk).collect()
             .join(pl.scan_parquet(ctx_path).filter(in_chunk).collect(), on=["i1", "i2"])
             .with_columns((pl.col("score").max().over("i1") - pl.col("score")).alias("gap_s1")))
        f = chunk_features(c, S, O, tok_f, aw_f)
        k = pl.col("keys")
        extra = c.select("i1", "i2",
                         *[((k & FLAGS3[x]) > 0).cast(pl.Float32).alias(f"k_{x.lower()}") for x in ("NT2", "NA3", "TA2", "AA", "TT")],
                         pl.col("arank").cast(pl.Float32))
        f.join(extra, on=["i1", "i2"]).write_parquet(out / f"part-{part:03d}.parquet")
        if part % 8 == 0:
            print(f"  features: S1 {min(start+chunk, n):>9,}/{n:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.split}] features done ({time.time()-t0:.0f}s) -> {out}", flush=True)


# ------------------------------------------------------------------ judge
def parts(split: str):
    return sorted((RUN6 / f"{split}_features").glob("part-*.parquet"))


def context_table(kept: pl.DataFrame, path: Path) -> Path:
    base = pl.concat([pl.read_parquet(p, columns=["i1", "i2", "score"]) for p in parts("train")])
    base.join(kept, on="i1", how="semi").select(
        "i1", "i2",
        (pl.col("score").max().over("i2") - pl.col("score")).cast(pl.Float32).alias("gap_r"),
        pl.col("score").rank("min", descending=True).over("i2").cast(pl.Float32).alias("rank_r"),
        pl.len().over("i2").cast(pl.Float32).alias("r_cands"),
    ).sort("i1").write_parquet(path, row_group_size=2_000_000)
    return path


def stage2_features(p1: pl.DataFrame, split: str, out_dir: Path, chunk: int = 100_000):
    """Same evidence as judge_v4.stage2_features, but on the TRANSLITERATED records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s1raw, _, oth_t, _ = load_raw(split)
    s1p = prepare(s1raw).select(pl.col("i").alias("i1"), pl.col("nums").alias("n1"))
    othp = prepare(oth_t).select(pl.col("i").alias("i2"), pl.col("nums").alias("n2"), pl.col("key_ts").alias("kts"))
    del s1raw, oth_t
    p1 = p1.select("i1", "i2", pl.col("p").alias("p1"))
    p1 = p1.with_columns(pl.col("p1").rank("min", descending=True).over("i2").cast(pl.Float32).alias("p1_rank_i2"))
    top = p1.group_by("i2").agg(
        pl.col("p1").max().alias("t1"), pl.col("i1").sort_by("p1", descending=True).first().alias("a1"),
        pl.col("p1").sort(descending=True).slice(1, 1).first().alias("t2"), (pl.col("p1") > 0.5).sum().alias("c05"))
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
        s1c = (conf.select("i1", "i2", "n2", "n1").with_columns((pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("sh"))
               .group_by("i1").agg(pl.col("sh").sum().cast(pl.Float32).alias("s1_cluster")))
        c = (c.join(sup, on=["i1", "i2"], how="left").join(name_cnt, on=["i1", "kts"], how="left").join(s1c, on="i1", how="left")
             .with_columns(pl.col("num_support").fill_null(0), pl.col("devnum_support").fill_null(0),
                           (pl.col("kc").fill_null(0) - pl.col("conf").cast(pl.UInt32)).clip(0).cast(pl.Float32).alias("name_support"),
                           pl.col("s1_cluster").fill_null(0)))
        c.select(["i1", "i2"] + [pl.col(f).cast(pl.Float32) for f in S2F]).write_parquet(out_dir / f"part-{part:03d}.parquet")
    return out_dir / "part-*.parquet"


def cmd_train(a):
    t0 = time.time()
    d = RUN6 / a.tag
    refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    n_s1 = pl.read_parquet(BL6 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, a.drop_frac)
    tr = truth("train").join(kept, on="i1", how="semi")
    ctx = context_table(kept, d / "train_ctx.parquet") if a.drop_frac > 0 else None
    P = parts("train")
    oof1 = cv_stage("stage1", P, kept, ctx, tr, FEATURES_V4, None, a.n_train_entities, a.rounds, d, t0)
    oof1.write_parquet(d / "train_p1.parquet")
    s2 = stage2_features(oof1, "train", d / "train_s2")
    del oof1
    oof2 = cv_stage("stage2", P, kept, ctx, tr, FEATURES_V4 + S2F, s2, a.n_train_entities, a.rounds, d, t0)
    oof2.write_parquet(d / "train_p2.parquet")
    remap = kept.with_row_index("j")
    ctry = pl.read_parquet(CD / "train_source1.parquet").with_row_index("i1").select("i1", "country")

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        rm = k.with_row_index("j")
        return macro_f05(pairs.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         tr.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.5, 0.6, 0.7, 0.8)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.25, 1.5)])
    base = owned(oof2, None)
    res = {json.dumps(r): score(decide(base, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  stage2 {k_:<70} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(base, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "drop_frac": a.drop_frac, "model": "xgboost-cuda+translit",
                                                 "best_stage2": {"rule": json.loads(best), **res[best]}, "by_country": by}, indent=2))
    print(f"\nBEST stage2 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f}   "
          f"(v5 without transliteration: 0.95968; v4 by country India 0.94017 / US 0.97296)\ndone ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, out = RUN6 / a.tag, (a.out_dir.resolve() if a.out_dir else None)   # no out_dir with --predict-only (v11)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if out is not None and (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((d / "decision.json").read_text())["best_stage2"]["rule"]
    P = parts("test")
    kept = pl.DataFrame({"i1": pl.arange(0, pl.read_parquet(BL6 / "test_s1_ids.parquet").height, eager=True, dtype=pl.UInt32)})
    pred_path = d / "test_p2.parquet"
    if pred_path.exists():
        p2 = pl.read_parquet(pred_path)
    else:
        m1 = [xgb.Booster(model_file=str(d / f"stage1_fold{f}.ubj")) for f in (0, 1)]
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
        p2 = []
        for p in P:
            x = load_part(p, kept, None, s2)
            X = x.select(FEATURES_V4 + S2F).to_numpy()
            p2.append(x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean([predict(m, X) for m in m2], axis=0))))
        p2 = pl.concat(p2)
        p2.write_parquet(pred_path)
        print(f"[test] stage 2 scored ({time.time()-t0:.0f}s)", flush=True)
    if getattr(a, "predict_only", False):      # v11: cached predictions only, no full-list files / validator
        print(f"[test] predictions cached in {d} ({time.time()-t0:.0f}s)", flush=True)
        return
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(BL6 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL6 / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    (out / "decision.json").write_text(json.dumps({"tag": a.tag, "model": "xgboost-cuda+translit", "rule": rule}, indent=2))
    print(f"[test] {chosen.height:,} matches -> {out} ({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blocking", "features", "train", "test"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--tag", default="gpu_translit")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--n-train-entities", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "test" and not a.out_dir:
        ap.error("test needs --out-dir (a new folder)")
    {"blocking": cmd_blocking, "features": cmd_features, "train": cmd_train, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
