"""
v7 = v6 + twin-decoy features (features_v7.NEW7: signed house-number offsets and a
decoy-word vocabulary). Same short-lists (runs/blocking_v6), same two-stage GPU judge.

v6 code is reused unchanged: its module globals (run folder, feature list) are pointed
at runs/v7 and FEATURES_V4 + NEW7, then its train/test commands run as before.

The decoy-word vocabulary is learned ONLY from the train S1 entities that the
test-like universe drops (19%) — those entities are never trained on or scored, so
the held-out estimate is not leaked.

Commands:
  python pipeline_v7.py vocab
  python pipeline_v7.py features --split train|test   # v6 parts + NEW7 -> runs/v7/<split>_features
  python pipeline_v7.py train --tag <tag>
  python pipeline_v7.py test --tag <tag> --out-dir <new folder> [--rule '<json>']
  add --variant v7b to any of features/train/test for the +21-offset variant (runs/v7b)
  --variant v7c: v7b + French word translation (FR_MAP); test features only, models copied from runs/v7b/<tag>
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

import pipeline_v6 as v6
from features_v4 import FEATURES_V4
from features_v7 import DECOY_OFFSETS, DECOY_OFFSETS_B, NEW7, build_vocab, pair_features, record_numbers
from judge_v4 import kept_entities

PROJECT = Path(__file__).resolve().parents[3]
RUN7 = PROJECT / "runs" / "v7"
VOCAB = RUN7 / "decoy_vocab.parquet"
FEATURES_V7 = FEATURES_V4 + NEW7
VARIANTS = {"v7": (RUN7, DECOY_OFFSETS), "v7b": (PROJECT / "runs" / "v7b", DECOY_OFFSETS_B),  # v7b = +21 offset
            "v7c": (PROJECT / "runs" / "v7c", DECOY_OFFSETS_B)}  # v7c = v7b + FR_MAP (test only; model copied from v7b)


def tables(split: str, fr_map: bool = False):
    s1raw, _, oth_t, _ = v6.load_raw(split)
    return record_numbers(s1raw, "i1", fr_map), record_numbers(oth_t, "i2", fr_map)


def cmd_vocab(a):
    v6.refuse(VOCAB)
    RUN7.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S, O = tables("train")
    n = S.height
    dropped = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)}).join(kept_entities(n, a.drop_frac), on="i1", how="anti")
    tr = v6.truth("train").with_columns(pl.lit(True).alias("y"))
    c = (pl.scan_parquet(v6.BL6 / "train_candidates_top100a20.parquet").select("i1", "i2", "rank", "score")
         .filter((pl.col("rank") <= 10) & (pl.col("score") >= 2.0)).join(dropped.lazy(), on="i1", how="semi").collect())
    c = c.join(tr, on=["i1", "i2"], how="left").with_columns(pl.col("y").fill_null(False)).select("i1", "i2", "y")
    v = build_vocab(c, S, O)
    v.write_parquet(VOCAB)
    print(f"vocab from {dropped.height:,} dropped entities, {c.height:,} pairs ({int(c['y'].sum()):,} true): {v.height:,} words ({time.time()-t0:.0f}s)")
    with pl.Config(tbl_rows=40):
        print(v.head(30)); print(v.tail(15))


def cmd_features(a):
    run, offsets = VARIANTS[a.variant]
    out = run / f"{a.split}_features"
    v6.refuse(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S, O = tables(a.split, fr_map=(a.variant == "v7c"))
    vocab = pl.read_parquet(VOCAB)
    src = sorted((v6.RUN6 / f"{a.split}_features").glob("part-*.parquet"))
    for k, p in enumerate(src):
        x = pl.read_parquet(p)
        lo, hi = x["i1"].min(), x["i1"].max()
        f = pair_features(x.select("i1", "i2"), S.filter(pl.col("i1").is_between(lo, hi)), O, vocab, offsets)
        y = x.join(f, on=["i1", "i2"], how="left")
        assert y.height == x.height
        y.write_parquet(out / p.name)
        if k % 8 == 0:
            print(f"  {a.split} part {k+1}/{len(src)} | {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.split}] v7 features done ({time.time()-t0:.0f}s) -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vocab", "features", "train", "test"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--tag", default="gpu_decoy")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--n-train-entities", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--rule")
    ap.add_argument("--variant", choices=list(VARIANTS), default="v7")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "vocab":
        return cmd_vocab(a) or 0
    if a.cmd == "features":
        return cmd_features(a) or 0
    if a.cmd == "test" and not a.out_dir:
        ap.error("test needs --out-dir (a new folder)")
    v6.RUN6, v6.FEATURES_V4 = VARIANTS[a.variant][0], FEATURES_V7  # reuse v6 train/test on the v7 feature parts
    {"train": v6.cmd_train, "test": v6.cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
