"""
Compute v3 pair features for a split's v2 candidates.

Usage:  python run_features_v3.py --split train [--max-chunks 1]
"""
import argparse
import time
from pathlib import Path

import polars as pl

from baseline import load_source
from blocking_v2 import prepare
from features_v3 import build_features

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--max-chunks", type=int, default=None, help="smoke test: only the first N S1 chunks")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "student_resource" / "dataset")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data_cache")
    ap.add_argument("--cand-dir", type=Path, default=PROJECT / "runs" / "blocking_v2")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "v3")
    a = ap.parse_args()
    t0 = time.time()
    s1raw = load_source(a.data_dir, a.cache_dir, a.split, "source1")
    s2raw = load_source(a.data_dir, a.cache_dir, a.split, "source2")
    othraw = pl.concat([s2raw, load_source(a.data_dir, a.cache_dir, a.split, "source3")])
    s1p, othp = prepare(s1raw), prepare(othraw)
    cand = pl.read_parquet(a.cand_dir / f"{a.split}_candidates.parquet")
    if a.max_chunks:
        cand = cand.filter(pl.col("i1") < a.max_chunks * a.chunk)
        s1_limit = a.max_chunks * a.chunk
    print(f"[{a.split}] loaded + prepared, {cand.height:,} candidates ({time.time()-t0:.0f}s)", flush=True)
    out = a.out_dir / (f"{a.split}_features" + ("_smoke" if a.max_chunks else ""))
    if a.max_chunks:  # only build the requested chunks
        s1p, s1raw = s1p.head(s1_limit), s1raw.head(s1_limit)
    build_features(cand, s1p, s1raw, othp, othraw, n_s2=s2raw.height, out_dir=out, chunk=a.chunk)
    print(f"[{a.split}] done ({time.time()-t0:.0f}s) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
