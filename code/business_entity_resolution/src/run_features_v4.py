"""
Compute v4 pair features for a split's short-lists v3 candidates.

Usage:  python run_features_v4.py --split train
Writes runs/v4/{split}_features/part-*.parquet (refuses to overwrite).
"""
import argparse
import time
from pathlib import Path

import polars as pl

from baseline import load_source
from blocking_v2 import prepare
from features_v4 import build_features

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--cand-file", default=None, help="candidate parquet name inside --cand-dir "
                    "(default: {split}_candidates_top100a20.parquet)")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "student_resource" / "dataset")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data_cache")
    ap.add_argument("--cand-dir", type=Path, default=PROJECT / "runs" / "blocking_v3")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "v4")
    a = ap.parse_args()
    out = a.out_dir / f"{a.split}_features"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite existing {out}")
    t0 = time.time()
    s1raw = load_source(a.data_dir, a.cache_dir, a.split, "source1")
    s2raw = load_source(a.data_dir, a.cache_dir, a.split, "source2")
    othraw = pl.concat([s2raw, load_source(a.data_dir, a.cache_dir, a.split, "source3")])
    s1p, othp = prepare(s1raw), prepare(othraw)
    cand_path = a.cand_dir / (a.cand_file or f"{a.split}_candidates_top100a20.parquet")
    print(f"[{a.split}] loaded + prepared ({time.time()-t0:.0f}s); streaming {cand_path.name}", flush=True)
    build_features(cand_path, s1p, s1raw, othp, othraw, n_s2=s2raw.height, out_dir=out, chunk=a.chunk)
    print(f"[{a.split}] done ({time.time()-t0:.0f}s) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
