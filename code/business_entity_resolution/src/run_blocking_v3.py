"""
Build v3 candidates for a split and save them to runs/blocking_v3/; on train,
report recall and the oracle (best-possible-judge) F0.5 for several cut-offs,
with and without the address-quota slots.

Usage:  python run_blocking_v3.py --split train
"""
import argparse
import time
from pathlib import Path

import polars as pl

from baseline import load_source
from blocking_v3 import build_candidates, prepare

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]


def oracle(cand: pl.DataFrame, pairs: pl.DataFrame, n_s1: int) -> tuple:
    hit = pairs.join(cand.select("i1", "i2"), on=["i1", "i2"], how="semi").group_by("i1").len("tp")
    nt = pairs.group_by("i1").len("nt")
    t = (pl.DataFrame({"i1": pl.arange(0, n_s1, eager=True, dtype=pl.UInt32)})
         .join(nt, on="i1", how="left").join(hit, on="i1", how="left").fill_null(0))
    f = t.select(pl.when(pl.col("nt") == 0).then(1.0)
                 .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")))).mean()).item()
    return t["tp"].sum() / pairs.height, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--addr-n", type=int, default=20)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "student_resource" / "dataset")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "blocking_v3")
    a = ap.parse_args()
    out = a.out_dir / f"{a.split}_candidates.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    s1_raw = load_source(a.data_dir, a.cache_dir, a.split, "source1")
    oth_raw = pl.concat([load_source(a.data_dir, a.cache_dir, a.split, "source2"),
                         load_source(a.data_dir, a.cache_dir, a.split, "source3")])
    s1, oth = prepare(s1_raw), prepare(oth_raw)
    print(f"[{a.split}] prepared {s1.height:,} S1 / {oth.height:,} S2+S3 in {time.time()-t0:.0f}s", flush=True)
    cand = build_candidates(s1, oth, top_n=a.top_n, addr_n=a.addr_n)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(out)
    pl.DataFrame({"entity_id": s1_raw["entity_id"]}).write_parquet(a.out_dir / f"{a.split}_s1_ids.parquet")
    pl.DataFrame({"entity_id": oth_raw["entity_id"]}).write_parquet(a.out_dir / f"{a.split}_oth_ids.parquet")
    print(f"[{a.split}] {cand.height:,} candidates saved ({time.time()-t0:.0f}s)", flush=True)
    if a.split == "train":
        s1_map = pl.DataFrame({"s1": s1_raw["entity_id"], "i1": pl.arange(0, s1_raw.height, eager=True, dtype=pl.UInt32)})
        o_map = pl.DataFrame({"r": oth_raw["entity_id"], "i2": pl.arange(0, oth_raw.height, eager=True, dtype=pl.UInt32)})
        gt = load_source(a.data_dir, a.cache_dir, "train", "ground_truth")
        pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
                 .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
                 .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
                 .join(s1_map, on="s1").join(o_map, on="r").select("i1", "i2"))
        print(f"\n{'candidate set':<34} {'recall':>8} {'oracle F0.5':>12} {'pairs':>13} {'per S1':>7}")
        for label, e in (
            ("top-100 by score only", pl.col("rank") <= 100),
            ("top-50  + top-20 by address", (pl.col("rank") <= 50) | (pl.col("arank") <= a.addr_n)),
            ("top-100 + top-20 by address", (pl.col("rank") <= 100) | (pl.col("arank") <= a.addr_n)),
            (f"top-{a.top_n} + top-{a.addr_n} by address (all)", pl.lit(True)),
        ):
            c = cand.filter(e)
            r, f = oracle(c, pairs, s1_raw.height)
            print(f"{label:<34} {r:>8.2%} {f:>12.4f} {c.height:>13,} {c.height/s1_raw.height:>7.1f}")
        print("   (v2 top-100: recall 92.16%, oracle 0.9695, 115.2M pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
