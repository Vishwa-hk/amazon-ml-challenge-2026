"""
Build v2 candidates for a split and save them; on train, report recall and the
oracle (best-possible-judge) F0.5 at several top-N cut-offs.

Usage:  python run_blocking_v2.py --split train [--top-n 100]
"""
import argparse
import time
from pathlib import Path

import polars as pl

from baseline import load_source
from blocking_v2 import build_candidates, prepare

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]


def oracle_table(cand: pl.DataFrame, s1_ids: pl.Series, oth_ids: pl.Series, gt: pl.DataFrame, ns) -> None:
    """Recall and oracle macro F0.5 (perfect judge keeps exactly the true candidates)."""
    s1_map = pl.DataFrame({"s1": s1_ids, "i1": pl.arange(0, len(s1_ids), eager=True, dtype=pl.UInt32)})
    o_map = pl.DataFrame({"r": oth_ids, "i2": pl.arange(0, len(oth_ids), eager=True, dtype=pl.UInt32)})
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
             .join(s1_map, on="s1").join(o_map, on="r").select("i1", "i2"))
    n_true = pairs.height
    ntrue_per = pairs.group_by("i1").len("nt")
    hit = pairs.join(cand.select("i1", "i2", "rank"), on=["i1", "i2"], how="left")
    all_s1 = s1_map.select("i1").join(ntrue_per, on="i1", how="left").with_columns(pl.col("nt").fill_null(0))
    print(f"{'top-N':>6} {'recall':>8} {'oracle F0.5':>12} {'cand pairs':>12} {'mean/S1':>8}")
    for n in ns:
        tp = hit.filter(pl.col("rank") <= n).group_by("i1").len("tp")
        t = all_s1.join(tp, on="i1", how="left").with_columns(pl.col("tp").fill_null(0))
        f = t.select(
            pl.when(pl.col("nt") == 0).then(1.0)
            .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp"))))
            .mean()
        ).item()
        k = cand.filter(pl.col("rank") <= n).height
        print(f"{n:>6} {t['tp'].sum()/n_true:>8.2%} {f:>12.4f} {k:>12,} {k/len(s1_ids):>8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "student_resource" / "dataset")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "blocking_v2")
    a = ap.parse_args()
    t0 = time.time()
    s1_raw = load_source(a.data_dir, a.cache_dir, a.split, "source1")
    oth_raw = pl.concat([load_source(a.data_dir, a.cache_dir, a.split, "source2"),
                         load_source(a.data_dir, a.cache_dir, a.split, "source3")])
    s1, oth = prepare(s1_raw), prepare(oth_raw)
    print(f"[{a.split}] prepared {s1.height:,} S1 / {oth.height:,} S2+S3 in {time.time()-t0:.0f}s", flush=True)
    cand = build_candidates(s1, oth, top_n=a.top_n)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(a.out_dir / f"{a.split}_candidates.parquet")
    pl.DataFrame({"entity_id": s1_raw["entity_id"]}).write_parquet(a.out_dir / f"{a.split}_s1_ids.parquet")
    pl.DataFrame({"entity_id": oth_raw["entity_id"]}).write_parquet(a.out_dir / f"{a.split}_oth_ids.parquet")
    print(f"[{a.split}] {cand.height:,} candidates saved ({time.time()-t0:.0f}s)", flush=True)
    if a.split == "train":
        gt = load_source(a.data_dir, a.cache_dir, "train", "ground_truth")
        oracle_table(cand, s1_raw["entity_id"], oth_raw["entity_id"], gt, [5, 10, 15, 20, 30, 50, 100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
