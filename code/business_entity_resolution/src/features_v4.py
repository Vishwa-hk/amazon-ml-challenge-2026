"""
v4 pair features = the 35 v3 features + which v3 rarest-word keys fired + the
candidate's rank by the address-only score (the address-quota slot).

Computed over short-lists v3 (runs/blocking_v3). Country is still NOT a feature.
"""
import time

import polars as pl

from blocking_v2 import _freq
from blocking_v3 import FLAGS3
from features_v3 import FEATURES, add_context, chunk_features, record_table

NEW = ["k_nt2", "k_na3", "k_ta2", "k_aa", "k_tt", "arank"]
FEATURES_V4 = FEATURES + NEW


def build_features(cand_path, s1p: pl.DataFrame, s1raw: pl.DataFrame, othp: pl.DataFrame,
                   othraw: pl.DataFrame, n_s2: int, out_dir, chunk: int = 100_000) -> None:
    """Write feature parquet parts (one per S1 chunk). Keeps `score` for recomputing context.

    Streams the candidate file chunk by chunk (it is ~290M rows on train). Only the
    record-competition context, which needs every candidate of a record, is
    computed over the full set — from a slim (i1, i2, score) projection.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = out_dir.parent / f"{out_dir.name}_context.parquet"
    # 1. record-competition context FIRST, from a slim projection, straight to disk —
    #    before the big text tables exist (both together exhausted 31 GB on 291M pairs)
    slim = pl.scan_parquet(cand_path).select("i1", "i2", pl.col("score").cast(pl.Float32)).collect()
    slim.select(
        "i1", "i2",
        (pl.col("score").max().over("i2") - pl.col("score")).alias("gap_r"),
        pl.col("score").rank("min", descending=True).over("i2").cast(pl.UInt32).alias("rank_r"),
        pl.len().over("i2").cast(pl.UInt32).alias("r_cands"),
    ).write_parquet(ctx_path, row_group_size=2_000_000)
    n_pairs = slim.height
    del slim
    print(f"  record-competition context over {n_pairs:,} candidates -> {ctx_path.name} ({time.time()-t0:.0f}s)", flush=True)
    # 2. text tables, then 3. features chunk by chunk (both inputs streamed by S1 range)
    tok_f, aw_f = _freq(s1p, othp, "toks"), _freq(s1p, othp, "awords")
    S = record_table(s1p, s1raw)
    O = record_table(othp, othraw, n_s2=n_s2)
    n = s1p.height
    for part, start in enumerate(range(0, n, chunk)):
        in_chunk = (pl.col("i1") >= start) & (pl.col("i1") < start + chunk)
        c = (pl.scan_parquet(cand_path).filter(in_chunk).collect()
             .join(pl.scan_parquet(ctx_path).filter(in_chunk).collect(), on=["i1", "i2"])
             .with_columns((pl.col("score").max().over("i1") - pl.col("score")).alias("gap_s1")))
        f = chunk_features(c, S, O, tok_f, aw_f)
        k = pl.col("keys")
        extra = c.select(
            "i1", "i2",
            *[((k & FLAGS3[n_]) > 0).cast(pl.Float32).alias(f"k_{n_.lower()}") for n_ in ("NT2", "NA3", "TA2", "AA", "TT")],
            pl.col("arank").cast(pl.Float32),
        )
        f.join(extra, on=["i1", "i2"]).write_parquet(out_dir / f"part-{part:03d}.parquet")
        print(f"  features: S1 {min(start+chunk, n):>9,}/{n:,} | {c.height/1e6:.1f}M pairs | {time.time()-t0:.0f}s", flush=True)
