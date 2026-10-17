"""
Short-lists v8 = short-lists v3 (all keys, top-150 by cheap score + top-20 by address score)
+ a RECORD-SIDE NAME SLOT for records without an address.

Why (27 Sep, v7b error analysis): the biggest remaining loss is true pairs missing from
the short-lists (0.0074 of F0.5), and 55% of those are records with NO address. Such a
record scores at most 2.0 (name only), so for its S1 entity it ranks below 100-150
address-bearing candidates and is cut — although, seen from the record, the true S1 is
usually its #1 candidate. Only 70% of no-address true pairs survived the short-lists.

The slot: for each no-address record, keep its top NSLOT S1 candidates by name overlap
(tok_jac > 0), computed within each S1 chunk (a superset of the global top NSLOT).
Column `nrank` = that record-side rank (null for other pairs). Everything else is
blocking_v3 unchanged.
"""
import time

import polars as pl

from blocking_v2 import _jac, build_key_tables
from blocking_v3 import CAP, FLAGS3, _freq, add_rarest, new_key_tables

NSLOT = 5


def build_candidates(s1: pl.DataFrame, oth: pl.DataFrame, top_n: int = 150, addr_n: int = 20, nslot: int = NSLOT,
                     chunk: int = 100_000, cap: int = CAP, verbose: bool = True, parts_dir=None) -> pl.DataFrame | None:
    """Candidate pairs (i1, i2, keys, tok_jac, addr_jac, num_sh, score, rank, arank, nrank).
    With parts_dir, each S1 chunk is written to parts_dir/part-XXX.parquet (nothing accumulates in RAM) and None
    is returned — on train the accumulated set (~290M rows) does not fit next to the key tables in 31 GB."""
    t0 = time.time()
    tf, af = _freq(s1, oth, "toks"), _freq(s1, oth, "awords")
    s1, oth = add_rarest(s1, tf, af), add_rarest(oth, tf, af)
    tables = build_key_tables(s1, oth)
    tables.update(new_key_tables(s1, oth, cap))
    if verbose:
        sizes = ", ".join(f"{k}: {a.height/1e6:.1f}M/{b.height/1e6:.1f}M" for k, (a, b) in tables.items())
        print(f"  key tables ({sizes}) in {time.time()-t0:.0f}s", flush=True)
    S = s1.select(pl.col("i").alias("i1"), pl.col("th").alias("th1"), pl.col("ah").alias("ah1"), pl.col("nums").alias("n1"))
    O = oth.select(pl.col("i").alias("i2"), pl.col("th").alias("th2"), pl.col("ah").alias("ah2"), pl.col("nums").alias("n2"),
                   ((pl.col("ah").list.len() == 0) & (pl.col("nums").list.len() == 0)).alias("noaddr"))
    if verbose:
        print(f"  no-address records: {int(O['noaddr'].sum()):,} of {O.height:,}", flush=True)
    # chunks by S1 index RANGE (not position), so a one-country subset with global indices also works (v11)
    out, raw_total, lo, n = [], 0, int(s1["i"].min()), int(s1["i"].max()) + 1
    for start in range(lo - lo % chunk, n, chunk):
        stop = min(start + chunk, n)
        parts = []
        for name, (a, b) in tables.items():
            ac = a.filter((pl.col("i") >= start) & (pl.col("i") < stop))
            parts.append(ac.join(b, on="k").select(pl.col("i").alias("i1"), pl.col("i_right").alias("i2"),
                                                   pl.lit(FLAGS3[name], dtype=pl.UInt16).alias("f")))
        raw = pl.concat(parts)
        raw_total += raw.height
        c = raw.group_by("i1", "i2").agg(pl.col("f").unique().sum().cast(pl.UInt16).alias("keys"))
        c = c.join(S.filter((pl.col("i1") >= start) & (pl.col("i1") < stop)), on="i1").join(O, on="i2")
        c = c.with_columns(
            _jac("th1", "th2").alias("tok_jac"),
            _jac("ah1", "ah2").alias("addr_jac"),
            (pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("num_sh"),
        ).drop("th1", "th2", "ah1", "ah2", "n1", "n2")
        c = c.with_columns(
            (2 * pl.col("tok_jac") + 2 * pl.col("addr_jac") + pl.col("num_sh").cast(pl.Float64)).alias("score"),
            (2 * pl.col("addr_jac") + pl.col("num_sh").cast(pl.Float64)).alias("ascore"),
        )
        c = c.with_columns(
            pl.col("score").rank("ordinal", descending=True).over("i1").cast(pl.UInt32).alias("rank"),
            pl.col("ascore").rank("ordinal", descending=True).over("i1").cast(pl.UInt32).alias("arank"),
        ).drop("ascore")
        # record-side slot, ranked on the (small) no-address subset only — keeps the chunk's memory peak down
        slot = (c.filter(pl.col("noaddr") & (pl.col("tok_jac") > 0)).select("i1", "i2", "tok_jac")
                .with_columns(pl.col("tok_jac").rank("ordinal", descending=True).over("i2").cast(pl.UInt32).alias("nrank"))
                .filter(pl.col("nrank") <= nslot).select("i1", "i2", "nrank"))
        c = c.drop("noaddr").join(slot, on=["i1", "i2"], how="left")
        kept = c.filter((pl.col("rank") <= top_n) | (pl.col("arank") <= addr_n) | pl.col("nrank").is_not_null())
        del c, raw, parts, slot
        if parts_dir is not None:
            kept.write_parquet(parts_dir / f"part-{start // chunk:03d}.parquet")
            del kept
        else:
            out.append(kept)
        if verbose:
            print(f"  S1 {stop:>9,}/{n:,} | raw key matches so far {raw_total/1e6:,.0f}M | {time.time()-t0:.0f}s", flush=True)
    return pl.concat(out) if parts_dir is None else None
