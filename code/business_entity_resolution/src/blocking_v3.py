"""
v3 candidate generation = all v2 keys + five "rarest-word" keys, within country.

v2 keys (flags 1-32, see blocking_v2.py) keep their global frequency limits.
New keys use each record's RAREST words (document frequency within country) so
records whose words are all common still get keys (v2's global caps dropped
them — e.g. "Dynamic Brothers Prive Limited, D 11"):
     64  NT2  house number + one of each side's 2 rarest name words
    128  NA3  house number + one of each side's 3 rarest address words
    256  TA2  one of the 2 rarest name words + one of the 3 rarest address words
    512  AA   two words shared among each side's 3 rarest address words (no number needed)
   1024  TT   two words shared among each side's 3 rarest name words (no address needed)
For the new keys only, buckets that would create more than CAP pairs are skipped
(a few thousand buckets hold ~90% of the raw volume).

Per S1 entity we keep the top_n candidates by the cheap score PLUS the top addr_n
by an address-only score (2*addr Jaccard + shared number): native-script and
brand-name matches score 0 on name overlap and would otherwise be crowded out.
"""
import time

import polars as pl

from blocking_v2 import FLAGS, _freq, _jac, _keyed, build_key_tables, prepare  # noqa: F401

FLAGS3 = {**FLAGS, "NT2": 64, "NA3": 128, "TA2": 256, "AA": 512, "TT": 1024}
CAP = 10_000


def _rarest(df: pl.DataFrame, col: str, freq: pl.DataFrame, k: int, name: str) -> pl.DataFrame:
    x = (df.select("i", "country", col).explode(col).drop_nulls()
         .join(freq, on=["country", col]).sort(["i", "df", col])
         .group_by("i", maintain_order=True).agg(pl.col(col).head(k).alias(name)))
    return df.select("i").join(x, on="i", how="left").with_columns(pl.col(name).fill_null([]))


def _pairs(col: str) -> pl.Expr:
    """All 2-word combinations of a (<=3)-word list as "a|b" (a < b)."""
    s = pl.col(col).list.sort()
    g = lambda j: s.list.get(j, null_on_oob=True)
    return pl.concat_list([pl.concat_str([g(0), g(1)], separator="|"), pl.concat_str([g(0), g(2)], separator="|"),
                           pl.concat_str([g(1), g(2)], separator="|")]).list.drop_nulls()


def add_rarest(df: pl.DataFrame, tf: pl.DataFrame, af: pl.DataFrame) -> pl.DataFrame:
    for col, fq, k, nm in (("toks", tf, 2, "rt2"), ("toks", tf, 3, "rt3"), ("awords", af, 3, "ra3")):
        df = df.join(_rarest(df, col, fq, k, nm), on="i")
    return df.with_columns(_pairs("ra3").alias("ap"), _pairs("rt3").alias("tp"))


def new_key_tables(s1: pl.DataFrame, oth: pl.DataFrame, cap: int = CAP) -> dict:
    specs = {"NT2": ["nums", "rt2"], "NA3": ["nums", "ra3"], "TA2": ["rt2", "ra3"], "AA": ["ap"], "TT": ["tp"]}
    out = {}
    for name, parts in specs.items():
        a, b = _keyed(s1, parts), _keyed(oth, parts)
        ok = (a.group_by("k").len("na").join(b.group_by("k").len("nb"), on="k")
              .filter(pl.col("na") * pl.col("nb") <= cap).select("k"))
        out[name] = (a.join(ok, on="k", how="semi"), b.join(ok, on="k", how="semi"))
    return out


def build_candidates(s1: pl.DataFrame, oth: pl.DataFrame, top_n: int = 150, addr_n: int = 20,
                     chunk: int = 100_000, cap: int = CAP, verbose: bool = True) -> pl.DataFrame:
    """Candidate pairs (i1, i2, keys, tok_jac, addr_jac, num_sh, score, rank, arank)."""
    t0 = time.time()
    tf, af = _freq(s1, oth, "toks"), _freq(s1, oth, "awords")
    s1, oth = add_rarest(s1, tf, af), add_rarest(oth, tf, af)
    tables = build_key_tables(s1, oth)
    tables.update(new_key_tables(s1, oth, cap))
    if verbose:
        sizes = ", ".join(f"{k}: {a.height/1e6:.1f}M/{b.height/1e6:.1f}M" for k, (a, b) in tables.items())
        print(f"  key tables ({sizes}) in {time.time()-t0:.0f}s", flush=True)
    S = s1.select(pl.col("i").alias("i1"), pl.col("th").alias("th1"), pl.col("ah").alias("ah1"), pl.col("nums").alias("n1"))
    O = oth.select(pl.col("i").alias("i2"), pl.col("th").alias("th2"), pl.col("ah").alias("ah2"), pl.col("nums").alias("n2"))
    out, raw_total, n = [], 0, s1.height
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        parts = []
        for name, (a, b) in tables.items():
            ac = a.filter((pl.col("i") >= start) & (pl.col("i") < stop))
            parts.append(ac.join(b, on="k").select(pl.col("i").alias("i1"), pl.col("i_right").alias("i2"),
                                                   pl.lit(FLAGS3[name], dtype=pl.UInt16).alias("f")))
        raw = pl.concat(parts)
        raw_total += raw.height
        c = raw.group_by("i1", "i2").agg(pl.col("f").unique().sum().cast(pl.UInt16).alias("keys"))
        c = c.join(S.slice(start, stop - start), on="i1").join(O, on="i2")
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
        out.append(c.filter((pl.col("rank") <= top_n) | (pl.col("arank") <= addr_n)))
        if verbose:
            print(f"  S1 {stop:>9,}/{n:,} | raw key matches so far {raw_total/1e6:,.0f}M | {time.time()-t0:.0f}s", flush=True)
    return pl.concat(out)
