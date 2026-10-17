"""
Diagnostic for short-lists v3 (train): which new keys would catch the true pairs
that v2 (top-100) still misses, and how many candidate pairs each would create.

New keys use each record's RAREST words (by document frequency within country)
instead of a global frequency cap, so no record loses coverage:
  NT2  house number + one of each side's 2 rarest name words
  NA3  house number + one of each side's 3 rarest address words
  TA   one of the 2 rarest name words + one of the 3 rarest address words
  AA   two shared words among each side's 3 rarest address words (no number needed)
  TT   two shared words among each side's 3 rarest name words (no address needed)
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from blocking_v2 import _freq, prepare  # noqa: E402

C, BL = ROOT / "data_cache", ROOT / "runs" / "blocking_v2"
LATIN_ONLY = r"^[\p{Latin}\p{Common}\p{Inherited}]*$"
t0 = time.time()
s1raw = pl.read_parquet(C / "train_source1.parquet")
othraw = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")])
s1, oth = prepare(s1raw), prepare(othraw)
tf, af = _freq(s1, oth, "toks"), _freq(s1, oth, "awords")


def rarest(df, col, freq, k, name):
    x = (df.select("i", "country", col).explode(col).drop_nulls()
         .join(freq, on=["country", col]).sort(["i", "df", col])
         .group_by("i", maintain_order=True).agg(pl.col(col).head(k).alias(name)))
    return df.select("i").join(x, on="i", how="left").with_columns(pl.col(name).fill_null([]))


def add_rare(df):
    for col, fq, k, nm in (("toks", tf, 2, "rt2"), ("toks", tf, 3, "rt3"), ("awords", af, 3, "ra3")):
        df = df.join(rarest(df, col, fq, k, nm), on="i")
    return df


s1, oth = add_rare(s1), add_rare(oth)
print(f"rarest-word lists built ({time.time()-t0:.0f}s)")

# true pairs v2 missed
s1_map = pl.DataFrame({"s1": s1raw["entity_id"], "i1": pl.arange(0, s1raw.height, eager=True, dtype=pl.UInt32)})
o_map = pl.DataFrame({"r": othraw["entity_id"], "i2": pl.arange(0, othraw.height, eager=True, dtype=pl.UInt32)})
gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
      .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"}).join(s1_map, on="s1").join(o_map, on="r").select("i1", "i2"))
cand = pl.read_parquet(BL / "train_candidates.parquet", columns=["i1", "i2"])
miss = gt.join(cand, on=["i1", "i2"], how="anti")
print(f"true pairs {gt.height:,} | v2 top-100 misses {miss.height:,} ({miss.height/gt.height:.2%})")

A = s1.select(pl.col("i").alias("i1"), pl.col("nums").alias("n1"), pl.col("rt2").alias("rt2_1"), pl.col("rt3").alias("rt3_1"),
              pl.col("ra3").alias("ra3_1"), pl.col("toks").alias("t1"), pl.col("awords").alias("a1"))
B = oth.select(pl.col("i").alias("i2"), pl.col("nums").alias("n2"), pl.col("rt2").alias("rt2_2"), pl.col("rt3").alias("rt3_2"),
               pl.col("ra3").alias("ra3_2"), pl.col("toks").alias("t2"), pl.col("awords").alias("a2"))
m = miss.join(A, on="i1").join(B, on="i2").join(
    othraw.with_row_index("i2").select("i2", (~pl.col("business_name").fill_null("").str.contains(LATIN_ONLY)).alias("native"),
                                       (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("no_addr")), on="i2")
inter = lambda a, b: pl.col(a).list.set_intersection(b).list.len()
m = m.with_columns(
    (inter("n1", "n2") > 0).alias("num"),
    (inter("rt2_1", "rt2_2") > 0).alias("rt2"),
    (inter("ra3_1", "ra3_2") > 0).alias("ra3"),
    (inter("ra3_1", "ra3_2") >= 2).alias("aa"),
    (inter("rt3_1", "rt3_2") >= 2).alias("tt"),
    (inter("t1", "t2") > 0).alias("any_tok"),
    (inter("a1", "a2") > 0).alias("any_aw"),
    ((pl.col("n1").list.len() == 0) | (pl.col("n2").list.len() == 0)).alias("one_no_num"),
)
m = m.with_columns(
    (pl.col("num") & pl.col("rt2")).alias("NT2"), (pl.col("num") & pl.col("ra3")).alias("NA3"),
    (pl.col("rt2") & pl.col("ra3")).alias("TA"), pl.col("aa").alias("AA"), pl.col("tt").alias("TT"))
m = m.with_columns(pl.any_horizontal("NT2", "NA3", "TA", "AA", "TT").alias("ANY"))
n = m.height
print("\nwhat the v2 misses are (share of misses):")
for c in ("native", "no_addr", "one_no_num", "num", "any_tok", "any_aw"):
    print(f"   {c:<12} {m[c].mean():6.1%}")
print("\nshare of v2 misses each new key would catch (before any top-N cut):")
for k in ("NT2", "NA3", "TA", "AA", "TT", "ANY"):
    print(f"   {k:<4} {m[k].mean():6.1%}   | native-script misses caught: {m.filter(pl.col('native'))[k].mean():6.1%}")
left = m.filter(~pl.col("ANY"))
print(f"\nstill uncaught: {left.height:,} ({left.height/gt.height:.2%} of all true pairs) — native {left['native'].mean():.1%}, "
      f"no address {left['no_addr'].mean():.1%}, share any name word {left['any_tok'].mean():.1%}, share any addr word {left['any_aw'].mean():.1%}")


def keyed(df, parts):
    x = df.select(["i", "country"] + parts)
    for p in parts:
        x = x.explode(p)
    return x.drop_nulls().select("i", pl.concat_str(["country"] + parts, separator="|").hash().alias("k")).unique()


def pair_list(col):  # all 2-word combinations of a (<=3)-word list, as "a|b" with a<b
    s = pl.col(col).list.sort()
    return pl.concat_list([
        pl.concat_str([s.list.get(0, null_on_oob=True), s.list.get(1, null_on_oob=True)], separator="|"),
        pl.concat_str([s.list.get(0, null_on_oob=True), s.list.get(2, null_on_oob=True)], separator="|"),
        pl.concat_str([s.list.get(1, null_on_oob=True), s.list.get(2, null_on_oob=True)], separator="|"),
    ]).list.drop_nulls()


s1 = s1.with_columns(pair_list("ra3").alias("ap"), pair_list("rt3").alias("tp"))
oth = oth.with_columns(pair_list("ra3").alias("ap"), pair_list("rt3").alias("tp"))
print("\ncandidate-pair volume (sum over buckets of n_S1 x n_S2S3), and share from buckets > 10k pairs:")
for name, parts in (("NT2", ["nums", "rt2"]), ("NA3", ["nums", "ra3"]), ("TA", ["rt2", "ra3"]), ("AA", ["ap"]), ("TT", ["tp"])):
    a = keyed(s1, parts).group_by("k").len("na")
    b = keyed(oth, parts).group_by("k").len("nb")
    j = a.join(b, on="k").with_columns((pl.col("na") * pl.col("nb")).alias("prod"))
    tot = j["prod"].sum()
    big = j.filter(pl.col("prod") > 10_000)["prod"].sum()
    print(f"   {name:<4} {tot:>15,} | from buckets > 10k: {big/tot:6.1%} ({j.filter(pl.col('prod') > 10_000).height:,} buckets)")
print(f"\ndone ({time.time()-t0:.0f}s)")
