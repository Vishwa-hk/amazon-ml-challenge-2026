"""
Diagnostic for v2 blocking: which candidate-generation keys would catch which
true pairs, and how many candidate pairs each key would generate (train).

Keys (always within the same country):
  K1  name keys: word-set / compact / core (generic words removed) equal
  K2  share a house number AND a name word
  K4  share a house number AND an address word (address words filtered by
      document frequency so states/cities/"road" don't create giant buckets)
  K5  share a name word AND an address word (for pairs with no number)
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from normalize_v2 import add_features_v2  # noqa: E402

C = ROOT / "data_cache"
t0 = time.time()
cols = ["i", "country", "key_ts", "key_cp", "key_core", "toks", "nums", "awords"]
s1 = add_features_v2(pl.read_parquet(C / "train_source1.parquet")).with_row_index("i") \
    .with_columns(pl.col("toks").list.unique()).select(["entity_id"] + cols)
oth = add_features_v2(pl.concat([pl.read_parquet(C / "train_source2.parquet"),
                                 pl.read_parquet(C / "train_source3.parquet")])).with_row_index("i") \
    .with_columns(pl.col("toks").list.unique()).select(["entity_id"] + cols)
print(f"features {time.time()-t0:.0f}s")

# document frequency of address words and name words, per country, over all records
def df_table(col):
    return (pl.concat([s1.select("country", col), oth.select("country", col)])
            .explode(col).drop_nulls().group_by("country", col).len("df").rename({col: "w"}))
adf, tdf = df_table("awords"), df_table("toks")

gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
      .rename({"source1_entity_id": "entity_id", "matched_entity_ids": "rid"}))
A = s1.rename({c: f"{c}1" for c in cols if c != "country"})
B = oth.rename({c: f"{c}2" for c in cols if c != "country"}).drop("country")
tp = gt.join(A, on="entity_id").join(B.rename({"entity_id": "rid"}), on="rid")
n_true = tp.height

ne = lambda c: (pl.col(f"{c}1") == pl.col(f"{c}2")) & (pl.col(f"{c}1") != "")
tp = tp.with_columns(
    (ne("key_ts") | ne("key_cp") | ne("key_core")).alias("K1"),
    (pl.col("nums1").list.set_intersection("nums2").list.len() > 0).alias("num_sh"),
    (pl.col("toks1").list.set_intersection("toks2").list.len() > 0).alias("tok_sh"),
    pl.col("awords1").list.set_intersection("awords2").alias("aw_sh"),
    pl.col("toks1").list.set_intersection("toks2").alias("tk_sh"),
).with_row_index("p")

# rarest shared address word / name word per pair (min document frequency)
def min_df(listcol, dft, name):
    return (tp.select("p", "country", listcol).explode(listcol).drop_nulls()
            .join(dft, left_on=["country", listcol], right_on=["country", "w"])
            .group_by("p").agg(pl.col("df").min().alias(name)))
tp = (tp.join(min_df("aw_sh", adf, "aw_min"), on="p", how="left")
        .join(min_df("tk_sh", tdf, "tk_min"), on="p", how="left"))

print(f"\ntrue pairs: {n_true:,}  (all share country by construction)")
print(f"K1 name keys (v2 normalization): {tp['K1'].mean():.2%}   [v1 name keys were 57.9%]")
print(f"K2 number + name word          : {(tp['num_sh'] & tp['tok_sh']).mean():.2%}")
for th in (500, 2000, 10_000, 50_000, None):
    aw = pl.col("aw_min").is_not_null() if th is None else (pl.col("aw_min") <= th)
    r = tp.select(
        (pl.col("num_sh") & aw).mean().alias("K4"),
        (pl.col("tok_sh") & aw).mean().alias("K5"),
        (pl.col("K1") | (pl.col("num_sh") & pl.col("tok_sh")) | (pl.col("num_sh") & aw)).mean().alias("U124"),
        (pl.col("K1") | (pl.col("num_sh") & pl.col("tok_sh")) | (pl.col("num_sh") & aw) | (pl.col("tok_sh") & aw))
        .mean().alias("U1245"),
    ).row(0, named=True)
    print(f"address-word df <= {str(th):>6}: K4 num+addr {r['K4']:.2%} | K5 name+addr {r['K5']:.2%} "
          f"| K1+K2+K4 {r['U124']:.2%} | K1+K2+K4+K5 {r['U1245']:.2%}")
left = tp.filter(~(pl.col("K1") | (pl.col("num_sh") & pl.col("tok_sh")) | (pl.col("num_sh") & pl.col("aw_min").is_not_null())
                   | (pl.col("tok_sh") & pl.col("aw_min").is_not_null())))
print(f"\nnot caught by any key (df unlimited): {left.height:,} ({left.height/n_true:.2%}); "
      f"of these: no number on one side {(left['nums1'].list.len().eq(0) | left['nums2'].list.len().eq(0)).mean():.1%}, "
      f"no shared address word {left['aw_min'].is_null().mean():.1%}, no shared name word {(~left['tok_sh']).mean():.1%}")

# --------------------------------------------------------------- join sizes
print("\ncandidate-pair upper bounds (sum over buckets of n_S1 x n_S2S3):")
def keyed(df, parts, filt=None):
    x = df.select(["i", "country"] + parts)
    for p in parts:
        if isinstance(x.schema[p], pl.List):
            x = x.explode(p)
    x = x.drop_nulls()
    if filt is not None:
        x = filt(x)
    return x.select("i", pl.concat_str(["country"] + parts, separator="|").hash().alias("k")).unique()
def pairs_ub(parts, filt=None):
    a = keyed(s1, parts, filt).group_by("k").len("na")
    b = keyed(oth, parts, filt).group_by("k").len("nb")
    return a.join(b, on="k").select((pl.col("na") * pl.col("nb")).sum()).item()
for k in ("key_ts", "key_cp", "key_core"):
    print(f"  K1 {k:<9}: {pairs_ub([k], lambda x, k=k: x.filter(pl.col(k) != '')):>15,}")
print(f"  K2 num+tok  : {pairs_ub(['nums', 'toks']):>15,}")
for th in (500, 2000, 10_000):
    f = lambda x, th=th: x.join(adf.filter(pl.col("df") <= th), left_on=["country", "awords"], right_on=["country", "w"], how="semi")
    print(f"  K4 num+addr (addr df<={th:>6}): {pairs_ub(['nums', 'awords'], f):>15,}")
for th in (500, 2000):
    f = lambda x, th=th: (x.join(adf.filter(pl.col("df") <= th), left_on=["country", "awords"], right_on=["country", "w"], how="semi")
                          .join(tdf.filter(pl.col("df") <= th * 10), left_on=["country", "toks"], right_on=["country", "w"], how="semi"))
    print(f"  K5 tok+addr (addr df<={th}, tok df<={th*10}): {pairs_ub(['toks', 'awords'], f):>15,}")
print(f"\ndone {time.time()-t0:.0f}s")
