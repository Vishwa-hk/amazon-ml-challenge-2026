"""
Diagnostic: recall of concrete v2 blocking designs under a pair budget (train).
Same keys as probe_v2_keys.py, but every key carries a document-frequency limit
so the candidate volume stays manageable, and recall is measured for the UNION.
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
prep = lambda df: add_features_v2(df).with_row_index("i").with_columns(pl.col("toks").list.unique()).select(["entity_id"] + cols)
s1 = prep(pl.read_parquet(C / "train_source1.parquet"))
oth = prep(pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")]))

def df_table(col):
    return (pl.concat([s1.select("country", col), oth.select("country", col)])
            .explode(col).drop_nulls().group_by("country", col).len("df").rename({col: "w"}))
adf, tdf = df_table("awords"), df_table("toks")
kdf = {k: pl.concat([s1.select("country", k), oth.select("country", k)]).group_by("country", k).len("df")
       for k in ("key_ts", "key_cp", "key_core")}

gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
      .rename({"source1_entity_id": "entity_id", "matched_entity_ids": "rid"}))
A = s1.rename({c: f"{c}1" for c in cols if c != "country"})
B = oth.rename({c: f"{c}2" for c in cols if c != "country"}).drop("country").rename({"entity_id": "rid"})
tp = gt.join(A, on="entity_id").join(B, on="rid").with_row_index("p")
ne = lambda c: (pl.col(f"{c}1") == pl.col(f"{c}2")) & (pl.col(f"{c}1") != "")
tp = tp.with_columns(
    ne("key_ts").alias("ts"), ne("key_cp").alias("cp"), ne("key_core").alias("core"),
    (pl.col("nums1").list.set_intersection("nums2").list.len() > 0).alias("num_sh"),
    pl.col("awords1").list.set_intersection("awords2").alias("aw_sh"),
    pl.col("toks1").list.set_intersection("toks2").alias("tk_sh"),
)
def min_df(listcol, dft, name):
    return (tp.select("p", "country", listcol).explode(listcol).drop_nulls()
            .join(dft, left_on=["country", listcol], right_on=["country", "w"])
            .group_by("p").agg(pl.col("df").min().alias(name)))
tp = tp.join(min_df("aw_sh", adf, "aw"), on="p", how="left").join(min_df("tk_sh", tdf, "tk"), on="p", how="left")
tp = tp.join(kdf["key_core"].rename({"key_core": "key_core1", "df": "core_df"}), on=["country", "key_core1"], how="left")
BIG = 10**12
tp = tp.with_columns(pl.col("aw").fill_null(BIG), pl.col("tk").fill_null(BIG))
n = tp.height
print(f"setup {time.time()-t0:.0f}s | true pairs {n:,}")

def keyed(df, parts, filt=None):
    x = df.select(["i", "country"] + parts)
    for p in parts:
        if isinstance(x.schema[p], pl.List):
            x = x.explode(p)
    x = x.drop_nulls()
    if filt is not None:
        x = filt(x)
    return x.select("i", pl.concat_str(["country"] + parts, separator="|").hash().alias("k")).unique()
def ub(parts, filt=None):
    a = keyed(s1, parts, filt).group_by("k").len("na")
    b = keyed(oth, parts, filt).group_by("k").len("nb")
    return a.join(b, on="k").select((pl.col("na") * pl.col("nb")).sum()).item()
semi = lambda tbl, col, th: (lambda x: x.join(tbl.filter(pl.col("df") <= th), left_on=["country", col], right_on=["country", "w"], how="semi"))

print("\nK2 (house number + name word), name-word frequency limit:")
for T in (200, 1000, 5000, 20000):
    print(f"  tok df <= {T:>6}: recall {(pl.Series(tp['num_sh'] & (tp['tk'] <= T))).mean():.2%} | pairs <= {ub(['nums', 'toks'], semi(tdf, 'toks', T)):,}")
core_sizes = {T: ub(["key_core"], lambda x, T=T: x.filter(pl.col("key_core") != "").join(
    kdf["key_core"].filter(pl.col("df") <= T), on=["country", "key_core"], how="semi")) for T in (100, 1000)}
print("K1 core key with frequency limit: " + ", ".join(
    f"df<={T}: recall {(tp['core'] & (tp['core_df'] <= T)).mean():.2%}, pairs <= {v:,}" for T, v in core_sizes.items()))

designs = {
    "D1  names(ts,cp) + num&addr(2000)":
        pl.col("ts") | pl.col("cp") | (pl.col("num_sh") & (pl.col("aw") <= 2000)),
    "D2  D1 + num&tok(1000)":
        pl.col("ts") | pl.col("cp") | (pl.col("num_sh") & (pl.col("aw") <= 2000)) | (pl.col("num_sh") & (pl.col("tk") <= 1000)),
    "D3  D2 + tok&addr(2000/20000)":
        pl.col("ts") | pl.col("cp") | (pl.col("num_sh") & (pl.col("aw") <= 2000)) | (pl.col("num_sh") & (pl.col("tk") <= 1000))
        | ((pl.col("aw") <= 2000) & (pl.col("tk") <= 20000)),
    "D4  D3 + core(1000) + num&tok(5000)":
        pl.col("ts") | pl.col("cp") | (pl.col("core") & (pl.col("core_df") <= 1000)) | (pl.col("num_sh") & (pl.col("aw") <= 2000))
        | (pl.col("num_sh") & (pl.col("tk") <= 5000)) | ((pl.col("aw") <= 2000) & (pl.col("tk") <= 20000)),
    "D5  D4 but num&addr(10000)":
        pl.col("ts") | pl.col("cp") | (pl.col("core") & (pl.col("core_df") <= 1000)) | (pl.col("num_sh") & (pl.col("aw") <= 10000))
        | (pl.col("num_sh") & (pl.col("tk") <= 5000)) | ((pl.col("aw") <= 2000) & (pl.col("tk") <= 20000)),
}
print("\nunion recall of concrete designs:")
for name, e in designs.items():
    print(f"  {name:<40} {tp.select(e.mean()).item():.2%}")
print(f"\ndone {time.time()-t0:.0f}s")
