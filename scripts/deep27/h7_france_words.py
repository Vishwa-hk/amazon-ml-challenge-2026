# Label-free: words ADDED in near-certain decoy pairs (offset in {3,4,5,7,9,11,13}) vs same-number pairs, by country (TEST)
import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from translit import latin_tokens
C="data_cache/"
s1=pl.read_parquet(C+"test_source1.parquet").with_row_index("i1")
oth=pl.concat([pl.read_parquet(C+"test_source2.parquet"),pl.read_parquet(C+"test_source3.parquet")]).with_row_index("i2")
cand=(pl.scan_parquet("runs/blocking_v6/test_candidates_top100a20.parquet").select("i1","i2","rank","score")
      .filter((pl.col("rank")<=5)&(pl.col("score")>=2.5)).collect())
first=lambda c: pl.col(c).fill_null("").str.extract(r"(\d+)").cast(pl.Int64,strict=False)
S=s1.select("i1","country",first("business_address").alias("p1"),latin_tokens("business_name").list.unique().alias("t1"))
O=oth.select("i2",first("business_address").alias("p2"),latin_tokens("business_name").list.unique().alias("t2"))
d=cand.join(S,on="i1").join(O,on="i2").with_columns((pl.col("p2")-pl.col("p1")).alias("dl"))
d=d.with_columns(pl.when(pl.col("dl").is_in([3,4,5,7,9,11,13])).then(pl.lit("dec")).when(pl.col("dl")==0).then(pl.lit("same")).otherwise(None).alias("g")).drop_nulls("g")
a=d.select("country","g",pl.col("t2").list.set_difference(pl.col("t1")).alias("w")).explode("w").drop_nulls()
n=d.group_by("country","g").len("n")
c=a.group_by("country","g","w").len("k").join(n,on=["country","g"]).with_columns((pl.col("k")/pl.col("n")).alias("r"))
p=c.pivot(on="g",index=["country","w"],values="r").fill_null(0.0).with_columns(((pl.col("dec")+1e-4)/(pl.col("same")+1e-4)).alias("lift"))
for ct in ["US","France","India"]:
    t=p.filter((pl.col("country")==ct)&(pl.col("dec")>0.003)).sort("lift",descending=True).head(25)
    print(ct, [(r["w"],round(r["dec"],4),round(r["lift"],1)) for r in t.iter_rows(named=True)])
