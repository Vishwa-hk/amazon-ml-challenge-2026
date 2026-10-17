# Label-free: signed first-number offset histogram on TEST candidate pairs (rank<=10), by country
import polars as pl
C="data_cache/"
s1=pl.read_parquet(C+"test_source1.parquet").with_row_index("i1").select("i1",pl.col("business_address").alias("a1"),"country")
oth=pl.concat([pl.read_parquet(C+"test_source2.parquet"),pl.read_parquet(C+"test_source3.parquet")]).with_row_index("i2").select("i2",pl.col("business_address").alias("a2"))
cand=(pl.scan_parquet("runs/blocking_v6/test_candidates_top100a20.parquet").select("i1","i2","rank","score")
      .filter(pl.col("i1")<400_000).filter((pl.col("rank")<=10)&(pl.col("score")>=2.5)).collect())
d=cand.join(s1,on="i1").join(oth,on="i2")
first=lambda c: pl.col(c).fill_null("").str.extract(r"(\d+)").cast(pl.Int64,strict=False)
d=d.with_columns((first("a2")-first("a1")).alias("dl")).drop_nulls("dl")
t=d.filter(pl.col("dl").abs()<=15).group_by("country","dl").len().pivot(on="country",index="dl",values="len").sort("dl").fill_null(0)
with pl.Config(tbl_rows=40): print(t)
