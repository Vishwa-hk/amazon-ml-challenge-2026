# Decoys with SAME first number (offset 0) - how do their names differ? (train, full universe, high score)
import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pipeline_v6 import truth
C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2")
own=truth("train").select("i2").unique()
cand=(pl.scan_parquet("runs/blocking_v6/train_candidates_top100a20.parquet").select("i1","i2","rank","score")
      .filter((pl.col("i1")<200_000)&(pl.col("rank")<=3)&(pl.col("score")>=3.0)).collect()).join(own,on="i2",how="anti")
first=lambda c: pl.col(c).fill_null("").str.extract(r"(\d+)").cast(pl.Int64,strict=False)
d=cand.join(s1.select("i1",pl.col("business_name").alias("n1"),pl.col("business_address").alias("a1")),on="i1").join(oth.select("i2",pl.col("business_name").alias("n2"),pl.col("business_address").alias("a2")),on="i2")
d=d.with_columns((first("a2")-first("a1")).alias("dl"))
print("decoys high-score: offset buckets", d.group_by(pl.col("dl").is_in([1,2,3,4,5,7,9,11,13]).alias("set"), (pl.col("dl")==0).alias("zero")).len())
for r in d.filter(pl.col("dl")==0).sample(25,seed=4).iter_rows(named=True):
    print(f"S1 {r['n1']} | {r['a1']}\n   D {r['n2']} | {r['a2']}")
print("---- no number")
for r in d.filter(pl.col("dl").is_null()).sample(15,seed=4).iter_rows(named=True):
    print(f"S1 {r['n1']} | {r['a1']}\n   D {r['n2']} | {r['a2']}")
