# Signed house-number offset: true variants vs decoys (train, candidate pairs rank<=10)
import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pipeline_v6 import truth
C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1").select("i1",pl.col("business_address").alias("a1"),"country")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2").select("i2",pl.col("business_address").alias("a2"))
tr=truth("train").with_columns(pl.lit(True).alias("y"))
cand=(pl.scan_parquet("runs/blocking_v6/train_candidates_top100a20.parquet").select("i1","i2","rank","score")
      .filter(pl.col("i1")<300_000).filter(pl.col("rank")<=10).collect())
d=cand.join(tr,on=["i1","i2"],how="left").with_columns(pl.col("y").fill_null(False))
d=d.join(s1,on="i1").join(oth,on="i2")
num=lambda c: pl.col(c).fill_null("").str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64,strict=False)).list.drop_nulls()
d=d.with_columns(num("a1").alias("n1"),num("a2").alias("n2"))
d=d.filter((pl.col("n1").list.len()>0)&(pl.col("n2").list.len()>0))
# primary number = first number
d=d.with_columns(pl.col("n1").list.first().alias("p1"),pl.col("n2").list.first().alias("p2"))
# was record decoy (not owned by ANY S1)?
own=truth("train").select("i2").unique().with_columns(pl.lit(True).alias("owned"))
d=d.join(own,on="i2",how="left").with_columns(pl.col("owned").fill_null(False))
d=d.with_columns((pl.col("p2")-pl.col("p1")).alias("dl"))
d=d.with_columns(pl.when(pl.col("y")).then(pl.lit("true")).when(~pl.col("owned")).then(pl.lit("decoy")).otherwise(pl.lit("other-owner")).alias("g"))
print(d.group_by("g").len())
sub=d.filter(pl.col("dl").abs()<=20)
t=sub.group_by("g","dl").len().pivot(on="g",index="dl",values="len").sort("dl").fill_null(0)
with pl.Config(tbl_rows=50): print(t)
# names similar: restrict to high cheap score to mimic hard cases
print("\nhigh-score pairs (score>=2.5):")
t2=d.filter(pl.col("score")>=2.5).with_columns(pl.when(pl.col("dl")==0).then(pl.lit("0")).when((pl.col("dl")>0)&(pl.col("dl")<=15)).then(pl.lit("+1..15")).when((pl.col("dl")<0)&(pl.col("dl")>=-15)).then(pl.lit("-1..-15")).otherwise(pl.lit("far")).alias("b"))
print(t2.group_by("g","b").len().pivot(on="g",index="b",values="len").fill_null(0))
print(t2.group_by("g","country","b").len().pivot(on="g",index=["country","b"],values="len").fill_null(0).sort("country","b"))
