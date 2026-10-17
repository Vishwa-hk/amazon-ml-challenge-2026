# Offsets between NEW record numbers and S1 numbers (all numbers), true vs decoy, by country; train rank<=5
import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pipeline_v6 import truth
C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2")
tr=truth("train").with_columns(pl.lit(True).alias("y")); own=tr.select("i2").unique().with_columns(pl.lit(True).alias("owned"))
cand=(pl.scan_parquet("runs/blocking_v6/train_candidates_top100a20.parquet").select("i1","i2","rank","score")
      .filter((pl.col("i1")<300_000)&(pl.col("rank")<=5)&(pl.col("score")>=2.5)).collect())
d=cand.join(tr,on=["i1","i2"],how="left").join(own,on="i2",how="left").with_columns(pl.col("y").fill_null(False),pl.col("owned").fill_null(False))
d=d.with_columns(pl.when(pl.col("y")).then(pl.lit("true")).when(~pl.col("owned")).then(pl.lit("decoy")).otherwise(pl.lit("other")).alias("g"))
nums=lambda c: pl.col(c).fill_null("").str.extract_all(r"\d{1,7}").list.eval(pl.element().cast(pl.Int64,strict=False)).list.unique()
S=s1.select("i1","country",nums("business_address").alias("n1")); O=oth.select("i2",nums("business_address").alias("n2"))
d=d.join(S,on="i1").join(O,on="i2").with_row_index("pid")
new=d.select("pid","g","country",pl.col("n2").list.set_difference(pl.col("n1")).alias("r"),"n1").explode("r").drop_nulls("r").explode("n1").drop_nulls("n1")
new=new.with_columns((pl.col("r")-pl.col("n1")).alias("dl")).filter((pl.col("dl")>=-200)&(pl.col("dl")<=200)&(pl.col("dl")!=0))
# per pair: set of distinct offsets; count pairs having each offset
t=new.unique(["pid","dl"]).group_by("country","g","dl").len().pivot(on="g",index=["country","dl"],values="len").fill_null(0).sort("country","dl")
t=t.with_columns((pl.col("decoy")/(pl.col("true")+pl.col("other")+1)).alias("ratio")); print(t.filter(pl.col("ratio")>2).sort("country","dl").select("country","dl","decoy","true","other","ratio").to_pandas().to_string())
