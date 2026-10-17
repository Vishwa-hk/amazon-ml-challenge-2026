import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pipeline_v6 import truth
from judge_v4 import kept_entities
C="data_cache/"
n=pl.read_parquet("runs/blocking_v6/train_s1_ids.parquet").height
kept=kept_entities(n,0.19)
tr=truth("train").join(kept,on="i1",how="semi")
oof=pl.read_parquet("runs/v7b/gpu_decoy21/train_p2.parquet",columns=["i1","i2"])
miss=tr.join(oof,on=["i1","i2"],how="anti")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2").select("i2","business_address","business_name")
noaddr=(pl.col("business_address").is_null()|(pl.col("business_address").fill_null("").str.strip_chars()=="")|(pl.col("business_address")=="None"))
miss=miss.join(oth,on="i2").with_columns(noaddr.alias("na"), pl.col("business_name").fill_null("").str.contains(r"(?i)\.com|www\.|^@|\.in\b|\.net|\.org").alias("dom"))
ids=miss.select("i2").unique()
full=pl.scan_parquet("runs/blocking_v6/train_candidates.parquet").select("i1","i2","score","rank").join(ids.lazy(),on="i2",how="semi").collect()
full=full.join(kept,on="i1",how="semi").with_columns(pl.col("score").rank("min",descending=True).over("i2").alias("rank_r"))
x=miss.join(full,on=["i1","i2"],how="left")
print("misses",x.height,"| in full candidate file:",x["score"].is_not_null().sum())
for k,sub in (("no-address",x.filter(pl.col("na"))),("domain",x.filter(pl.col("dom"))),("other",x.filter(~pl.col("na")&~pl.col("dom")))):
    f=sub.filter(pl.col("score").is_not_null())
    print(f"{k:<11} n={sub.height:>7,} in-full={f.height:>7,} | rank(S1 side) q: {f['rank'].quantile(0.25)} {f['rank'].median()} {f['rank'].quantile(0.75)} | rank_r(record side)<=1: {(f['rank_r']<=1).sum():,} <=3: {(f['rank_r']<=3).sum():,} <=10: {(f['rank_r']<=10).sum():,}")
