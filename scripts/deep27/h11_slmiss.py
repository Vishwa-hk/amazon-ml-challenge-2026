# Categorize short-list misses (v7b, test-like universe) and estimate points per category
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
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2")
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1").select("i1","country")
m=miss.join(oth.select("i2","business_name","business_address"),on="i2").join(s1,on="i1")
LAT=r"^[\p{Latin}\p{Common}\p{Inherited}]*$"
nm=pl.col("business_name").fill_null("")
m=m.with_columns(pl.when(nm.str.contains(r"(?i)\.com|www\.|^@|\.in\b|\.net|\.org")).then(pl.lit("domain"))
    .when(~nm.str.contains(LAT)).then(pl.lit("native"))
    .when(pl.col("business_address").is_null()|(pl.col("business_address").fill_null("").str.strip_chars()=="")|(pl.col("business_address")=="None")).then(pl.lit("no-address"))
    .otherwise(pl.lit("other")).alias("cat"))
print("short-list misses:",m.height,"of true pairs",tr.height)
print(m.group_by("country","cat").len().sort("country","len",descending=[False,True]))
# how common are these categories among ALL true pairs, and recall per category
a=tr.join(oth.select("i2","business_name","business_address"),on="i2").with_columns(
    pl.when(nm.str.contains(r"(?i)\.com|www\.|^@|\.in\b|\.net|\.org")).then(pl.lit("domain"))
    .when(~nm.str.contains(LAT)).then(pl.lit("native"))
    .when(pl.col("business_address").is_null()|(pl.col("business_address").fill_null("").str.strip_chars()=="")|(pl.col("business_address")=="None")).then(pl.lit("no-address"))
    .otherwise(pl.lit("other")).alias("cat"))
print(a.group_by("cat").len().join(m.group_by("cat").len("miss"),on="cat").with_columns((1-pl.col("miss")/pl.col("len")).alias("recall")))
for c in ["other","no-address","native"]:
    print("---",c)
    for r in m.filter(pl.col("cat")==c).sample(8,seed=1).join(pl.read_parquet(C+"train_source1.parquet").with_row_index("i1").select("i1",pl.col("business_name").alias("n1"),pl.col("business_address").alias("a1")),on="i1").iter_rows(named=True):
        print(f"  S1 {r['n1']} | {r['a1']}\n   R {r['business_name']} | {r['business_address']}")
