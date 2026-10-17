import polars as pl, numpy as np
C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("r1")
s2=pl.read_parquet(C+"train_source2.parquet").with_row_index("r2")
s3=pl.read_parquet(C+"train_source3.parquet").with_row_index("r3")
gt=pl.read_parquet(C+"train_ground_truth.parquet")
print(s1.head(5)); print(s2.head(5)); print(gt.head(5))
# gt order == s1 order?
print("gt order same as s1:", (gt["source1_entity_id"]==s1["entity_id"]).mean())
p=(gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls()
   .rename({"source1_entity_id":"entity_id","matched_entity_ids":"m"}).filter(pl.col("m")!=""))
p=p.join(s1.select("entity_id","r1"),on="entity_id")
a=p.join(s2.select(pl.col("entity_id").alias("m"),"r2"),on="m")
b=p.join(s3.select(pl.col("entity_id").alias("m"),"r3"),on="m")
print("S2 pairs",a.height,"spearman r1~r2", np.corrcoef(a["r1"].rank(),a["r2"].rank())[0,1])
print("S3 pairs",b.height,"spearman r1~r3", np.corrcoef(b["r1"].rank(),b["r3"].rank())[0,1])
# numeric id relation
f=lambda s: s.str.extract(r"(\d+)$").cast(pl.Int64)
a=a.with_columns(f(pl.col("entity_id")).alias("n1"),f(pl.col("m")).alias("n2"))
print("id corr S1~S2", np.corrcoef(a["n1"],a["n2"])[0,1])
print("id diff sample", (a["n2"]-a["n1"]).head(10).to_list())
# contiguity: within an S1, are its S2 rows adjacent?
g=a.group_by("r1").agg(pl.col("r2").max()-pl.col("r2").min(),pl.len()).filter(pl.col("len")>1)
print("S2 row span of same-S1 clusters: median",g["r2"].median(),"| random expectation ~",s2.height/3)
# id digit lengths / prefixes
print(s1["entity_id"].str.len_chars().value_counts().sort("entity_id"))
print(s2["entity_id"].str.len_chars().value_counts().sort("entity_id"))
