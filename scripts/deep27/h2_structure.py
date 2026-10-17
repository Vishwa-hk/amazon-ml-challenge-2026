# Structure of the generator: per S1 counts by source; shape of unmatched records; exact dupes
import polars as pl
C="data_cache/"
gt=pl.read_parquet(C+"train_ground_truth.parquet")
p=(gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
   .rename({"source1_entity_id":"s1","matched_entity_ids":"m"}).filter(pl.col("m").is_not_null() & (pl.col("m")!="")))
c=p.group_by("s1").agg((pl.col("m").str.starts_with("S2")).sum().alias("n2"),(pl.col("m").str.starts_with("S3")).sum().alias("n3"))
print("joint (n2,n3) top:"); print(c.group_by("n2","n3").len().sort("len",descending=True).head(20))
print("S1 with zero S2:",(c["n2"]==0).mean()," zero S3:",(c["n3"]==0).mean())
s1=pl.read_parquet(C+"train_source1.parquet")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")])
matched=set(p["m"].to_list())
oth=oth.with_columns(pl.col("entity_id").is_in(list(matched)).alias("matched"))
print("unmatched share by country"); print(oth.group_by("country").agg(pl.col("matched").mean()))
# exact name+addr duplicates of S1 names among unmatched?
s1n=s1.select(pl.col("business_name").str.to_lowercase().alias("k")).unique()
o=oth.with_columns(pl.col("business_name").str.to_lowercase().alias("k"))
print("record name(lower) exactly equals some S1 name: matched",o.filter(pl.col("matched")).join(s1n,on="k",how="semi").height/o.filter(pl.col("matched")).height,
      " unmatched",o.filter(~pl.col("matched")).join(s1n,on="k",how="semi").height/o.filter(~pl.col("matched")).height)
