# Label-free per-country diagnostics of submissions: empty rate, mean matches, and (train) expected values
import sys, polars as pl
C="data_cache/"
s1=pl.read_parquet(C+"test_source1.parquet").select(pl.col("entity_id").alias("source1_entity_id"),"country")
for sub in sys.argv[1:]:
    m=pl.read_csv(f"runs/submissions/{sub}/matching_results.tsv",separator="\t",infer_schema=False).join(s1,on="source1_entity_id")
    m=m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element()!="")).list.len().alias("k"))
    r=m.group_by("country").agg((pl.col("k")==0).mean().round(4).alias("empty"),pl.col("k").mean().round(3).alias("mean_k")).sort("country")
    print(f"{sub:<28}", " | ".join(f"{a}: empty {b:.2%} mean {c}" for a,b,c in r.iter_rows()))
print("train truth: empty 5.58%, mean 3.46 (both countries)")
