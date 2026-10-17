import polars as pl, sys
C="data_cache/"
s1=pl.read_parquet(C+"test_source1.parquet").select(pl.col("entity_id").alias("s1"),"country")
def pairs(sub):
    m=pl.read_csv(f"runs/submissions/{sub}/matching_results.tsv",separator="\t",infer_schema=False)
    return (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").is_not_null()&(pl.col("matched_entity_ids")!=""))
            .rename({"source1_entity_id":"s1","matched_entity_ids":"r"}))
full=pairs(sys.argv[1])
for other in sys.argv[2:]:
    o=pairs(other)
    lost=full.join(o,on=["s1","r"],how="anti").join(s1,on="s1").group_by("country").len("only_in_full")
    gain=o.join(full,on=["s1","r"],how="anti").join(s1,on="s1").group_by("country").len("only_in_export")
    ent=full.join(o,on=["s1","r"],how="anti").select("s1").vstack(o.join(full,on=["s1","r"],how="anti").select("s1")).unique().join(s1,on="s1").group_by("country").len("S1_changed")
    tot=s1.group_by("country").len("S1_total")
    print(f"--- {sys.argv[1]} vs {other}")
    print(tot.join(ent,on="country",how="left").join(lost,on="country",how="left").join(gain,on="country",how="left").fill_null(0).with_columns((pl.col("S1_changed")/pl.col("S1_total")*100).round(3).alias("pct_S1_changed")).sort("country"))
