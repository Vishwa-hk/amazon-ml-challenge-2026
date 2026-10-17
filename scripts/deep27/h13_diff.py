import polars as pl, sys
C="data_cache/"
def pairs(sub):
    m=pl.read_csv(f"runs/submissions/{sub}/matching_results.tsv",separator="\t",infer_schema=False)
    return m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids")!="").rename({"source1_entity_id":"s1","matched_entity_ids":"r"})
a,b=pairs(sys.argv[1]),pairs(sys.argv[2])
s1=pl.read_parquet(C+"test_source1.parquet").rename({"entity_id":"s1"})
o=pl.concat([pl.read_parquet(C+"test_source2.parquet"),pl.read_parquet(C+"test_source3.parquet")]).rename({"entity_id":"r","business_name":"n2","business_address":"a2"}).drop("country")
fr=s1.filter(pl.col("country")=="France").select("s1")
lost=a.join(b,on=["s1","r"],how="anti").join(fr,on="s1",how="semi"); gain=b.join(a,on=["s1","r"],how="anti").join(fr,on="s1",how="semi")
print("France pairs only in",sys.argv[1],lost.height,"| only in",sys.argv[2],gain.height)
for nm,d in (("dropped",lost),("added",gain)):
    print("---",nm)
    for r in d.sample(min(14,d.height),seed=2).join(s1,on="s1").join(o,on="r").iter_rows(named=True):
        print(f"  S1 {r['business_name']} | {r['business_address']}\n   R {r['n2']} | {r['a2']}")
