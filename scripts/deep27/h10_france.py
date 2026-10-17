# Label-free France look: candidates in the uncertain band (0.1<p<0.6) for France S1s — what differs?
import polars as pl
C="data_cache/"
s1=pl.read_parquet(C+"test_source1.parquet").with_row_index("i1")
fr=s1.filter(pl.col("country")=="France").select("i1")
p2=pl.scan_parquet("runs/v7/gpu_decoy/test_p2.parquet").join(fr.lazy(),on="i1",how="semi").filter((pl.col("p")>0.1)&(pl.col("p")<0.6)&(pl.col("rank")<=5)).collect()
print("France uncertain-band pairs:",p2.height, "of France S1", fr.height)
oth=pl.concat([pl.read_parquet(C+"test_source2.parquet"),pl.read_parquet(C+"test_source3.parquet")]).with_row_index("i2")
ex=p2.sample(40,seed=7).join(s1.select("i1","business_name","business_address"),on="i1").join(oth.select("i2",pl.col("business_name").alias("n2"),pl.col("business_address").alias("a2")),on="i2")
for r in ex.iter_rows(named=True):
    print(f"p={r['p']:.2f} S1 {r['business_name']} | {r['business_address']}\n        R  {r['n2']} | {r['a2']}")
