import polars as pl
C = "data_cache/"
s1 = pl.read_parquet(C + "test_source1.parquet").with_row_index("i1")
fr = s1.filter(pl.col("country") == "France").select("i1")
p = pl.read_parquet("runs/v11/v11/test_p2_v11fr.parquet").join(fr, on="i1", how="semi")
print("France candidate pairs", p.height, "| p quartiles", [round(p["p"].quantile(q), 3) for q in (0.1, 0.25, 0.5, 0.75)])
band = p.filter((pl.col("p") > 0.15) & (pl.col("p") < 0.6))
print("uncertain band 0.15-0.6:", band.height, f"({band.height/p.height:.1%})")
oth = pl.concat([pl.read_parquet(C + "test_source2.parquet"), pl.read_parquet(C + "test_source3.parquet")]).with_row_index("i2")
ex = band.sample(36, seed=21).join(s1.select("i1", "business_name", "business_address"), on="i1").join(oth.select("i2", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="i2")
for r in ex.sort("p").iter_rows(named=True):
    print(f"p={r['p']:.2f} S1 {r['business_name']} | {r['business_address']}\n       R  {r['n2']} | {r['a2']}")
