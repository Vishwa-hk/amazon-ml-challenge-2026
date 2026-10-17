import polars as pl
C = "data_cache/"
err = pl.read_parquet("runs/v9/v9_full/errors_heldout.parquet")
s1 = pl.read_parquet(C + "train_source1.parquet").with_row_index("i1").select("i1", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
oth = pl.concat([pl.read_parquet(C + "train_source2.parquet"), pl.read_parquet(C + "train_source3.parquet")]).with_row_index("i2").select("i2", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
for kind, num, k in (("FP", "num=", 16), ("FN below-cut", "num=", 16), ("FN below-cut", "far/other", 10)):
    sub = err.filter((pl.col("kind") == kind) & (pl.col("num") == num) & (pl.col("name") != "name0")).sample(k, seed=11).join(s1, on="i1").join(oth, on="i2")
    print(f"=== {kind} / {num}")
    for r in sub.iter_rows(named=True):
        print(f"p={r['p']:.2f} S1 {r['n1']} | {r['a1']}\n       R  {r['n2']} | {r['a2']}")
