import polars as pl
E="runs/v6_error_analysis/"; C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1").select("i1",pl.col("business_address").alias("a1"),"country")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2").select("i2",pl.col("business_address").alias("a2"))
first=lambda c: pl.col(c).fill_null("").str.extract(r"(\d+)").cast(pl.Int64,strict=False)
S={1,2,3,4,5,7,9,11,13}
def b(df):
    d=df.select("i1","i2").join(s1,on="i1").join(oth,on="i2").with_columns((first("a2")-first("a1")).alias("dl"))
    return d.with_columns(pl.when(pl.col("dl").is_null()).then(pl.lit("no-num")).when(pl.col("dl")==0).then(pl.lit("0"))
        .when(pl.col("dl").is_in(list(S))).then(pl.lit("decoy-set")).when(pl.col("dl").abs()<=15).then(pl.lit("small-other")).otherwise(pl.lit("far")).alias("b"))
for f in ["FP_decoy","FP_wrong_owner","FN_lost_owner","FN_below_cut","correct_match"]:
    d=b(pl.read_parquet(E+f+".parquet"))
    vc=d.group_by("b").len().with_columns((pl.col("len")/pl.col("len").sum()).round(3).alias("share")).sort("b")
    print(f, d.height, dict(zip(vc["b"],vc["share"])))
