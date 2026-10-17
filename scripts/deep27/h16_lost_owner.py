import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v6 as v6
v6.BL6 = Path("runs/blocking_v8")
C="data_cache/"
fn=pl.read_parquet("runs/v8_error_analysis/FN_lost_owner.parquet")
p2=pl.scan_parquet("runs/v8/gpu_slot/train_p2.parquet").join(fn.select("i2").unique().lazy(),on="i2",how="semi").collect()
win=p2.sort("p",descending=True).group_by("i2").first().rename({"i1":"w1","p":"pw"}).select("i2","w1","pw")
x=fn.join(win,on="i2").filter(pl.col("w1")!=pl.col("i1"))
tr=v6.truth("train")
# is the winner a real S1 whose OWN true set contains... (can't: each record has one owner). how similar are winner & true S1?
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1").select("i1","business_name","business_address")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2").select("i2",pl.col("business_name").alias("n2"),pl.col("business_address").alias("a2"))
x=x.join(s1.rename({"business_name":"nt","business_address":"at"}),on="i1").join(s1.rename({"i1":"w1","business_name":"nw","business_address":"aw"}),on="w1").join(oth,on="i2")
noaddr=pl.col("a2").is_null()|(pl.col("a2").fill_null("").str.strip_chars().is_in(["","None","null","<NULL>"]))
print("lost-owner pairs:",x.height,"| record has no address:",x.filter(noaddr).height,"| true & winner S1 names identical (lower):",x.filter(pl.col("nt").str.to_lowercase()==pl.col("nw").str.to_lowercase()).height)
print("p(true) quartiles",x["p"].quantile(0.25),x["p"].median(),x["p"].quantile(0.75)," | p(winner) median",x["pw"].median())
for r in x.sample(18,seed=3).iter_rows(named=True):
    print(f"p_true={r['p']:.2f} p_win={r['pw']:.2f}\n  REC  {r['n2']} | {r['a2']}\n  TRUE {r['nt']} | {r['at']}\n  WIN  {r['nw']} | {r['aw']}")
