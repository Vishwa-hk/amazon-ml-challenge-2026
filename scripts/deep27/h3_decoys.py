# How are decoy (unmatched) records generated? Show decoy, its best-scoring S1, and that S1's true variants.
import sys; sys.path.insert(0,"code/business_entity_resolution/src")
import polars as pl
from pipeline_v6 import truth
C="data_cache/"
s1=pl.read_parquet(C+"train_source1.parquet").with_row_index("i1")
oth=pl.concat([pl.read_parquet(C+"train_source2.parquet"),pl.read_parquet(C+"train_source3.parquet")]).with_row_index("i2")
tr=truth("train")
un=oth.join(tr,on="i2",how="anti").select("i2")
print("unmatched records",un.height,"of",oth.height)
samp=un.sample(4000,seed=1)
cand=(pl.scan_parquet("runs/blocking_v6/train_candidates_top100a20.parquet").select("i1","i2","score")
      .join(samp.lazy(),on="i2",how="semi").collect())
print("sampled decoys with any candidate:",cand["i2"].n_unique()/samp.height)
best=cand.sort("score",descending=True).group_by("i2").first()
# is the decoy's best S1 itself matched to records? how many true variants?
nv=tr.group_by("i1").len("nv")
best=best.join(nv,on="i1",how="left").fill_null(0)
print("best-S1 true-variant count dist:",best["nv"].describe())
N1=s1.select("i1","business_name","business_address")
N2=oth.select("i2","business_name","business_address")
ex=best.sort("score",descending=True).head(4000).sample(30,seed=2)
for r in ex.iter_rows(named=True):
    a=N1.filter(pl.col("i1")==r["i1"]).row(0); d=N2.filter(pl.col("i2")==r["i2"]).row(0)
    print(f"\nscore {r['score']:.2f}  S1 : {a[1]} | {a[2]}\n            DEC: {d[1]} | {d[2]}")
    v=tr.filter(pl.col("i1")==r["i1"]).join(N2,on="i2").head(3)
    for w in v.iter_rows(): print(f"            tru: {w[2]} | {w[3]}")
