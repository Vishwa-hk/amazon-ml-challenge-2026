# Gamble ONLY on records without an address (held-out train, v12b judge).
import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v6 as v6
from judge_v3 import decide, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import region
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
C = "data_cache/"
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
truth = v6.truth("train").join(kept, on="i1", how="semi")
cut = region(Path("runs/v8/gpu_slot/train_p1.parquet"), kept, 0.01)
oof = pl.read_parquet("runs/v12/v12b/train_p2.parquet").join(cut, on=["i1", "i2"], how="semi")
oth = pl.concat([pl.read_parquet(C + "train_source2.parquet"), pl.read_parquet(C + "train_source3.parquet")]).with_row_index("i2")
noaddr = oth.filter(pl.col("business_address").is_null() | pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None", "null", "<NULL>", "N/A"])).select("i2")
print("records without address:", noaddr.height)
own = owned(oof, 1000)
rm = kept.with_row_index("j")
def F(p):
    return macro_f05(p.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), truth.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), kept.height)
base = decide(own, {"kind": "expected_f05", "t_scale": 1.0, "max_rank": 1000, "floor": 0.01}, already_owned=True)
b = F(base); print(f"current rule                                   F0.5 {b['f05']:.5f}")
tr = truth.with_columns(pl.lit(True).alias("y"))
for t in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
    extra = own.join(noaddr, on="i2", how="semi").filter(pl.col("p") >= t).select("i1", "i2")
    new = extra.join(base, on=["i1", "i2"], how="anti")
    hit = new.join(tr, on=["i1", "i2"], how="left")["y"].fill_null(False).mean() if new.height else 0
    f = F(pl.concat([base, extra]).unique())
    print(f"gamble on no-address records only, p >= {t:<4}  F0.5 {f['f05']:.5f} ({f['f05']-b['f05']:+.5f}) | {new.height:,} extra matches, {hit:.1%} of them correct")
