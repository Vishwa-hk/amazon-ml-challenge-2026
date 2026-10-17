# Vishwa's "gamble" idea, measured on held-out train (v12b judge, test-like universe, submitted candidate region).
import sys, json; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v6 as v6
from judge_v3 import decide, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import region
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
truth = v6.truth("train").join(kept, on="i1", how="semi")
cut = region(Path("runs/v8/gpu_slot/train_p1.parquet"), kept, 0.01)
oof = pl.read_parquet("runs/v12/v12b/train_p2.parquet").join(cut, on=["i1", "i2"], how="semi")
own = owned(oof, 1000)                       # each record's single most likely business
rm = kept.with_row_index("j")
def F(pairs):
    return macro_f05(pairs.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), truth.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), kept.height)
base = decide(own, {"kind": "expected_f05", "t_scale": 1.0, "max_rank": 1000, "floor": 0.01}, already_owned=True)
b = F(base); print(f"current rule (expected-F0.5)                        F0.5 {b['f05']:.5f}  P {b['precision']:.4f} R {b['recall']:.4f}  matches {base.height:,}")
for t in (0.5, 0.4, 0.3, 0.2):
    extra = own.filter(pl.col("p") >= t).select("i1", "i2")
    g = pl.concat([base, extra]).unique()
    f = F(g); print(f"gamble: + every record's top business if p >= {t:<4}   F0.5 {f['f05']:.5f}  P {f['precision']:.4f} R {f['recall']:.4f}  matches {g.height:,}  ({f['f05']-b['f05']:+.5f})")
for t in (0.5, 0.4):
    g = own.filter(pl.col("p") >= t).select("i1", "i2")
    f = F(g); print(f"plain rule: match if p >= {t} (more likely right than wrong)  F0.5 {f['f05']:.5f}  P {f['precision']:.4f} R {f['recall']:.4f}  ({f['f05']-b['f05']:+.5f})")
# how often is the extra gambled pair actually right?
tr = truth.with_columns(pl.lit(True).alias("y"))
for lo, hi in ((0.5, 1.01), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3)):
    ex = own.filter((pl.col("p") >= lo) & (pl.col("p") < hi)).join(base, on=["i1", "i2"], how="anti").join(tr, on=["i1", "i2"], how="left")
    print(f"  records NOT chosen now with top-business p in [{lo},{hi}): {ex.height:>7,} | actually correct {ex['y'].fill_null(False).mean():.1%}")
