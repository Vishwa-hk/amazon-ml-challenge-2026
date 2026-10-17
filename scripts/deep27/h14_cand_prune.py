# Trade-off: size of the SUBMITTED candidate list per S1 vs F0.5, when stage-2 decisions are
# restricted to that list. v8 OOF (test-like universe), 1/3 of kept train entities.
import sys, json; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
import pipeline_v6 as v6
from pathlib import Path
from judge_v3 import decide, owned, macro_f05
from judge_v4 import kept_entities
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
D = Path("runs/v8/gpu_slot")
rule = json.loads((D / "decision.json").read_text())["best_stage2"]["rule"]
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
sub = kept_entities(n, 0.19).filter(pl.col("i1") % 3 == 0)
tr = v6.truth("train").join(sub, on="i1", how="semi")
p1 = pl.scan_parquet(D / "train_p1.parquet").join(sub.lazy(), on="i1", how="semi").select("i1", "i2", pl.col("p").alias("p1")).collect()
p2 = pl.scan_parquet(D / "train_p2.parquet").join(sub.lazy(), on="i1", how="semi").select("i1", "i2", "rank", "p").collect()
p = p2.join(p1, on=["i1", "i2"], how="left").with_columns(pl.col("p1").rank("ordinal", descending=True).over("i1").alias("r1"))
rm = sub.with_row_index("j")
def score(keep):
    ch = decide(owned(keep, rule.get("max_rank")), rule, already_owned=True)
    f = macro_f05(ch.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), tr.join(rm, on="i1").select(pl.col("j").alias("i1"), "i2"), sub.height)
    per = keep.group_by("i1").len()
    return f["f05"], keep.height / sub.height, per["len"].median() if per.height else 0
base = score(p)
print(f"entities {sub.height:,} | full short-list: F0.5 {base[0]:.5f}, mean {base[1]:.1f} candidates/S1")
for t in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    f, m, med = score(p.filter(pl.col("p1") >= t)); print(f"  stage-1 p >= {t:<6} F0.5 {f:.5f} ({f-base[0]:+.5f}) | mean {m:5.2f} / median {med} candidates per S1")
for k in [3, 5, 8, 10, 15, 20]:
    f, m, med = score(p.filter(pl.col("r1") <= k)); print(f"  stage-1 top-{k:<3}   F0.5 {f:.5f} ({f-base[0]:+.5f}) | mean {m:5.2f} candidates per S1")
for t in [0.05, 0.01]:
    f, m, med = score(p.filter(pl.col("p") >= t)); print(f"  (ref) stage-2 p >= {t}  F0.5 {f:.5f} | mean {m:5.2f}")
print("true matches per S1 (mean):", tr.height / sub.height)
