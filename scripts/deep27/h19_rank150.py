import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v6 as v6
from judge_v4 import kept_entities
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
tr = v6.truth("train").join(kept, on="i1", how="semi")
full = pl.scan_parquet(BL8 / "train_candidates.parquet").select("i1", "i2", "rank", "arank", pl.col("nrank").fill_null(99))
hit = full.join(tr.lazy(), on=["i1", "i2"], how="semi").collect()
incut = hit.filter((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5))
extra = hit.filter(~((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5)))
print(f"true pairs (kept universe) {tr.height:,} | in cut {incut.height:,} ({incut.height/tr.height:.3%}) | extra in ranks 101-150 {extra.height:,} ({extra.height/tr.height:.3%}) | missing even from top-150 {tr.height-hit.height:,}")
extra_pairs = full.filter(~((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5))).select(pl.len()).collect().item()
print(f"cost: {extra_pairs:,} extra candidate pairs to score ({extra_pairs/n:.1f} per S1)")
