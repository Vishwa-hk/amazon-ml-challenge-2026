# The 75,781 true pairs missing from the full v8 top-150 candidate file: what evidence do they carry?
import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v8 as p8   # v8 inputs (translit + segmentation)
import pipeline_v6 as v6
from blocking_v2 import prepare, _jac
from judge_v4 import kept_entities
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
tr = v6.truth("train").join(kept, on="i1", how="semi")
full = pl.scan_parquet(BL8 / "train_candidates.parquet").select("i1", "i2")
miss = tr.join(full.join(tr.lazy(), on=["i1", "i2"], how="semi").collect(), on=["i1", "i2"], how="anti")
print("missing true pairs:", miss.height)
s1raw, _, oth, _ = p8.load_raw_v8("train")
S = prepare(s1raw).select(pl.col("i").alias("i1"), "country", pl.col("th").alias("th1"), pl.col("ah").alias("ah1"), pl.col("nums").alias("n1"))
O = prepare(oth).select(pl.col("i").alias("i2"), pl.col("th").alias("th2"), pl.col("ah").alias("ah2"), pl.col("nums").alias("n2"))
x = miss.join(S, on="i1").join(O, on="i2").with_columns(_jac("th1", "th2").alias("tok_jac"), _jac("ah1", "ah2").alias("addr_jac"),
    (pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("num_sh"), (pl.col("ah2").list.len() == 0).alias("rec_no_addr_words"))
x = x.with_columns(pl.when(pl.col("tok_jac") == 0).then(pl.lit("name0")).when(pl.col("tok_jac") < 0.5).then(pl.lit("name<.5")).otherwise(pl.lit("name>=.5")).alias("name"),
                   pl.when(pl.col("addr_jac") >= 0.5).then(pl.lit("addr>=.5")).when(pl.col("addr_jac") > 0).then(pl.lit("addr>0")).otherwise(pl.lit("addr0")).alias("addr"))
print(x.group_by("country", "name", "addr", "num_sh").len().sort("len", descending=True).head(20))
strong = x.filter((pl.col("addr_jac") >= 0.5) & pl.col("num_sh"))
print(f"strong address evidence (addr_jac>=0.5 & shared number): {strong.height:,} of {x.height:,} ({strong.height/x.height:.1%})")
