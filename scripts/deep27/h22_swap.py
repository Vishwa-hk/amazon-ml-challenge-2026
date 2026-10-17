# Is "one dictionary word swapped for another" a decoy signal on TRAIN (US/India)?
import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v8 as p8
import pipeline_v6 as v6
from translit import latin_tokens
from judge_v4 import kept_entities
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
tr = v6.truth("train").with_columns(pl.lit(True).alias("y"))
own = tr.select("i2").unique().with_columns(pl.lit(True).alias("owned"))
p1 = pl.read_parquet("runs/v8/gpu_slot/train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.05).join(kept, on="i1", how="semi")
s1raw, _, oth, _ = p8.load_raw_v8("train")
S = s1raw.with_row_index("i1").select("i1", "country", latin_tokens("business_name").list.unique().alias("t1"))
O = oth.with_row_index("i2").select("i2", latin_tokens("business_name").list.unique().alias("t2"))
from normalize import DROP_TOKENS
from normalize_v2 import GENERIC, HONORIFICS
STOP = set(DROP_TOKENS) | set(GENERIC) | set(HONORIFICS) | {"pvt", "private", "limited", "ltd", "llp", "inc", "llc", "corp", "co", "company", "pc", "pllc", "lp", "the", "and", "of"}
dic = (S.explode("t1").group_by("country", "t1").len("c").filter(pl.col("c") >= 20).filter(~pl.col("t1").is_in(list(STOP)))
       .select("country", pl.col("t1").alias("w"), pl.lit(True).alias("dict")))
print("content-word dictionary size:", dic.height)
x = p1.join(S, on="i1").join(O, on="i2").with_columns(pl.col("t2").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("t2").alias("miss")).with_row_index("pid")
def cnt(col, name):
    e = x.select("pid", "country", pl.col(col).alias("w")).explode("w").drop_nulls().join(dic, on=["country", "w"], how="left")
    return e.group_by("pid").agg(pl.col("dict").fill_null(False).sum().alias(name + "_dict"), (~pl.col("dict").fill_null(False)).sum().alias(name + "_other"))
x = x.join(cnt("add", "a"), on="pid", how="left").join(cnt("miss", "m"), on="pid", how="left").fill_null(0)
x = x.join(tr, on=["i1", "i2"], how="left").join(own, on="i2", how="left").with_columns(pl.col("y").fill_null(False), pl.col("owned").fill_null(False))
x = x.with_columns(pl.when(pl.col("y")).then(pl.lit("true")).when(~pl.col("owned")).then(pl.lit("decoy")).otherwise(pl.lit("other")).alias("g"),
                   pl.when((pl.col("a_dict") >= 1) & (pl.col("m_dict") >= 1)).then(pl.lit("swap")).when(pl.col("a_dict") >= 1).then(pl.lit("add-dict"))
                   .when(pl.col("m_dict") >= 1).then(pl.lit("drop-dict")).when((pl.col("a_other") > 0) | (pl.col("m_other") > 0)).then(pl.lit("typo/other")).otherwise(pl.lit("same")).alias("kind"))
t = x.group_by("country", "kind", "g").len().pivot(on="g", index=["country", "kind"], values="len").fill_null(0).sort("country", "kind")
t = t.with_columns((pl.col("decoy") / (pl.col("true") + 1)).round(3).alias("decoy_per_true"))
with pl.Config(tbl_rows=20): print(t)
