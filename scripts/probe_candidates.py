"""
Diagnostic (not part of the submission): how good is name-key blocking, and do
house numbers separate true from false candidates?  Uses train only.
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from normalize import add_features  # noqa: E402

C = ROOT / "data_cache"
t0 = time.time()
s1 = add_features(pl.read_parquet(C / "train_source1.parquet"))
oth = add_features(pl.concat([pl.read_parquet(C / "train_source2.parquet"),
                              pl.read_parquet(C / "train_source3.parquet")]))
print(f"features built in {time.time()-t0:.0f}s")

gt = (pl.read_parquet(C / "train_ground_truth.parquet")
      .with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True)
      .drop_nulls().rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
      .with_columns(pl.lit(True).alias("is_true")))
n_true = gt.height

# US ZIP presence (would make 5-digit numbers non-discriminative)
for name, df in (("S1", s1), ("S2+S3", oth)):
    us = df.filter(pl.col("country") == "US")
    z = us.select(pl.col("business_address").fill_null("").str.contains(r"\b\d{5}\b").mean()).item()
    print(f"{name} US addresses containing a 5-digit number: {z:.2%}")

A = s1.select(pl.col("entity_id").alias("s1"), "country", "key_ts", "key_cp", pl.col("nums").alias("n1"))
B = oth.select(pl.col("entity_id").alias("r"), "country", "key_ts", "key_cp", pl.col("nums").alias("n2"))

for key in ("key_ts", "key_cp"):
    a = A.filter(pl.col(key) != "").group_by(["country", key]).len("na")
    b = B.filter(pl.col(key) != "").group_by(["country", key]).len("nb")
    prod = a.join(b, on=["country", key]).select((pl.col("na") * pl.col("nb")).sum()).item()
    print(f"{key}: candidate pairs if joined = {prod:,}")

cand = pl.concat([
    A.filter(pl.col(k) != "").join(B.filter(pl.col(k) != ""), on=["country", k]).select("s1", "r", "n1", "n2")
    for k in ("key_ts", "key_cp")
]).unique(["s1", "r"])
cand = cand.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("is_true").fill_null(False))
cand = cand.with_columns(
    pl.col("n1").list.set_intersection("n2").list.len().alias("shared"),
    ((pl.col("n1").list.len() == 0) | (pl.col("n2").list.len() == 0)).alias("one_empty"),
)
per_r = cand.group_by("r").agg(pl.col("s1").n_unique().alias("r_cands"))
cand = cand.join(per_r, on="r")

tp = cand["is_true"].sum()
print(f"\ncandidate pairs: {cand.height:,} | true among them: {tp:,} "
      f"| recall ceiling of name-key blocking: {tp/n_true:.2%} | candidate precision: {tp/cand.height:.2%}")
print("\nhouse-number relationship, by label:")
st = cand.group_by("is_true").agg(
    pl.len().alias("pairs"),
    (pl.col("shared") >= 1).mean().alias("share_a_number"),
    pl.col("one_empty").mean().alias("one_side_no_number"),
    ((pl.col("shared") == 0) & ~pl.col("one_empty")).mean().alias("numbers_conflict"),
    (pl.col("r_cands") == 1).mean().alias("record_has_single_S1_candidate"),
).sort("is_true")
for r in st.iter_rows(named=True):
    print(f"  {'TRUE ' if r['is_true'] else 'FALSE'} pairs={r['pairs']:>11,} | share a number {r['share_a_number']:.2%} "
          f"| one side has no number {r['one_side_no_number']:.2%} | numbers conflict {r['numbers_conflict']:.2%} "
          f"| record has a single S1 candidate {r['record_has_single_S1_candidate']:.2%}")

# Same relationship on ALL true pairs (not just those name-blocking finds)
s1n = s1.select(pl.col("entity_id").alias("s1"), pl.col("nums").alias("n1"))
on = oth.select(pl.col("entity_id").alias("r"), pl.col("nums").alias("n2"))
allt = gt.join(s1n, on="s1").join(on, on="r").with_columns(
    pl.col("n1").list.set_intersection("n2").list.len().alias("shared"),
    ((pl.col("n1").list.len() == 0) | (pl.col("n2").list.len() == 0)).alias("one_empty"))
print(f"\nALL true pairs: share a number {(allt['shared']>=1).mean():.2%} | one side no number "
      f"{allt['one_empty'].mean():.2%} | conflict {((allt['shared']==0)&~allt['one_empty']).mean():.2%}")
print(f"\ndone in {time.time()-t0:.0f}s")
