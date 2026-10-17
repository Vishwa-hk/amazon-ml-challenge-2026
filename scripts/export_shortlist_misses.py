"""
Export the TRAINING true matches that short-lists v3 still misses, human-readable,
for studying the remaining blocking gap (e.g. Varun's transliteration work).
Training data only (it has an answer key) — fair to study and share in the team.

Writes runs/exports/v3_shortlist_misses_train.tsv and a summary, never overwriting.
Columns: category, country, s1_id, s1_name, s1_address, rec_id, rec_name, rec_address
Categories: native_script | no_address | no_house_number_one_side | other
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
C, BL = ROOT / "data_cache", ROOT / "runs" / "blocking_v3"
OUT = ROOT / "runs" / "exports"
OUT.mkdir(parents=True, exist_ok=True)
dest = OUT / "v3_shortlist_misses_train.tsv"
if dest.exists():
    raise SystemExit(f"refusing to overwrite {dest}")
LATIN_ONLY = r"^[\p{Latin}\p{Common}\p{Inherited}]*$"

s1 = pl.read_parquet(C / "train_source1.parquet").with_row_index("i1")
oth = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")]).with_row_index("i2")
gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
      .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "rec_id"}))
pairs = (gt.join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
         .join(oth.select("i2", pl.col("entity_id").alias("rec_id")), on="rec_id"))
cand = pl.scan_parquet(BL / "train_candidates.parquet").select("i1", "i2").collect()
miss = pairs.join(cand, on=["i1", "i2"], how="anti")
digits = lambda c: pl.col(c).fill_null("").str.contains(r"\d")
m = (miss.join(s1.select("i1", "country", pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_address")), on="i1")
     .join(oth.select("i2", pl.col("business_name").alias("rec_name"), pl.col("business_address").alias("rec_address")), on="i2"))
m = m.with_columns(
    pl.when(~pl.col("rec_name").fill_null("").str.contains(LATIN_ONLY)).then(pl.lit("native_script"))
    .when(pl.col("rec_address").fill_null("").str.strip_chars() == "").then(pl.lit("no_address"))
    .when(~digits("s1_address") | ~digits("rec_address")).then(pl.lit("no_house_number_one_side"))
    .otherwise(pl.lit("other")).alias("category"))
m = m.select("category", "country", "s1_id", "s1_name", "s1_address", "rec_id", "rec_name", "rec_address").sort("category", "country")
m.write_csv(dest, separator="\t", line_terminator="\n")
summary = m.group_by("category", "country").len().sort("category", "country")
lines = [f"v3 short-list misses (train): {m.height:,} of {pairs.height:,} true pairs ({m.height/pairs.height:.2%})"]
lines += [f"  {r['category']:<26} {r['country']:<6} {r['len']:>8,}" for r in summary.iter_rows(named=True)]
(OUT / "v3_shortlist_misses_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"-> {dest}")
