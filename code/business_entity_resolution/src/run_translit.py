"""
Build the transliteration dictionary from TRAIN pairs, check it honestly, and save it.

  1. held-out check: dictionary from half of the train S1 entities, evaluated on the
     other half's aligned word pairs (coverage + accuracy)
  2. final dictionary from all train pairs -> runs/translit/dictionary.parquet
  3. coverage of native-script words in TEST records (inputs only, no labels)

Usage:  python run_translit.py
"""
import time
from pathlib import Path

import polars as pl

from translit import LATIN_ONLY, aligned_pairs, build_dictionary, native_tokens

PROJECT = Path(__file__).resolve().parents[3]
C, OUT = PROJECT / "data_cache", PROJECT / "runs" / "translit"


def true_pairs() -> pl.DataFrame:
    s1 = pl.read_parquet(C / "train_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("s1_name"))
    oth = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")]) \
        .select(pl.col("entity_id").alias("r"), pl.col("business_name").alias("rec_name"))
    gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
          .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"}))
    return gt.join(s1, on="s1").join(oth, on="r")


def main() -> int:
    dest = OUT / "dictionary.parquet"
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tp = true_pairs()
    nat = tp.filter(~pl.col("rec_name").fill_null("").str.contains(LATIN_ONLY))
    print(f"true pairs {tp.height:,} | with a native-script record name {nat.height:,} ({time.time()-t0:.0f}s)")

    half = (pl.col("s1").hash(seed=3) % 2) == 0
    d_a = build_dictionary(tp.filter(half))
    ev = aligned_pairs(tp.filter(~half)).join(d_a.select("native", pl.col("latin").alias("pred")), on="native", how="left")
    cov, acc = ev["pred"].is_not_null().mean(), (ev["pred"] == ev["latin"]).mean()
    acc_known = ev.filter(pl.col("pred").is_not_null()).select((pl.col("pred") == pl.col("latin")).mean()).item()
    print(f"held-out check (dictionary from half, tested on the other half, {ev.height:,} aligned words):")
    print(f"   coverage {cov:.1%} | correct overall {acc:.1%} | correct when known {acc_known:.1%}")
    aligned_share = aligned_pairs(tp).select(pl.len()).item()
    print(f"   (aligned words from all train pairs: {aligned_share:,})")

    d = build_dictionary(tp)
    d.write_parquet(dest)
    print(f"final dictionary: {d.height:,} native words -> {dest}")
    print("   examples: " + ", ".join(f"{r['native']}->{r['latin']}" for r in d.sort("n", descending=True).head(12).iter_rows(named=True)))

    te = pl.concat([pl.read_parquet(C / "test_source2.parquet"), pl.read_parquet(C / "test_source3.parquet")])
    ten = te.filter(~pl.col("business_name").fill_null("").str.contains(LATIN_ONLY)).select(native_tokens("business_name").alias("t"))
    words = ten.with_row_index("row").explode("t").filter(~pl.col("t").str.contains(LATIN_ONLY))
    known = words.join(d.select(pl.col("native").alias("t")), on="t", how="semi")
    full = words.with_columns(pl.col("t").is_in(d["native"].implode()).alias("k")).group_by("row").agg(pl.col("k").all())
    print(f"TEST: {ten.height:,} native-script record names | native words covered {known.height/words.height:.1%} "
          f"| names fully covered {full['k'].mean():.1%}")
    print(f"done ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
