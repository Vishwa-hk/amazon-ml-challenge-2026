"""
Day-1 EDA for the Amazon ML Challenge 2026 (Business Entity Resolution).

Answers the questions that shape the approach:
  - data integrity (row counts, duplicate ids, empty fields)
  - how many Source-1 entities are singletons, how big match lists are
  - country mix per source (train: US/India; test adds France)
  - how "easy" true pairs are (exact / normalized name equality)
  - whether S2/S3 records map to at most one S1 entity
  - what fraction of S2/S3 records are distractors (never matched)

Also caches every TSV as Parquet under data_cache/ for fast reuse.

Run:  .venv/Scripts/python.exe scripts/eda.py
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "student_resource" / "dataset"
CACHE = ROOT / "data_cache"
CACHE.mkdir(exist_ok=True)
REPORT = ROOT / "eda_report.txt"

_out = open(REPORT, "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    _out.write(msg + "\n")


def load(split: str, name: str) -> pl.DataFrame:
    """Read a TSV as all-string columns; cache as Parquet."""
    pq = CACHE / f"{split}_{name}.parquet"
    if pq.exists():
        return pl.read_parquet(pq)
    df = pl.read_csv(
        DS / split / f"{split}_{name}.tsv",
        separator="\t",
        quote_char=None,      # addresses contain quotes; never treat them specially
        infer_schema=False,   # everything as String
    )
    df.write_parquet(pq)
    return df


def norm_expr(col: str) -> pl.Expr:
    """Light normalization: lowercase, non-letter/digit -> space, squeeze."""
    return (
        pl.col(col).fill_null("").str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )


def empty_rate(df: pl.DataFrame, col: str) -> float:
    return df.select(
        (pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")).mean()
    ).item()


t0 = time.time()
log("=" * 70)
log("LOAD + INTEGRITY")
log("=" * 70)
data = {}
for split in ("train", "test"):
    for name in ("source1", "source2", "source3"):
        t = time.time()
        df = load(split, name)
        data[(split, name)] = df
        dup = df.height - df["entity_id"].n_unique()
        log(f"{split}_{name}: {df.height:>9,} rows | cols={df.columns} | "
            f"dup ids={dup} | empty name={empty_rate(df,'business_name'):.2%} "
            f"| empty addr={empty_rate(df,'business_address'):.2%} | {time.time()-t:.1f}s")
gt = load("train", "ground_truth")
log(f"train_ground_truth: {gt.height:,} rows | cols={gt.columns}")

log("\n" + "=" * 70)
log("COUNTRY MIX")
log("=" * 70)
for (split, name), df in data.items():
    vc = df["country"].value_counts().sort("count", descending=True)
    parts = ", ".join(f"{r['country']}={r['count']:,}" for r in vc.iter_rows(named=True))
    log(f"{split}_{name}: {parts}")

log("\n" + "=" * 70)
log("GROUND TRUTH STRUCTURE (train)")
log("=" * 70)
s1_ids = set(data[("train", "source1")]["entity_id"].to_list())
gt_ids = gt["source1_entity_id"]
log(f"GT rows == train S1 rows: {gt.height == len(s1_ids)} | "
    f"GT ids all in S1: {gt_ids.is_in(list(s1_ids)).all()} | dup GT ids: {gt.height - gt_ids.n_unique()}")

pairs = (
    gt.with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"})
    .filter(pl.col("mid").is_not_null() & (pl.col("mid").str.strip_chars() != ""))
    .with_columns(pl.col("mid").str.strip_chars())
    .with_columns(pl.col("mid").str.slice(0, 2).alias("src"))
)
log(f"true pairs: {pairs.height:,}  (S2: {(pairs['src']=='S2').sum():,} | S3: {(pairs['src']=='S3').sum():,})")

per = (
    pairs.group_by("s1").agg(
        pl.len().alias("n"),
        (pl.col("src") == "S2").sum().alias("n2"),
        (pl.col("src") == "S3").sum().alias("n3"),
    )
)
all_s1 = gt.select(pl.col("source1_entity_id").alias("s1"))
per = all_s1.join(per, on="s1", how="left").fill_null(0)
singletons = (per["n"] == 0).sum()
log(f"singletons (no match): {singletons:,} / {per.height:,} = {singletons/per.height:.2%}")
log(f"mean matches per S1: {per['n'].mean():.2f} | median: {per['n'].median()} | max: {per['n'].max()}")
log(f"mean S2 per S1: {per['n2'].mean():.2f} | mean S3 per S1: {per['n3'].mean():.2f}")
dist = (per.with_columns(pl.when(pl.col("n") >= 10).then(10).otherwise(pl.col("n")).alias("b"))
        ["b"].value_counts().sort("b"))
log("match-list size distribution (10 = 10+):")
for r in dist.iter_rows(named=True):
    log(f"   {r['b']:>2}: {r['count']:>9,}  ({r['count']/per.height:.2%})")

# Does any S2/S3 record belong to more than one S1 entity?
multi = pairs.group_by("mid").len().filter(pl.col("len") > 1).height
log(f"S2/S3 ids matched to >1 S1 entity: {multi:,}")

# Existence + distractor rate
for name, src in (("source2", "S2"), ("source3", "S3")):
    ids = data[("train", name)]["entity_id"]
    matched = pairs.filter(pl.col("src") == src)["mid"]
    exists = matched.is_in(ids).mean()
    cov = ids.is_in(matched).mean()
    log(f"{src}: matched ids that exist in train_{name}: {exists:.2%} | "
        f"fraction of train_{name} records matched to some S1: {cov:.2%} "
        f"(=> distractors {1-cov:.2%})")

log("\n" + "=" * 70)
log("PER-COUNTRY (train S1)")
log("=" * 70)
s1c = data[("train", "source1")].select(pl.col("entity_id").alias("s1"), "country")
pc = per.join(s1c, on="s1").group_by("country").agg(
    pl.len().alias("entities"),
    (pl.col("n") == 0).mean().alias("singleton_rate"),
    pl.col("n").mean().alias("mean_matches"),
).sort("entities", descending=True)
for r in pc.iter_rows(named=True):
    log(f"{r['country']}: {r['entities']:,} entities | singleton rate {r['singleton_rate']:.2%} "
        f"| mean matches {r['mean_matches']:.2f}")

log("\n" + "=" * 70)
log("HOW HARD ARE TRUE PAIRS?  (name/address agreement on true matches)")
log("=" * 70)
s1 = data[("train", "source1")].select(
    pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
    pl.col("business_address").alias("a1"), pl.col("country").alias("c1"))
other = pl.concat([data[("train", "source2")], data[("train", "source3")]]).select(
    pl.col("entity_id").alias("mid"), pl.col("business_name").alias("n2"),
    pl.col("business_address").alias("a2"), pl.col("country").alias("c2"))
tp = pairs.join(s1, on="s1").join(other, on="mid")
tp = tp.with_columns(norm_expr("n1").alias("nn1"), norm_expr("n2").alias("nn2"))
tp = tp.with_columns(
    pl.col("nn1").str.split(" ").list.sort().list.join(" ").alias("ts1"),
    pl.col("nn2").str.split(" ").list.sort().list.join(" ").alias("ts2"),
)
agg = tp.group_by("src").agg(
    (pl.col("n1") == pl.col("n2")).mean().alias("raw_name_eq"),
    (pl.col("nn1") == pl.col("nn2")).mean().alias("norm_name_eq"),
    (pl.col("ts1") == pl.col("ts2")).mean().alias("tokenset_name_eq"),
    (pl.col("c1") == pl.col("c2")).mean().alias("country_eq"),
    (pl.col("a2").is_null() | (pl.col("a2").str.strip_chars() == "")).mean().alias("match_addr_empty"),
).sort("src")
for r in agg.iter_rows(named=True):
    log(f"{r['src']}: raw name equal {r['raw_name_eq']:.2%} | normalized equal {r['norm_name_eq']:.2%} "
        f"| token-set equal {r['tokenset_name_eq']:.2%} | same country {r['country_eq']:.2%} "
        f"| matched record has empty address {r['match_addr_empty']:.2%}")

# Ambiguity: how many distinct S1 entities share a normalized name (within country)?
s1n = data[("train", "source1")].with_columns(norm_expr("business_name").alias("nn"))
grp = s1n.group_by(["country", "nn"]).len()
amb = grp.filter(pl.col("len") > 1)
log(f"S1 normalized-name collisions (same country): {amb.height:,} names shared by "
    f"{amb['len'].sum():,} S1 entities ({amb['len'].sum()/s1n.height:.2%} of S1)")
top = amb.sort("len", descending=True).head(8)
log("most common shared S1 names: " + "; ".join(f"'{r['nn']}' x{r['len']} ({r['country']})"
                                              for r in top.iter_rows(named=True)))

log("\n" + "=" * 70)
log("SAMPLES")
log("=" * 70)
s1_full = data[("train", "source1")]
lookup = pl.concat([data[("train", "source2")], data[("train", "source3")]])
for country in ("US", "India"):
    ids = (per.join(s1c, on="s1").filter((pl.col("country") == country) & (pl.col("n") >= 2))
           .sample(3, seed=7)["s1"].to_list())
    for sid in ids:
        r = s1_full.filter(pl.col("entity_id") == sid).row(0, named=True)
        log(f"\n[{country}] {sid} | {r['business_name']} | {r['business_address']}")
        mids = pairs.filter(pl.col("s1") == sid)["mid"].to_list()
        for m in lookup.filter(pl.col("entity_id").is_in(mids)).iter_rows(named=True):
            log(f"     -> {m['entity_id']} | {m['business_name']} | {m['business_address']} | {m['country']}")
sing = per.filter(pl.col("n") == 0).sample(3, seed=7)["s1"].to_list()
log("\n[singletons]")
for r in s1_full.filter(pl.col("entity_id").is_in(sing)).iter_rows(named=True):
    log(f"     {r['entity_id']} | {r['business_name']} | {r['business_address']} | {r['country']}")
fr = data[("test", "source1")].filter(pl.col("country") == "France")
log(f"\n[test France S1 samples] ({fr.height:,} France entities in test S1)")
for r in fr.sample(min(5, fr.height), seed=7).iter_rows(named=True):
    log(f"     {r['entity_id']} | {r['business_name']} | {r['business_address']}")

log(f"\nEDA done in {time.time()-t0:.0f}s. Report: {REPORT}")
_out.close()
