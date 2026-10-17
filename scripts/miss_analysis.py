"""
Miss analysis for baseline v1 (train only). Diagnostic, not part of the submission.

Answers:
  A. Why were ~45% of true pairs never short-listed?  (category breakdown + examples)
  B. Why do some true pairs have conflicting house numbers?
  C. What are the baseline's false merges?  (wrong owner vs decoy)
  D. What are the decoys (S2/S3 records that match nothing)?
  E. Which substitutions does the noise generator use?  (mined from true pairs)
  F. Why were some singletons given matches?

Writes a full report with examples to runs/miss_analysis/report.txt.
"""
import difflib
import random
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from normalize import add_features  # noqa: E402

C = ROOT / "data_cache"
RUN = ROOT / "runs" / "baseline_v1"
OUT = ROOT / "runs" / "miss_analysis"
OUT.mkdir(parents=True, exist_ok=True)
rep = open(OUT / "report.txt", "w", encoding="utf-8")
random.seed(7)


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    rep.write(m + "\n")


def id_list(path, col):
    return (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col(col).str.split(","))
            .explode(col, empty_as_null=True).drop_nulls()
            .rename({"source1_entity_id": "s1", col: "r"}))


t0 = time.time()
s1raw = pl.read_parquet(C / "train_source1.parquet")
othraw = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")])
s1 = add_features(s1raw).select(
    pl.col("entity_id").alias("s1"), pl.col("business_name").alias("name1"),
    pl.col("business_address").alias("addr1"), "country", pl.col("toks").alias("t1"),
    pl.col("key_ts").alias("k1"), pl.col("key_cp").alias("c1"), pl.col("nums").alias("n1"))
oth = add_features(othraw).select(
    pl.col("entity_id").alias("r"), pl.col("business_name").alias("name2"),
    pl.col("business_address").alias("addr2"), pl.col("country").alias("country2"),
    pl.col("toks").alias("t2"), pl.col("key_ts").alias("k2"), pl.col("key_cp").alias("c2"),
    pl.col("nums").alias("n2"))
gt = id_list(C.parent / "student_resource/dataset/train/train_ground_truth.tsv", "matched_entity_ids")
cand = id_list(RUN / "train_candidate_pairs.tsv", "candidate_entity_ids").with_columns(pl.lit(True).alias("in_cand"))
pred = id_list(RUN / "train_matching_results.tsv", "matched_entity_ids").with_columns(pl.lit(True).alias("in_pred"))
log(f"loaded in {time.time()-t0:.0f}s | true pairs {gt.height:,} | candidates {cand.height:,} | predicted {pred.height:,}")

LATIN_ONLY = r"^[\p{Latin}\p{Common}\p{Inherited}]*$"


def pair_features(df):
    """Attach both records and relationship flags to a frame of (s1, r) pairs."""
    df = df.join(s1, on="s1").join(oth, on="r")
    inter = pl.col("t1").list.set_intersection("t2").list.len()
    return df.with_columns(
        ((pl.col("k1") == pl.col("k2")) & (pl.col("k1") != "")).alias("ts_eq"),
        ((pl.col("c1") == pl.col("c2")) & (pl.col("c1") != "")).alias("cp_eq"),
        ((pl.col("k1") == "") | (pl.col("k2") == "")).alias("empty_key"),
        (~pl.col("name2").fill_null("").str.contains(LATIN_ONLY)).alias("non_latin"),
        inter.alias("tok_shared"),
        pl.col("t1").list.unique().list.len().alias("nt1"),
        pl.col("t2").list.unique().list.len().alias("nt2"),
        pl.col("n1").list.set_intersection("n2").list.len().alias("num_shared"),
        ((pl.col("n1").list.len() == 0) | (pl.col("n2").list.len() == 0)).alias("num_one_empty"),
    )


def examples(df, k=12, cols=("name1", "addr1", "name2", "addr2")):
    rows = df.sample(min(k, df.height), seed=7).select(cols).iter_rows()
    for r in rows:
        log(f"      S1: {r[0]}  |  {r[1]}")
        log(f"      ->  {r[2]}  |  {r[3]}")
        log("")


# ---------------------------------------------------------------- A + B
log("\n" + "=" * 78 + "\nA. WHY WERE TRUE PAIRS MISSED BY THE SHORT-LISTS?\n" + "=" * 78)
tp = gt.join(cand, on=["s1", "r"], how="left").with_columns(pl.col("in_cand").fill_null(False))
miss = pair_features(tp.filter(~pl.col("in_cand")))
n_true = gt.height
log(f"missed true pairs: {miss.height:,} of {n_true:,} ({miss.height/n_true:.2%})")

cat = (
    pl.when((pl.col("ts_eq") | pl.col("cp_eq"))).then(pl.lit("1 name matched, numbers conflict"))
    .when(pl.col("empty_key")).then(pl.lit("2 empty name key"))
    .when(pl.col("non_latin")).then(pl.lit("3 native-script name"))
    .when(pl.col("tok_shared") == 0).then(pl.lit("4 no shared name word"))
    .when((pl.col("tok_shared") == pl.col("nt2")) | (pl.col("tok_shared") == pl.col("nt1")))
    .then(pl.lit("5 words dropped/added (subset)"))
    .when((pl.col("nt1") == pl.col("nt2")) & (pl.col("tok_shared") == pl.col("nt1") - 1))
    .then(pl.lit("6 one word different (typo/substitution)"))
    .otherwise(pl.lit("7 partial word overlap"))
)
miss = miss.with_columns(cat.alias("cat"))
addr_rescue = (pl.col("num_shared") >= 1)
summ = miss.group_by("cat").agg(
    pl.len().alias("n"),
    addr_rescue.mean().alias("share_number"),
    pl.col("num_one_empty").mean().alias("one_side_no_number"),
).sort("cat")
for r in summ.iter_rows(named=True):
    log(f"  {r['cat']:<44} {r['n']:>9,}  ({r['n']/miss.height:6.2%} of misses, {r['n']/n_true:6.2%} of all true) "
        f"| share a house no. {r['share_number']:.1%} | one side no no. {r['one_side_no_number']:.1%}")
by_src = miss.with_columns(pl.col("r").str.slice(0, 2).alias("src")).group_by(["src", "country"]).len().sort(["src", "country"])
log("  misses by source/country: " + ", ".join(f"{r['src']}/{r['country']}={r['len']:,}" for r in by_src.iter_rows(named=True)))
for c in summ["cat"].to_list():
    log(f"\n  --- examples: {c}")
    examples(miss.filter(pl.col("cat") == c), k=10)

log("\n" + "=" * 78 + "\nB. TRUE PAIRS WITH CONFLICTING HOUSE NUMBERS (all true pairs)\n" + "=" * 78)
allt = pair_features(gt)
conf = allt.filter((pl.col("num_shared") == 0) & ~pl.col("num_one_empty"))
log(f"true pairs with conflicting numbers: {conf.height:,} ({conf.height/n_true:.2%})")
conf_s = conf.sample(min(50000, conf.height), seed=7).with_columns(
    pl.col("n1").list.len().alias("len1"), pl.col("n2").list.len().alias("len2"))
# digit-level closeness of the numbers
close = 0
kinds = Counter()
for a, b in conf_s.select("n1", "n2").iter_rows():
    sa, sb = [str(x) for x in a], [str(x) for x in b]
    best = max(difflib.SequenceMatcher(None, x, y).ratio() for x in sa for y in sb)
    close += best >= 0.6
    if any(x in y or y in x for x in sa for y in sb):
        kinds["one number contains the other (e.g. 45 vs 145)"] += 1
    elif any(sorted(x) == sorted(y) for x in sa for y in sb):
        kinds["same digits reordered"] += 1
    elif best >= 0.6:
        kinds["similar digits (typo-like)"] += 1
    else:
        kinds["unrelated numbers"] += 1
log(f"  in a sample of {conf_s.height:,}: numbers look similar (ratio>=0.6) in {close/conf_s.height:.1%}")
for k, v in kinds.most_common():
    log(f"    {k}: {v/conf_s.height:.1%}")
log("  examples:")
examples(conf, k=15)

# ---------------------------------------------------------------- C
log("\n" + "=" * 78 + "\nC. BASELINE FALSE MERGES\n" + "=" * 78)
fp = pred.join(gt.with_columns(pl.lit(True).alias("is_true")), on=["s1", "r"], how="left") \
         .filter(pl.col("is_true").is_null())
owner = gt.rename({"s1": "true_owner"})
fp = fp.join(owner, on="r", how="left")
n_decoy = fp["true_owner"].is_null().sum()
log(f"false merges: {fp.height:,} of {pred.height:,} predicted ({fp.height/pred.height:.2%})")
log(f"  record is a DECOY (belongs to no S1): {n_decoy:,} ({n_decoy/fp.height:.1%})")
log(f"  record belongs to a DIFFERENT S1 (wrong owner): {fp.height-n_decoy:,} ({(fp.height-n_decoy)/fp.height:.1%})")
fpf = pair_features(fp.select("s1", "r", "true_owner"))
log("  examples (decoys):")
examples(fpf.filter(pl.col("true_owner").is_null()), k=10)
wo = fpf.filter(pl.col("true_owner").is_not_null()).join(
    s1.select(pl.col("s1").alias("true_owner"), pl.col("name1").alias("owner_name"), pl.col("addr1").alias("owner_addr")),
    on="true_owner")
log("  examples (wrong owner) — predicted S1 / record / true owner:")
for r in wo.sample(min(10, wo.height), seed=7).select("name1", "addr1", "name2", "addr2", "owner_name", "owner_addr").iter_rows():
    log(f"      pred S1 : {r[0]}  |  {r[1]}")
    log(f"      record  : {r[2]}  |  {r[3]}")
    log(f"      true S1 : {r[4]}  |  {r[5]}\n")

# ---------------------------------------------------------------- D
log("\n" + "=" * 78 + "\nD. WHAT ARE THE DECOYS?\n" + "=" * 78)
matched_r = gt.select("r").unique()
dec = oth.join(matched_r, on="r", how="anti")
real = oth.join(matched_r, on="r", how="semi")
log(f"decoys: {dec.height:,} of {oth.height:,} S2/S3 records ({dec.height/oth.height:.2%})")
s1keys = s1.filter(pl.col("k1") != "").select(pl.col("country"), pl.col("k1").alias("key")).unique()
for label, df in (("decoys", dec), ("real matched records", real)):
    d = df.filter(pl.col("k2") != "").select(pl.col("country2").alias("country"), pl.col("k2").alias("key"))
    hit = d.join(s1keys, on=["country", "key"], how="semi").height
    log(f"  {label}: word-set name key equals SOME S1 name key (same country): {hit/df.height:.1%}")
# do decoys form their own clusters across S2 and S3, like real entities do?
dk = dec.filter(pl.col("k2") != "").with_columns(pl.col("r").str.slice(0, 2).alias("src"))
g = dk.group_by(["country2", "k2"]).agg(pl.len().alias("n"), pl.col("src").n_unique().alias("nsrc"))
dk = dk.join(g, on=["country2", "k2"])
log(f"  decoys whose name key is shared with another decoy: {(dk['n'] > 1).mean():.1%} "
    f"| shared across both S2 and S3: {(dk['nsrc'] == 2).mean():.1%}")
log("  random decoy examples (with the closest S1 by word-set key, if any):")
dsam = dec.sample(12, seed=11).join(
    s1.select(pl.col("k1").alias("k2"), pl.col("country").alias("country2"), "name1", "addr1"),
    on=["k2", "country2"], how="left").unique("r")
for r in dsam.select("name2", "addr2", "country2", "name1", "addr1").iter_rows():
    log(f"      decoy : {r[0]}  |  {r[1]}  ({r[2]})")
    log(f"      S1 w/ same key: {r[3]}  |  {r[4]}\n")

# ---------------------------------------------------------------- E
log("\n" + "=" * 78 + "\nE. NOISE RECIPE: TOKEN SUBSTITUTIONS IN TRUE PAIRS\n" + "=" * 78)
# raw-text tokens (lowercased, punctuation split) so we see the generator's actual edits
def raw_toks(col):
    return (pl.col(col).fill_null("").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" "))
samp = allt.sample(400_000, seed=3).with_columns(
    raw_toks("name1").alias("rn1"), raw_toks("name2").alias("rn2"),
    raw_toks("addr1").alias("ra1"), raw_toks("addr2").alias("ra2"))
for field, a, b in (("NAME", "rn1", "rn2"), ("ADDRESS", "ra1", "ra2")):
    dropped, added, subs = Counter(), Counter(), Counter()
    for x, y in samp.select(a, b).iter_rows():
        sx, sy = set(x) - {""}, set(y) - {""}
        only1, only2 = sx - sy, sy - sx
        dropped.update(only1)
        added.update(only2)
        if len(only1) == 1 and len(only2) == 1:
            subs[(next(iter(only1)), next(iter(only2)))] += 1
    log(f"\n  {field}: most common words present in S1 but MISSING from the record:")
    log("    " + ", ".join(f"{w}({c:,})" for w, c in dropped.most_common(40)))
    log(f"  {field}: most common words ADDED in the record:")
    log("    " + ", ".join(f"{w}({c:,})" for w, c in added.most_common(40)))
    log(f"  {field}: most common one-word substitutions (S1 word -> record word):")
    log("    " + ", ".join(f"{p[0]}->{p[1]}({c:,})" for p, c in subs.most_common(60)))

# ---------------------------------------------------------------- F
log("\n" + "=" * 78 + "\nF. SINGLETONS THAT WERE GIVEN MATCHES\n" + "=" * 78)
sing = pl.read_parquet(C / "train_ground_truth.parquet").filter(
    pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "")).select(
    pl.col("source1_entity_id").alias("s1"))
sp = pred.join(sing, on="s1", how="semi").join(owner, on="r", how="left")
log(f"predicted pairs on singletons: {sp.height:,} | record is a decoy: {sp['true_owner'].is_null().mean():.1%}")
examples(pair_features(sp.select("s1", "r")), k=10)

log(f"\ndone in {time.time()-t0:.0f}s — full report: {OUT/'report.txt'}")
rep.close()
