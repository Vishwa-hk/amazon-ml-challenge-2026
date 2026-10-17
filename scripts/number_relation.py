"""
Diagnostic: how do house numbers differ in TRUE pairs vs look-alike DECOYS?
Look-alike decoy pair = a decoy S2/S3 record joined to an S1 entity with the same
word-set name key in the same country.
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from normalize import add_features, clean_text  # noqa: E402

C = ROOT / "data_cache"


def raw_nums(col):  # digit strings as written (keep leading zeros, all lengths)
    return clean_text(pl.col(col)).str.extract_all(r"\d+")


def atok(col):
    return (clean_text(pl.col(col)).str.replace_all(r"[^\p{L}]+", " ").str.strip_chars()
            .str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique())


def prep(df, sfx):
    df = add_features(df).with_columns(raw_nums("business_address").alias("rn"), atok("business_address").alias("at"))
    return df.select(pl.col("entity_id").alias(f"id{sfx}"), pl.col("country").alias(f"c{sfx}"),
                     pl.col("key_ts").alias(f"k{sfx}"), pl.col("rn").alias(f"rn{sfx}"), pl.col("at").alias(f"at{sfx}"))


s1 = prep(pl.read_parquet(C / "train_source1.parquet"), "1")
oth = prep(pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")]), "2")
gt = (pl.read_parquet(C / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
      .rename({"source1_entity_id": "id1", "matched_entity_ids": "id2"}))

true_pairs = gt.join(s1, on="id1").join(oth, on="id2")
decoys = oth.join(gt.select("id2").unique(), on="id2", how="anti")
look = (decoys.filter(pl.col("k2") != "")
        .join(s1.filter(pl.col("k1") != ""), left_on=["c2", "k2"], right_on=["c1", "k1"]))


def one_digit_drop(a, b):
    """b equals a with exactly one digit removed (anywhere)."""
    if len(b) != len(a) - 1:
        return False
    return any(a[:i] + a[i + 1:] == b for i in range(len(a)))


def relation(n1, n2):
    if not n1 or not n2:
        return "one side has no number"
    s1n, s2n = set(n1), set(n2)
    if s1n & s2n:
        return "exact same number"
    i1 = {x.lstrip("0") or "0" for x in n1}
    i2 = {x.lstrip("0") or "0" for x in n2}
    if i1 & i2:
        return "same except leading zeros"
    best = "unrelated"
    rank = {"digit dropped/added": 1, "one digit changed": 2, "off by 1-10": 3, "unrelated": 9}
    for a in i1:
        for b in i2:
            if one_digit_drop(a, b) or one_digit_drop(b, a):
                r = "digit dropped/added"
            elif len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
                r = "one digit changed"
            elif a.isdigit() and b.isdigit() and abs(int(a) - int(b)) <= 10:
                r = "off by 1-10"
            else:
                r = "unrelated"
            if rank[r] < rank[best]:
                best = r
    return best


def street_overlap(a, b):
    a, b = set(a or []), set(b or [])
    return len(a & b) / len(a | b) if a | b else 0.0


for label, df in (("TRUE pairs", true_pairs), ("LOOK-ALIKE DECOY pairs", look)):
    samp = df.sample(min(300_000, df.height), seed=5)
    rel, hi_street = Counter(), Counter()
    for n1, n2, a1, a2 in samp.select("rn1", "rn2", "at1", "at2").iter_rows():
        r = relation(n1, n2)
        rel[r] += 1
        if street_overlap(a1, a2) >= 0.5:
            hi_street[r] += 1
    print(f"\n{label}: {df.height:,} (sampled {samp.height:,})")
    for r, c in rel.most_common():
        print(f"   {r:<28} {c/samp.height:7.2%}   | of these, address words overlap >=50%: {hi_street[r]/c:6.1%}")
