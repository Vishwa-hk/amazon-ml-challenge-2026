"""
v7 extra pair features = the TWIN-DECOY signature found on 27 Sep (deep analysis).

Observation (train, labels; confirmed label-free on test for US, India AND France):
decoy records are copies of a real S1 business with the house number shifted by
exactly +1, +2, +3, +4, +5, +7, +9, +11 or +13 and one name word swapped/added
("Wall Biotech LLC, 3417 O Henry Blvd" -> "Wall Biotech Inc., 3418 O Henry Blvd").
True copies get different number damage (digit drops, leading zeros, suffixes,
±1/±2/±10 typos). The v4 features only knew "numbers differ", not how.

Number features (up to 4 numbers per address, compared as integers):
  off_first      first record number - first S1 number (clipped to ±60; NaN if missing)
  off_first_set  off_first is one of the decoy offsets
  off_any_set    some record number NOT in the S1 address = some S1 number + a decoy offset
  off_any_neg    some record number NOT in the S1 address = some S1 number - (1..15)
  rfirst_in_s1   the record's first number appears in the S1 address
  n_rnew         how many record numbers are absent from the S1 address
Name features (decoy-word vocabulary learned from TRAIN pairs only, see build_vocab):
  add_lo_max / add_lo_sum   decoy log-odds of words in the record name but not the S1 name
  miss_lo_max               decoy log-odds of S1 words missing from the record name
  n_added / n_missing
"""
import numpy as np
import polars as pl

from translit import latin_tokens

DECOY_OFFSETS = [1, 2, 3, 4, 5, 7, 9, 11, 13]
DECOY_OFFSETS_B = DECOY_OFFSETS + [21]   # v7b: +21 found later the same day (US 11,986 decoys vs 37 true)
# v7c: France has no training labels, so the decoy-word vocabulary (learned on US/India) mis-reads French
# words ("sas" looked like a decoy word). French legal forms and business words are translated to their
# English equivalents before the lookup — a hand-written general rule, applied to French records only.
FR_MAP = {"sarl": "llc", "eurl": "llc", "sas": "corp", "sasu": "corp", "sa": "inc", "sci": "lp", "snc": "partners",
          "groupe": "group", "holding": "holdings", "participations": "holdings", "developpement": "development",
          "fils": "sons", "et": "and", "cie": "company", "compagnie": "company", "societe": "company",
          "entreprise": "enterprises", "entreprises": "enterprises", "services": "services"}
NEW7 = ["off_first", "off_first_set", "off_any_set", "off_any_neg", "rfirst_in_s1", "n_rnew",
        "add_lo_max", "add_lo_sum", "miss_lo_max", "n_added", "n_missing"]
K = 4


def record_numbers(df: pl.DataFrame, idx: str, fr_map: bool = False, fr_dots: bool = False) -> pl.DataFrame:
    """idx, n0..n3 (first 4 integers of the address, ≤7 digits), tokens of the name
    (French records' tokens translated with FR_MAP when fr_map=True — v7c; with fr_dots=True
    French dotted legal forms "S.A.R.L." / "E.U.R.L." are first collapsed to "SARL" / "EURL" — v8fr)."""
    nums = (pl.col("business_address").fill_null("").str.extract_all(r"\d{1,7}")
            .list.eval(pl.element().cast(pl.Int64, strict=False)))
    tok = latin_tokens("business_name")
    if fr_map:
        if fr_dots:
            undotted = pl.col("business_name").fill_null("").str.replace_all(r"\b([A-Za-z])\.", "$1")
            df = df.with_columns(pl.when(pl.col("country") == "France").then(undotted)
                                 .otherwise(pl.col("business_name")).alias("_fr_name"))
            fr = latin_tokens("_fr_name")
        else:
            fr = tok
        tok = pl.when(pl.col("country") == "France").then(fr.list.eval(pl.element().replace(FR_MAP))).otherwise(tok)
    x = df.with_row_index(idx).select(idx, nums.alias("nums"), tok.list.unique().alias("tok"))
    return x.select(idx, "tok", *[pl.col("nums").list.get(j, null_on_oob=True).alias(f"n{j}") for j in range(K)])


def pair_features(c: pl.DataFrame, S: pl.DataFrame, O: pl.DataFrame, vocab: pl.DataFrame, offsets=DECOY_OFFSETS) -> pl.DataFrame:
    """c: (i1, i2) rows -> (i1, i2, NEW7). S indexed by i1, O by i2 (from record_numbers)."""
    s = S.rename({f"n{j}": f"s{j}" for j in range(K)} | {"tok": "t1"})
    o = O.rename({f"n{j}": f"r{j}" for j in range(K)} | {"tok": "t2"})
    x = c.select("i1", "i2").join(s, on="i1", how="left").join(o, on="i2", how="left")
    sc = [pl.col(f"s{j}") for j in range(K)]
    in_s1 = lambda r: pl.any_horizontal([(r == sj).fill_null(False) for sj in sc])
    new = [pl.col(f"r{j}").is_not_null() & ~in_s1(pl.col(f"r{j}")) for j in range(K)]
    any_set = pl.any_horizontal([new[j] & pl.any_horizontal([(pl.col(f"r{j}") - sj).is_in(offsets).fill_null(False) for sj in sc])
                                 for j in range(K)])
    any_neg = pl.any_horizontal([new[j] & pl.any_horizontal([((pl.col(f"r{j}") - sj) < 0).fill_null(False) & ((pl.col(f"r{j}") - sj) >= -15).fill_null(False) for sj in sc])
                                 for j in range(K)])
    d = pl.col("r0") - pl.col("s0")
    x = x.with_columns(
        d.clip(-60, 60).cast(pl.Float32).alias("off_first"),
        pl.when(d.is_null()).then(None).otherwise(d.is_in(offsets)).cast(pl.Float32).alias("off_first_set"),
        any_set.cast(pl.Float32).alias("off_any_set"),
        any_neg.cast(pl.Float32).alias("off_any_neg"),
        pl.when(pl.col("r0").is_null()).then(None).otherwise(in_s1(pl.col("r0"))).cast(pl.Float32).alias("rfirst_in_s1"),
        pl.sum_horizontal([n.cast(pl.Int32) for n in new]).cast(pl.Float32).alias("n_rnew"),
        pl.col("t2").list.set_difference(pl.col("t1")).alias("added"),
        pl.col("t1").list.set_difference(pl.col("t2")).alias("missing"),
    )
    lo = vocab.select(pl.col("tok"), pl.col("lo"))
    add = (x.select("i1", "i2", "added").explode("added").drop_nulls().join(lo.rename({"tok": "added"}), on="added")
           .group_by("i1", "i2").agg(pl.col("lo").max().alias("add_lo_max"), pl.col("lo").sum().alias("add_lo_sum")))
    mis = (x.select("i1", "i2", "missing").explode("missing").drop_nulls().join(lo.rename({"tok": "missing"}), on="missing")
           .group_by("i1", "i2").agg(pl.col("lo").max().alias("miss_lo_max")))
    x = (x.join(add, on=["i1", "i2"], how="left").join(mis, on=["i1", "i2"], how="left")
         .with_columns(pl.col("added").list.len().cast(pl.Float32).alias("n_added"),
                       pl.col("missing").list.len().cast(pl.Float32).alias("n_missing")))
    return x.select("i1", "i2", *[pl.col(f).cast(pl.Float32) for f in NEW7])


def build_vocab(pairs: pl.DataFrame, S: pl.DataFrame, O: pl.DataFrame, min_n: int = 30, a: float = 5.0) -> pl.DataFrame:
    """pairs: (i1, i2, y) confusable train candidate pairs. Word -> log-odds of 'this pair is NOT a match'
    when the word is added to / missing from the record name (both directions pooled)."""
    x = pairs.join(S.select("i1", pl.col("tok").alias("t1")), on="i1").join(O.select("i2", pl.col("tok").alias("t2")), on="i2")
    x = x.select("y", pl.concat_list(pl.col("t2").list.set_difference(pl.col("t1")),
                                     pl.col("t1").list.set_difference(pl.col("t2"))).alias("tok"))
    n_pos, n_neg = int(x["y"].sum()), int((~x["y"]).sum())
    v = x.explode("tok").drop_nulls().group_by("tok").agg(pl.col("y").sum().alias("pos"), (~pl.col("y")).sum().alias("neg"))
    v = v.filter(pl.col("pos") + pl.col("neg") >= min_n).with_columns(
        (((pl.col("neg") + a) / n_neg).log() - ((pl.col("pos") + a) / n_pos).log()).cast(pl.Float32).alias("lo"))
    return v.sort("lo", descending=True)
