"""
v3 pair features for the ML judge, computed in chunks over v2 candidates.

Deliberately language-agnostic: `country` is NOT a feature (France has no
training data), only similarities, rarities and structure.

Per pair
  name   : RapidFuzz ratio / token_set / token_sort / partial on v2-normalized names,
           Jaro-Winkler on compact keys, ratio on raw lowercase names,
           name-word Jaccard, rarest shared name word (document frequency),
           length and word-count differences, which name keys matched
  address: token_set on cleaned addresses, ratio on sorted canonical address
           words, address-word Jaccard, rarest shared address word, empty address
  numbers: shared house number, token_set / partial ratio of the number lists
           (catches dropped / changed digits), number counts, one side empty
  context: which v2 blocking keys fired, cheap score and its rank for the S1
           entity, gap to that entity's best candidate, rank of this S1 among
           the record's candidates, how many S1 entities compete for the record,
           gap to the record's best; record is S2 or S3; record name non-Latin
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocking_v2 import FLAGS, _freq
from normalize import clean_text

FEATURES = [
    "name_ratio", "name_tset", "name_tsort", "name_partial", "cp_jw", "raw_ratio",
    "tok_jac", "tok_min_df", "len_diff", "ntok_diff", "k_ts", "k_cp", "k_core",
    "addr_tset", "addr_ratio", "addr_jac", "aw_min_df", "a_empty",
    "num_sh", "num_tset", "num_partial", "n_nums1", "n_nums2", "num_one_empty",
    "k_nt", "k_na", "k_ta", "score", "rank", "gap_s1", "rank_r", "r_cands", "gap_r",
    "is_s3", "non_latin",
]
LATIN_ONLY = r"^[\p{Latin}\p{Common}\p{Inherited}]*$"
_plain = lambda col: clean_text(pl.col(col)).str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def record_table(prepared: pl.DataFrame, raw: pl.DataFrame, n_s2: int = None) -> pl.DataFrame:
    """Per-record strings and lists used by the pair features (same row order as `prepared`)."""
    base = prepared.select(
        "i", "country", "toks", "awords", "nums",
        pl.col("toks").list.join(" ").alias("name_n"),
        pl.col("key_cp").alias("cp"),
        pl.col("awords").list.sort().list.join(" ").alias("aw_s"),
        pl.col("nums").list.sort().list.join(" ").alias("num_s"),
    )
    extra = raw.select(
        _plain("business_name").alias("raw_n"),
        _plain("business_address").alias("addr_n"),
        (~pl.col("business_name").fill_null("").str.contains(LATIN_ONLY)).alias("non_latin"),
    )
    t = pl.concat([base, extra], how="horizontal")
    if n_s2 is not None:
        t = t.with_columns((pl.col("i") >= n_s2).alias("is_s3"))
    return t


def add_context(cand: pl.DataFrame) -> pl.DataFrame:
    """Competition features that need every candidate of a record (all S1 chunks)."""
    return cand.with_columns(
        (pl.col("score").max().over("i1") - pl.col("score")).alias("gap_s1"),
        (pl.col("score").max().over("i2") - pl.col("score")).alias("gap_r"),
        pl.col("score").rank("min", descending=True).over("i2").cast(pl.UInt32).alias("rank_r"),
        pl.len().over("i2").cast(pl.UInt32).alias("r_cands"),
    )


def _min_df(pairs: pl.DataFrame, col1: str, col2: str, freq: pl.DataFrame, name: str) -> pl.DataFrame:
    """Document frequency of the rarest shared element (null when nothing is shared)."""
    f = freq.rename({freq.columns[1]: "w"})
    sh = (pairs.select("row", "country", pl.col(col1).list.set_intersection(col2).alias("w"))
          .explode("w").drop_nulls()
          .join(f, on=["country", "w"])
          .group_by("row").agg(pl.col("df").min().alias(name)))
    return pairs.join(sh, on="row", how="left")


def chunk_features(c: pl.DataFrame, S: pl.DataFrame, O: pl.DataFrame, tok_f: pl.DataFrame,
                   aw_f: pl.DataFrame) -> pl.DataFrame:
    s = S.rename({k: f"{k}1" for k in S.columns if k not in ("i", "country")}).rename({"i": "i1"})
    o = O.rename({k: f"{k}2" for k in O.columns if k != "i"}).rename({"i": "i2"}).drop("country2")
    p = c.join(s, on="i1").join(o, on="i2").with_row_index("row")
    p = _min_df(p, "toks1", "toks2", tok_f, "tok_min_df")
    p = _min_df(p, "awords1", "awords2", aw_f, "aw_min_df").sort("row")
    L = {col: p[col].to_list() for col in
         ("name_n1", "name_n2", "cp1", "cp2", "raw_n1", "raw_n2", "addr_n1", "addr_n2",
          "aw_s1", "aw_s2", "num_s1", "num_s2")}
    rf = lambda scorer, a, b: process.cpdist(L[a], L[b], scorer=scorer, workers=-1, dtype=np.float32)
    feats = {
        "name_ratio": rf(fuzz.ratio, "name_n1", "name_n2"),
        "name_tset": rf(fuzz.token_set_ratio, "name_n1", "name_n2"),
        "name_tsort": rf(fuzz.token_sort_ratio, "name_n1", "name_n2"),
        "name_partial": rf(fuzz.partial_ratio, "name_n1", "name_n2"),
        "cp_jw": rf(JaroWinkler.normalized_similarity, "cp1", "cp2"),
        "raw_ratio": rf(fuzz.ratio, "raw_n1", "raw_n2"),
        "addr_tset": rf(fuzz.token_set_ratio, "addr_n1", "addr_n2"),
        "addr_ratio": rf(fuzz.ratio, "aw_s1", "aw_s2"),
        "num_tset": rf(fuzz.token_set_ratio, "num_s1", "num_s2"),
        "num_partial": rf(fuzz.partial_ratio, "num_s1", "num_s2"),
    }
    del L
    k = pl.col("keys")
    out = p.select(
        "i1", "i2",
        *[pl.Series(n, v) for n, v in feats.items()],
        "tok_jac", "addr_jac", "score", "rank", "gap_s1", "rank_r", "r_cands", "gap_r",
        pl.col("tok_min_df").fill_null(10**7),
        pl.col("aw_min_df").fill_null(10**7),
        (pl.col("name_n1").str.len_chars().cast(pl.Int32) - pl.col("name_n2").str.len_chars().cast(pl.Int32)).abs().alias("len_diff"),
        (pl.col("toks1").list.len().cast(pl.Int32) - pl.col("toks2").list.len().cast(pl.Int32)).abs().alias("ntok_diff"),
        ((k & FLAGS["TS"]) > 0).alias("k_ts"), ((k & FLAGS["CP"]) > 0).alias("k_cp"),
        ((k & FLAGS["CORE"]) > 0).alias("k_core"), ((k & FLAGS["NT"]) > 0).alias("k_nt"),
        ((k & FLAGS["NA"]) > 0).alias("k_na"), ((k & FLAGS["TA"]) > 0).alias("k_ta"),
        ((pl.col("addr_n1") == "") | (pl.col("addr_n2") == "")).alias("a_empty"),
        "num_sh",
        pl.col("nums1").list.len().alias("n_nums1"),
        pl.col("nums2").list.len().alias("n_nums2"),
        ((pl.col("nums1").list.len() == 0) | (pl.col("nums2").list.len() == 0)).alias("num_one_empty"),
        pl.col("is_s32").alias("is_s3"), pl.col("non_latin2").alias("non_latin"),
    )
    return out.with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES])


def build_features(cand: pl.DataFrame, s1p: pl.DataFrame, s1raw: pl.DataFrame, othp: pl.DataFrame,
                   othraw: pl.DataFrame, n_s2: int, out_dir, chunk: int = 100_000) -> None:
    """Write feature parquet parts (one per S1 chunk) to out_dir."""
    t0 = time.time()
    tok_f, aw_f = _freq(s1p, othp, "toks"), _freq(s1p, othp, "awords")
    S = record_table(s1p, s1raw)
    O = record_table(othp, othraw, n_s2=n_s2)
    cand = add_context(cand)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = s1p.height
    for part, start in enumerate(range(0, n, chunk)):
        c = cand.filter((pl.col("i1") >= start) & (pl.col("i1") < start + chunk))
        chunk_features(c, S, O, tok_f, aw_f).write_parquet(out_dir / f"part-{part:03d}.parquet")
        print(f"  features: S1 {min(start+chunk, n):>9,}/{n:,} | {c.height/1e6:.1f}M pairs | {time.time()-t0:.0f}s", flush=True)
