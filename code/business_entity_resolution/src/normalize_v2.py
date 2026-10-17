"""
v2 normalization (v1's normalize.py is left untouched so the v1 submission stays
reproducible). Changes, each motivated by the miss analysis:

Names
  * honorific prefixes dropped: sri, shri, smt, mr, mrs, ms, dr, m/s
  * digit-for-letter lookalikes repaired inside mixed tokens (0->o 1->l 3->e 4->a
    5->s 7->t 8->b), e.g. "0ganisc" -> "oganisc", "5uperior" -> "superior"
  * if dropping legal/filler words would leave nothing (acronym names like
    "PC", "SA"), the undropped tokens are kept instead
  * "core" tokens: additionally drop generic business nouns (center, services,
    partners, group, ...) that sources add, drop or swap at random

House numbers
  * 1-6 digit numbers; 6-digit numbers are dropped ONLY for India (PIN codes).
    v1 dropped them everywhere, which broke US 6-digit house numbers.

Address words
  * letters-only tokens, street-type abbreviations unified (rd->road, st->street,
    and the generator's st->"saint" corruption), literal "null" removed.
    Frequent words (states, cities, "road"...) are filtered later by document
    frequency, so no hand-made state/city lists are needed.
"""
import polars as pl

from normalize import DROP_TOKENS, _ABBREV, _TLD, clean_text

HONORIFICS = ["sri", "shri", "smt", "mr", "mrs", "ms", "dr"]
GENERIC = [
    "center", "centre", "services", "service", "partners", "partner", "group",
    "associates", "solutions", "global", "international", "enterprises",
    "enterprise", "industries", "technologies", "technology", "tech", "trading",
    "consultants", "consultancy", "consulting", "holdings", "ventures", "products",
    "systems", "company", "brothers",
]
LOOKALIKE = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
STREET_CANON = {
    "rd": "road", "st": "street", "saint": "street", "str": "street", "dr": "drive",
    "ave": "avenue", "av": "avenue", "ln": "lane", "ct": "court", "cir": "circle",
    "pl": "place", "blvd": "boulevard", "bd": "boulevard", "trl": "trail",
    "ter": "terrace", "hwy": "highway", "pkwy": "parkway", "sq": "square",
}
ADDR_DROP = ["null", "nan", "none"]


def name_tokens_v2(col: str) -> pl.Expr:
    """Tokens before legal/filler removal (used as the acronym fallback)."""
    e = clean_text(pl.col(col))
    e = e.str.replace_all(r"https?://", " ").str.replace_all(r"\bwww\.", " ")
    e = e.str.replace_all(_TLD, " ")
    e = e.str.replace_all(r"\bm\s*/\s*s\b", " ")          # "M/s" honorific
    e = e.str.replace_all(r"['’`]", "")
    e = e.str.replace_all(r"[^\p{L}\p{N}.]+", " ")
    toks = e.str.strip_chars().str.split(" ").list.eval(
        pl.when(pl.element().str.contains(_ABBREV))
        .then(pl.element().str.replace_all(r"\.", ""))
        .otherwise(pl.element().str.replace_all(r"\.", " "))
    )
    toks = toks.list.join(" ").str.split(" ")
    return toks.list.eval(
        pl.element().filter(pl.element().str.len_chars() > 0)
        .pipe(lambda x: pl.when(x.str.contains(r"\p{L}") & x.str.contains(r"\d"))
              .then(x.str.replace_many(LOOKALIKE)).otherwise(x))
    )


def add_features_v2(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(name_tokens_v2("business_name").alias("raw_toks"))
    drop = DROP_TOKENS + HONORIFICS
    toks = pl.col("raw_toks").list.eval(pl.element().filter(~pl.element().is_in(drop)))
    df = df.with_columns(
        pl.when(toks.list.len() > 0).then(toks)
        .otherwise(pl.col("raw_toks").list.eval(pl.element().filter(~pl.element().is_in(HONORIFICS))))
        .alias("toks")
    )
    df = df.with_columns(pl.col("toks").list.eval(pl.element().filter(~pl.element().is_in(GENERIC))).alias("core"))
    nums = (clean_text(pl.col("business_address")).str.extract_all(r"\d+")
            .list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter((pl.element().str.len_chars() >= 1) & (pl.element().str.len_chars() <= 6))))
    nums = pl.when(pl.col("country") == "India").then(
        nums.list.eval(pl.element().filter(pl.element().str.len_chars() <= 5))).otherwise(nums)
    words = (clean_text(pl.col("business_address")).str.replace_all(r"[^\p{L}]+", " ").str.strip_chars()
             .str.split(" ").list.eval(pl.element().replace(STREET_CANON))
             .list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(ADDR_DROP))))
    return df.with_columns(
        pl.col("toks").list.unique().list.sort().list.join(" ").alias("key_ts"),
        pl.col("toks").list.join("").alias("key_cp"),
        pl.col("core").list.unique().list.sort().list.join(" ").alias("key_core"),
        nums.list.unique().alias("nums"),
        words.list.unique().alias("awords"),
    )
