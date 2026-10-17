"""
Word-level transliteration (Indian scripts -> English words), learned ONLY from the
provided training data — no external data or services (fair-play safe).

Observation (miss analysis): native-script business names in S2/S3 are the English
S1 name spelled word by word in Devanagari / Bengali / Tamil / Telugu / Gujarati /
Kannada ..., in the same word order ("Alpha Food Private Limited" ->
"अल्फा फूड प्राइवेट लिमिटेड"). So for training true pairs whose record name is
non-Latin and whose word count equals the S1 name's, the words align one-to-one;
counting the aligned (native word, English word) pairs gives a dictionary.

Keep a native word's most frequent English word when it has at least MIN_COUNT
observations and that English word holds at least MIN_SHARE of them.
Applying it replaces known native words in a name with their English word and
leaves unknown words untouched.
"""
import polars as pl

from normalize import clean_text

LATIN_ONLY = r"^[\p{Latin}\p{Common}\p{Inherited}]*$"
MIN_COUNT, MIN_SHARE = 2, 0.5


def latin_tokens(col: str) -> pl.Expr:
    return (clean_text(pl.col(col)).str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
            .str.split(" ").list.eval(pl.element().filter(pl.element() != "")))


def native_tokens(col: str) -> pl.Expr:
    """Keep combining marks: Indic vowel signs are marks and are part of the word."""
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
            .str.strip_chars().str.split(" ").list.eval(pl.element().filter(pl.element() != "")))


def aligned_pairs(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1_name, rec_name) of TRUE matches -> aligned (native, latin) word pairs."""
    x = (pairs.filter(~pl.col("rec_name").fill_null("").str.contains(LATIN_ONLY)
                      & pl.col("s1_name").fill_null("").str.contains(LATIN_ONLY))
         .with_columns(native_tokens("rec_name").alias("tn"), latin_tokens("s1_name").alias("tl"))
         .filter((pl.col("tn").list.len() == pl.col("tl").list.len()) & (pl.col("tn").list.len() > 0))
         .select("tn", "tl").explode(["tn", "tl"]))
    return x.filter(~pl.col("tn").str.contains(LATIN_ONLY)).rename({"tn": "native", "tl": "latin"})


def build_dictionary(pairs: pl.DataFrame, min_count: int = MIN_COUNT, min_share: float = MIN_SHARE) -> pl.DataFrame:
    a = aligned_pairs(pairs).group_by("native", "latin").len("n")
    tot = a.group_by("native").agg(pl.col("n").sum().alias("tot"))
    best = a.sort("n", descending=True).group_by("native", maintain_order=True).first()
    return (best.join(tot, on="native").with_columns((pl.col("n") / pl.col("tot")).alias("share"))
            .filter((pl.col("tot") >= min_count) & (pl.col("share") >= min_share))
            .select("native", "latin", pl.col("tot").alias("n"), "share"))


def transliterate(df: pl.DataFrame, dictionary: pl.DataFrame, col: str = "business_name") -> pl.DataFrame:
    """Replace known native words in non-Latin names with their English word."""
    mapping = dict(zip(dictionary["native"].to_list(), dictionary["latin"].to_list()))
    is_native = ~pl.col(col).fill_null("").str.contains(LATIN_ONLY)
    translated = native_tokens(col).list.eval(pl.element().replace(mapping)).list.join(" ")
    return df.with_columns(pl.when(is_native).then(translated).otherwise(pl.col(col)).alias(col))
