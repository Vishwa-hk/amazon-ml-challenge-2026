"""
Text normalization for business names and addresses (polars expressions).

Everything here is deterministic string processing on the provided data only
(no external lookups), so it is fair-play safe and fast on ~12M rows.

Name pipeline
  1. Unicode NFKD + drop combining marks  -> "Córporation" == "Corporation"
  2. lowercase
  3. strip URL scaffolding (http://, www., .com/.in/.fr/...) so that
     "universalindiadesigns.com" becomes a name token, not junk
  4. dotted abbreviations collapse ("L.L.C." -> "llc"); other dots split words
  5. apostrophes vanish ("Orelee's" -> "orelees"); every other non letter/digit
     becomes a space (kills "--", "(INC)", "&", commas, "|", ...)
  6. drop legal-form and filler tokens (inc, llc, pvt, limited, sarl, and, ...)
     because sources add/drop them inconsistently
Keys built from the surviving tokens
  * tokenset key: sorted unique tokens joined by " "  -> immune to word order
  * compact key : tokens in original order joined by "" -> immune to spacing,
                  catches domain-style names ("universal india designs" ==
                  "universalindiadesigns")

Address pipeline (baseline only needs numbers)
  * house numbers: every run of digits, leading zeros removed, kept only if
    1-5 digits ("208-A" -> 208, "PLT NO.45"/"#45" -> 45, "570/13" -> 570, 13).
    6-digit Indian PIN codes are excluded (shared by a whole area).
"""
import polars as pl

LEGAL_TOKENS = [
    # English / US
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp",
    "lp", "ltd", "limited", "plc", "pllc", "pc",
    # India
    "pvt", "private", "opc", "pvtltd",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "cie",
]
FILLER_TOKENS = ["and", "the", "et", "of"]
DROP_TOKENS = LEGAL_TOKENS + FILLER_TOKENS

_TLD = r"\.(?:co\.in|com|net|org|in|fr|io|biz|info|us|co)\b"
_ABBREV = r"^(?:\p{L}\.)+\p{L}?\.?$"  # single letters separated by dots: l.l.c., s.a.


def clean_text(e: pl.Expr) -> pl.Expr:
    """Accent-fold and lowercase. Nulls become empty strings."""
    return (
        e.fill_null("")
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}+", "")
        .str.to_lowercase()
    )


def name_tokens(col: str) -> pl.Expr:
    """List of normalized name tokens with legal forms and fillers removed."""
    e = clean_text(pl.col(col))
    e = e.str.replace_all(r"https?://", " ").str.replace_all(r"\bwww\.", " ")
    e = e.str.replace_all(_TLD, " ")
    e = e.str.replace_all(r"['’`]", "")
    e = e.str.replace_all(r"[^\p{L}\p{N}.]+", " ")  # keep dots for the abbreviation step
    toks = e.str.strip_chars().str.split(" ").list.eval(
        pl.when(pl.element().str.contains(_ABBREV))
        .then(pl.element().str.replace_all(r"\.", ""))
        .otherwise(pl.element().str.replace_all(r"\.", " "))
    )
    toks = toks.list.join(" ").str.split(" ")
    return toks.list.eval(
        pl.element().filter(
            (pl.element().str.len_chars() > 0) & ~pl.element().is_in(DROP_TOKENS)
        )
    )


def tokenset_key(tokens: pl.Expr) -> pl.Expr:
    return tokens.list.unique().list.sort().list.join(" ")


def compact_key(tokens: pl.Expr) -> pl.Expr:
    return tokens.list.join("")


def house_numbers(col: str) -> pl.Expr:
    """Sorted unique list of 1-5 digit numbers in the address (as Int64)."""
    return (
        clean_text(pl.col(col))
        .str.extract_all(r"\d+")
        .list.eval(
            pl.element().filter(pl.element().str.len_chars() <= 7)  # avoid int overflow junk
            .cast(pl.Int64, strict=False)
        )
        .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() < 100_000)))
        .list.unique()
        .list.sort()
    )


def add_features(df: pl.DataFrame, name_col="business_name", addr_col="business_address") -> pl.DataFrame:
    """Attach tokens, both name keys, and house numbers to a source frame."""
    df = df.with_columns(name_tokens(name_col).alias("toks"))
    return df.with_columns(
        tokenset_key(pl.col("toks")).alias("key_ts"),
        compact_key(pl.col("toks")).alias("key_cp"),
        house_numbers(addr_col).alias("nums"),
    )
