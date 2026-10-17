"""
Domain-style name segmentation (v8): "Tristatekeystonehawk.com" -> "tristate keystone hawk".

Observation (27 Sep, v7b short-list misses): one recipe step glues the S1 name's words into
a web handle ("technologyvisionprivate.com", "@hyderabadsports", "www.siyaramind.com"),
sometimes in shuffled order. Glued, the name shares no word with its S1 entity, so these
records reached the short-lists only 92.5% of the time.

Fix: strip the web decoration and split the glued string back into words with a
unigram Viterbi segmentation over the word frequencies of the split's own S1 names
(given data only; no external dictionary).
"""
import math
import re

import polars as pl

from translit import latin_tokens

DOMAIN = r"(?i)(\.com\b|\.co\.in\b|\.in\b|\.net\b|\.org\b|^www\.|^@)"
_STRIP = re.compile(r"(?i)^(www\.|@)|(\.co\.in|\.com|\.in|\.net|\.org)$")
MAXW = 20


def word_costs(s1: pl.DataFrame) -> dict:
    """word -> -log P(word), from S1 name tokens (letters only, len >= 2)."""
    w = (s1.select(latin_tokens("business_name").alias("t")).explode("t").drop_nulls()
         .filter(pl.col("t").str.contains(r"^[a-z]{2,}$")).group_by("t").len("n"))
    total = float(w["n"].sum())
    return {t: math.log(total / n) for t, n in zip(w["t"].to_list(), w["n"].to_list())}


def split_glued(s: str, cost: dict, unk: float = 25.0) -> str:
    """Viterbi word break of a lowercase letter string; unknown single letters cost `unk`."""
    n = len(s)
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - MAXW), i):
            w = s[j:i]
            c = cost.get(w)
            if c is None:
                if i - j != 1:
                    continue
                c = unk
            if best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, j
    out, i = [], n
    while i > 0:
        out.append(s[back[i]:i])
        i = back[i]
    return " ".join(reversed(out))


def segment_names(df: pl.DataFrame, cost: dict, col: str = "business_name") -> tuple[pl.DataFrame, int]:
    """Replace domain-style names by their segmented words. Returns (df, number changed)."""
    is_dom = pl.col(col).fill_null("").str.contains(DOMAIN)
    names = df.filter(is_dom)[col].unique().to_list()
    mapping = {}
    for nm in names:
        toks = nm.replace("|", " ").split()
        out = []
        for t in toks:
            if re.search(DOMAIN, t):            # only the web token is split; other words stay as they are
                core = _STRIP.sub("", _STRIP.sub("", t.strip(".,;:()[]")).lower())
                parts = re.findall(r"[a-z]+|\d+", core)
                out.append(" ".join(split_glued(p, cost) if p.isalpha() and len(p) > 3 else p for p in parts))
            elif t.lower() != "www":
                out.append(t)
        seg = " ".join(x for x in out if x)
        if seg:
            mapping[nm] = seg
    out = df.with_columns(pl.when(is_dom).then(pl.col(col).replace(mapping)).otherwise(pl.col(col)).alias(col))
    return out, int(df.select(is_dom.sum()).item())
