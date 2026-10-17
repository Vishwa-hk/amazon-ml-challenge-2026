"""
v2 candidate generation (blocking). Every key is matched WITHIN the same country.

Key types (bit flag in the `keys` column):
    1  TS    word-set name key equal                 (v2 normalization)
    2  CP    compact name key equal                   ("universal india designs" == "universalindiadesigns")
    4  CORE  core name key equal (generic words removed; key frequency <= 1000)
    8  NT    share a house number AND a name word     (name-word frequency <= 5000)
   16  NA    share a house number AND an address word (address-word frequency <= 10000)
   32  TA    share a name word AND an address word    (address-word freq <= 2000, name-word freq <= 20000)
Frequencies are document frequencies per country over all three sources of the
split being processed (input text only, never labels).

Scale: Source-1 is processed in chunks so the raw key matches (hundreds of
millions of pairs on the full data) never sit in memory together. Within each
chunk every candidate gets a cheap similarity score and each S1 entity keeps its
best `top_n` candidates — no bucket is silently dropped, so every entity keeps
coverage (the failure mode of hard bucket caps).

Cheap score = 2 * name-word Jaccard + 2 * address-word Jaccard + shared house number.
"""
import time

import polars as pl

from normalize_v2 import add_features_v2

FLAGS = {"TS": 1, "CP": 2, "CORE": 4, "NT": 8, "NA": 16, "TA": 32}
LIMITS = {"CORE": 1000, "NT_tok": 5000, "NA_addr": 10000, "TA_addr": 2000, "TA_tok": 20000}


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """v2 features + integer row index + hashed token lists for fast Jaccard."""
    df = add_features_v2(df).with_row_index("i").with_columns(pl.col("toks").list.unique())
    return df.select(
        "i", "country", "key_ts", "key_cp", "key_core", "toks", "nums", "awords",
        pl.col("toks").list.eval(pl.element().hash()).alias("th"),
        pl.col("awords").list.eval(pl.element().hash()).alias("ah"),
    )


def _freq(s1: pl.DataFrame, oth: pl.DataFrame, col: str) -> pl.DataFrame:
    both = pl.concat([s1.select("country", col), oth.select("country", col)])
    if isinstance(both.schema[col], pl.List):
        both = both.explode(col).drop_nulls()
    return both.group_by("country", col).len("df")


def _keyed(df: pl.DataFrame, parts: list, keep=None) -> pl.DataFrame:
    """(i, k) rows: one per distinct key of each record; k = hash(country|parts)."""
    x = df.select(["i", "country"] + parts)
    for p in parts:
        if isinstance(x.schema[p], pl.List):
            x = x.explode(p)
    x = x.drop_nulls()
    for p in parts:
        if x.schema[p] == pl.String:
            x = x.filter(pl.col(p) != "")
    if keep is not None:
        x = keep(x)
    return x.select("i", pl.concat_str(["country"] + parts, separator="|").hash().alias("k")).unique()


def build_key_tables(s1: pl.DataFrame, oth: pl.DataFrame) -> dict:
    tf, af = _freq(s1, oth, "toks"), _freq(s1, oth, "awords")
    cf = _freq(s1, oth, "key_core")
    rare = lambda tbl, col, lim: (lambda x: x.join(tbl.filter(pl.col("df") <= lim).drop("df"), on=["country", col], how="semi"))
    specs = {
        "TS": (["key_ts"], None),
        "CP": (["key_cp"], None),
        "CORE": (["key_core"], rare(cf, "key_core", LIMITS["CORE"])),
        "NT": (["nums", "toks"], rare(tf, "toks", LIMITS["NT_tok"])),
        "NA": (["nums", "awords"], rare(af, "awords", LIMITS["NA_addr"])),
        "TA": (["toks", "awords"], lambda x: rare(tf, "toks", LIMITS["TA_tok"])(rare(af, "awords", LIMITS["TA_addr"])(x))),
    }
    return {name: (_keyed(s1, parts, keep), _keyed(oth, parts, keep)) for name, (parts, keep) in specs.items()}


def _jac(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(b).list.len()
    union = pl.col(a).list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0)


def build_candidates(s1: pl.DataFrame, oth: pl.DataFrame, top_n: int = 50, chunk: int = 100_000,
                     verbose: bool = True) -> pl.DataFrame:
    """Return candidate pairs (i1, i2, keys, tok_jac, addr_jac, num_sh, score, rank)."""
    t0 = time.time()
    tables = build_key_tables(s1, oth)
    if verbose:
        sizes = ", ".join(f"{k}: {a.height/1e6:.1f}M/{b.height/1e6:.1f}M" for k, (a, b) in tables.items())
        print(f"  key tables ({sizes}) in {time.time()-t0:.0f}s", flush=True)
    S = s1.select(pl.col("i").alias("i1"), pl.col("th").alias("th1"), pl.col("ah").alias("ah1"), pl.col("nums").alias("n1"))
    O = oth.select(pl.col("i").alias("i2"), pl.col("th").alias("th2"), pl.col("ah").alias("ah2"), pl.col("nums").alias("n2"))
    out, raw_total = [], 0
    n = s1.height
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        parts = []
        for name, (a, b) in tables.items():
            ac = a.filter((pl.col("i") >= start) & (pl.col("i") < stop))
            parts.append(ac.join(b, on="k").select(pl.col("i").alias("i1"), pl.col("i_right").alias("i2"),
                                                   pl.lit(FLAGS[name], dtype=pl.UInt8).alias("f")))
        raw = pl.concat(parts)
        raw_total += raw.height
        c = raw.group_by("i1", "i2").agg(pl.col("f").unique().sum().cast(pl.UInt8).alias("keys"))
        c = c.join(S.slice(start, stop - start), on="i1").join(O, on="i2")
        c = c.with_columns(
            _jac("th1", "th2").alias("tok_jac"),
            _jac("ah1", "ah2").alias("addr_jac"),
            (pl.col("n1").list.set_intersection("n2").list.len() > 0).alias("num_sh"),
        ).drop("th1", "th2", "ah1", "ah2", "n1", "n2")
        c = c.with_columns((2 * pl.col("tok_jac") + 2 * pl.col("addr_jac") + pl.col("num_sh").cast(pl.Float64)).alias("score"))
        c = c.with_columns(pl.col("score").rank("ordinal", descending=True).over("i1").cast(pl.UInt32).alias("rank"))
        out.append(c.filter(pl.col("rank") <= top_n))
        if verbose:
            print(f"  S1 {stop:>9,}/{n:,} | raw key matches so far {raw_total/1e6:,.0f}M | {time.time()-t0:.0f}s", flush=True)
    return pl.concat(out)
