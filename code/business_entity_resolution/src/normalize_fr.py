"""
French address normalisation (v11), applied to France rows only (France has no training rows,
so every trained model and every train artefact is unchanged).

Observation (label-free, test inputs): S1 addresses always end with the REGION
("Hauts-de-France", "Nouvelle-Aquitaine", "Pays de la Loire"), while 455K+ record addresses carry
the DEPARTMENT instead ("Nord", "Gironde", "Loire-Atlantique", "Pas-de-Calais") — the French
version of the US "VA" vs "Virginia" swap. Street types are also abbreviated (R., Av, Bd, All,
Imp, Rte, Ch) and numbers prefixed ("N°12", "NO 3").

The department -> region map is LEARNED from the data, not typed in:
  * a REGION is an S1 address part that co-occurs with >= 3 different other parts (its cities);
  * a DEPARTMENT is a frequent record-only part whose records mostly (>= 50%) also name an S1 city;
    it maps to the region those cities have in the S1 addresses (majority vote).
City spelling variants ("St-Nazaire", "Merignac") co-occur with regions, not cities, so they are
not mistaken for departments.
"""
import re
from collections import Counter, defaultdict

import polars as pl

STREET_FR = [
    (r"(?i)\br\.?\s+(\S)", "Rue $1"), (r"(?i)\ball\.?\s+(\S)", "Allée $1"), (r"(?i)\bimp\.?\s+(\S)", "Impasse $1"),
    (r"(?i)\brte\.?\s+(\S)", "Route $1"), (r"(?i)\bch\.?\s+(\S)", "Chemin $1"), (r"(?i)\bbd\.?\s+(\S)", "Boulevard $1"),
    (r"(?i)\bav\.?\s+(\S)", "Avenue $1"), (r"(?i)\bpl\.?\s+(\S)", "Place $1"),
    (r"(?i)\bn\s?[°o]\.?\s*(\d)", "$1"), (r"(?i)\bno\.?\s+(\d)", "$1"),
]


def _parts(addr) -> list:
    return [p.strip().lower() for p in (addr or "").split(",") if p.strip() and not re.search(r"\d", p)]


def learn_department_map(s1: pl.DataFrame, oth: pl.DataFrame, min_count: int = 1000) -> dict:
    """{department (lowercase) -> region (S1 spelling)} from France S1 + record addresses."""
    rows = [(a, _parts(a)) for a in s1.filter(pl.col("country") == "France")["business_address"].to_list()]
    cnt, co = Counter(), defaultdict(Counter)
    for _, ps in rows:
        for p in ps:
            cnt[p] += 1
            for q in ps:
                if q != p:
                    co[p][q] += 1
    regions = {p for p in cnt if sum(1 for q, k in co[p].items() if k >= max(100, 0.01 * cnt[p])) >= 3}
    spelled, city_region = {}, defaultdict(Counter)
    for a, ps in rows:
        for r_ in (p for p in ps if p in regions):
            spelled[r_] = next(x.strip() for x in a.split(",") if x.strip().lower() == r_)
            for c in (p for p in ps if p not in regions):
                city_region[c][r_] += 1
    rec_cnt, with_city, votes = Counter(), Counter(), defaultdict(Counter)
    for a in oth.filter(pl.col("country") == "France")["business_address"].to_list():
        ps = _parts(a)
        cities = [c for c in ps if c in city_region]
        for p in ps:
            if p not in cnt:
                rec_cnt[p] += 1
                if cities:
                    with_city[p] += 1
                    for c in cities:
                        votes[p][city_region[c].most_common(1)[0][0]] += 1
    return {p: spelled[votes[p].most_common(1)[0][0]] for p, n in rec_cnt.items()
            if n >= min_count and with_city[p] >= 0.5 * n}


def normalise_fr(df: pl.DataFrame, dmap: dict) -> pl.DataFrame:
    """Rewrite France business_address strings: department -> region, French street types unified."""
    a = pl.col("business_address").fill_null("")
    for pat, rep in STREET_FR:
        a = a.str.replace_all(pat, rep)
    keys, vals = list(dmap), list(dmap.values())
    parts = a.str.split(",").list.eval(
        pl.when(pl.element().str.strip_chars().str.to_lowercase().is_in(keys))
        .then(pl.element().str.strip_chars().str.to_lowercase().replace(keys, vals))
        .otherwise(pl.element().str.strip_chars()))
    new = parts.list.join(", ")
    fr = (pl.col("country") == "France") & pl.col("business_address").is_not_null()   # missing addresses stay missing
    return df.with_columns(pl.when(fr).then(new).otherwise(pl.col("business_address")).alias("business_address"))
