# Reverse-engineering the no-address records: for each one whose true S1 (A) has a same-named twin (B),
# does any trace point to A more often than to B?  50% = no signal.
import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl, numpy as np, re
from rapidfuzz import fuzz
from pathlib import Path
import pipeline_v6 as v6
from blocking_v2 import prepare
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
C = "data_cache/"
s1 = pl.read_parquet(C + "train_source1.parquet").with_row_index("i1")
s2 = pl.read_parquet(C + "train_source2.parquet"); n2 = s2.height
oth = pl.concat([s2, pl.read_parquet(C + "train_source3.parquet")]).with_row_index("i2")
tr = v6.truth("train")
key = prepare(s1).select(pl.col("i").alias("i1"), "country", "key_ts")
grp = key.filter(pl.col("key_ts") != "").group_by("country", "key_ts").agg(pl.col("i1")).filter(pl.col("i1").list.len() >= 2)
twin = grp.explode("i1").join(grp.rename({"i1": "grp"}), on=["country", "key_ts"]).with_columns(pl.col("grp").list.set_difference(pl.concat_list("i1")).alias("others"))
noaddr = oth.filter(pl.col("business_address").is_null() | pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None", "null", "<NULL>", "N/A"])).select("i2")
x = tr.join(noaddr, on="i2", how="semi").join(twin.select("i1", "others"), on="i1")
x = x.with_columns(pl.col("others").list.sample(n=1, seed=3).list.first().alias("b")).rename({"i1": "a"}).select("a", "b", "i2")
print(f"ambiguous no-address true records (true S1 has a same-named twin): {x.height:,}")
# siblings = the other true records of A and all true records of B
sib = tr.group_by("i1").agg(pl.col("i2").alias("recs"))
x = x.join(sib.rename({"i1": "a", "recs": "ra"}), on="a").join(sib.rename({"i1": "b", "recs": "rb"}), on="b", how="left") \
     .with_columns(pl.col("ra").list.set_difference(pl.concat_list("i2")).alias("ra"), pl.col("rb").fill_null([]))
x = x.filter(pl.col("ra").list.len() > 0)
names = dict(zip(oth["i2"].to_list(), oth["business_name"].fill_null("").to_list()))
ids = dict(zip(oth["i2"].to_list(), oth["entity_id"].str.extract(r"(\d+)$").cast(pl.Int64).to_list()))
s1n = dict(zip(s1["i1"].to_list(), s1["business_name"].fill_null("").to_list()))
tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
def case(s):
    t = re.sub(r"[^A-Za-z]", "", s)
    return "U" if t.isupper() else "L" if t.islower() else "T" if s.istitle() else "M"
res = {k: [0, 0, 0] for k in ("shared corrupted token with siblings", "raw name closer to S1", "record id closer to siblings", "same source as most siblings", "same letter-case as a sibling")}
def vote(k, sa, sb):
    res[k][0 if sa > sb else 1 if sa < sb else 2] += 1
rows = x.sample(min(x.height, 40000), seed=1).iter_rows(named=True)
for r in rows:
    rn, A, B = names[r["i2"]], s1n[r["a"]], s1n[r["b"]]
    corr = tok(rn) - tok(A) - tok(B)                                  # tokens the generator made (typos, junk)
    ta = set().union(*[tok(names[i]) for i in r["ra"]]); tb = set().union(*[tok(names[i]) for i in r["rb"]]) if r["rb"] else set()
    if corr:
        vote("shared corrupted token with siblings", len(corr & ta) > 0, len(corr & tb) > 0)
    vote("raw name closer to S1", fuzz.ratio(rn.lower(), A.lower()), fuzz.ratio(rn.lower(), B.lower()))
    if r["rb"]:
        da = min(abs(ids[r["i2"]] - ids[i]) for i in r["ra"]); db = min(abs(ids[r["i2"]] - ids[i]) for i in r["rb"])
        vote("record id closer to siblings", -da, -db)
        src = r["i2"] >= n2
        fa = np.mean([i >= n2 for i in r["ra"]]); fb = np.mean([i >= n2 for i in r["rb"]])
        vote("same source as most siblings", -abs(src - fa), -abs(src - fb))
        vote("same letter-case as a sibling", case(rn) in {case(names[i]) for i in r["ra"]}, case(rn) in {case(names[i]) for i in r["rb"]})
print(f"{'trace':<40} {'points to TRUE':>15} {'points to twin':>15} {'tie':>8}   share (excl. ties)")
for k, (a, b, t) in res.items():
    print(f"{k:<40} {a:>15,} {b:>15,} {t:>8,}   {a/max(a+b,1):.1%}")
