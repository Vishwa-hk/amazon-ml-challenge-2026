"""
Browse the TRAINING answer key in human-readable form.

Shows a Source-1 business next to every Source-2/3 record the answer key says
is the same business (training data only — the test set has no answer key).

Usage (from the project folder):
  .venv/Scripts/python.exe scripts/show_matches.py                 # 5 random businesses
  .venv/Scripts/python.exe scripts/show_matches.py -n 20           # 20 random
  .venv/Scripts/python.exe scripts/show_matches.py --id S1-965667  # one specific business
  .venv/Scripts/python.exe scripts/show_matches.py --country India --singletons
  .venv/Scripts/python.exe scripts/show_matches.py --search "pioneer research"
"""
import argparse
from pathlib import Path

import polars as pl

C = Path(__file__).resolve().parents[1] / "data_cache"

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=5, help="how many random businesses")
ap.add_argument("--id", help="a specific S1 entity id")
ap.add_argument("--country", help="US or India")
ap.add_argument("--singletons", action="store_true", help="only businesses with no matches")
ap.add_argument("--search", help="only S1 names containing this text (case-insensitive)")
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

s1 = pl.read_parquet(C / "train_source1.parquet")
gt = pl.read_parquet(C / "train_ground_truth.parquet").rename({"source1_entity_id": "entity_id"})
df = s1.join(gt, on="entity_id")
if a.id:
    df = df.filter(pl.col("entity_id") == a.id)
if a.country:
    df = df.filter(pl.col("country") == a.country)
if a.singletons:
    df = df.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
if a.search:
    df = df.filter(pl.col("business_name").str.to_lowercase().str.contains(a.search.lower(), literal=True))
if df.is_empty():
    raise SystemExit("nothing matches those filters")
pick = df if a.id else df.sample(min(a.n, df.height), seed=a.seed)

others = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")])
for r in pick.iter_rows(named=True):
    ids = [x for x in (r["matched_entity_ids"] or "").split(",") if x]
    print(f"\n{r['entity_id']} ({r['country']})  |  {r['business_name']}  |  {r['business_address']}")
    if not ids:
        print("    (no matches — singleton)")
        continue
    for m in others.filter(pl.col("entity_id").is_in(ids)).sort("entity_id").iter_rows(named=True):
        print(f"    -> {m['entity_id']}  |  {m['business_name']}  |  {m['business_address']}")
