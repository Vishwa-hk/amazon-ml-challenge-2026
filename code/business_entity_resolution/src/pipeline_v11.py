"""
v11 inputs = v8 inputs (transliterated + segmented names) + French address normalisation
(normalize_fr.py: department -> region learned from the data, French street types unified).

France has no training rows, so train artefacts and every trained model are v8's, unchanged.
Only the TEST side is rebuilt, into new folders:
  runs/blocking_v11/   test short-lists            runs/v11/base/   test v4 features
  runs/v11fr/          test features (+NEW7, FR_MAP) and runs/v11fr/gpu_slot/ (copies of the v8 models,
                       stage-1/2 test predictions, stage-2 evidence)

Commands (the same as pipeline_v8, test split only):
  python pipeline_v11.py blocking --split test
  python pipeline_v11.py features --split test --fr-map
  copy runs/v8/gpu_slot/{stage*_fold*.ubj,decision.json} -> runs/v11fr/gpu_slot/
  python pipeline_v11.py test --fr-map --tag gpu_slot --out-dir <new folder>
"""
import sys
from pathlib import Path

import pipeline_v6 as v6
import pipeline_v8 as p8
from normalize_fr import learn_department_map, normalise_fr

PROJECT = Path(__file__).resolve().parents[3]
_load_raw_v8 = p8.load_raw_v8


def load_raw_v11(split: str):
    s1raw, n_s2, oth_t, native = _load_raw_v8(split)
    dmap = learn_department_map(s1raw, oth_t)
    print(f"[{split}] French department -> region map learned from the data: {dmap}", flush=True)
    return normalise_fr(s1raw, dmap), n_s2, normalise_fr(oth_t, dmap), native


p8.load_raw_v8 = load_raw_v11          # blocking
v6.load_raw = load_raw_v11             # features, NEW7 tables, stage-2 evidence
BL8_TEST_CUT = p8.BL8 / "test_candidates_top100a20.parquet"   # v8's test short-lists (US/India reused)
p8.BL8 = PROJECT / "runs" / "blocking_v11"
p8.RUN8 = PROJECT / "runs" / "v11"
p8.RUN8FR = PROJECT / "runs" / "v11fr"


def cmd_blocking_france(a):
    """Blocking keys are per country, so only FRANCE is re-blocked (on the normalised addresses);
    US/India short-lists are copied from v8's test cut. Uses a fraction of the full run's memory
    (the full test run exhausted RAM on 27 Sep ~15:56)."""
    import polars as pl
    from blocking_v2 import prepare
    from blocking_v8 import build_candidates
    if a.split != "test":
        raise SystemExit("v11 blocking is test-only")
    bl = p8.BL8
    cut, fr_full = bl / "test_candidates_top100a20.parquet", bl / "test_candidates_france.parquet"
    v6.refuse(cut)
    bl.mkdir(parents=True, exist_ok=True)
    s1raw, _, oth_t, _ = load_raw_v11("test")
    pl.DataFrame({"entity_id": s1raw["entity_id"]}).write_parquet(bl / "test_s1_ids.parquet")
    pl.DataFrame({"entity_id": oth_t["entity_id"]}).write_parquet(bl / "test_oth_ids.parquet")
    held = {"s1": prepare(s1raw).filter(pl.col("country") == "France")}
    del s1raw
    held["oth"] = prepare(oth_t).filter(pl.col("country") == "France")
    del oth_t
    fr_s1 = held["s1"].select(pl.col("i").alias("i1"))
    parts = bl / "test_candidate_parts_france"
    v6.refuse(parts)
    parts.mkdir(parents=True, exist_ok=True)
    build_candidates(held.pop("s1"), held.pop("oth"), top_n=150, addr_n=20, chunk=50_000, parts_dir=parts)
    pl.scan_parquet(parts / "part-*.parquet").sink_parquet(fr_full, row_group_size=2_000_000)
    fr_cut = pl.scan_parquet(fr_full).filter((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5)).drop("nrank")
    rest = pl.scan_parquet(BL8_TEST_CUT).join(fr_s1.lazy(), on="i1", how="anti")
    pl.concat([rest, fr_cut]).sink_parquet(cut, row_group_size=2_000_000)
    n_fr = pl.scan_parquet(fr_full).filter((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5)).select(pl.len()).collect().item()
    n_all = pl.scan_parquet(cut).select(pl.len()).collect().item()
    n_v8_fr = pl.scan_parquet(BL8_TEST_CUT).join(fr_s1.lazy(), on="i1", how="semi").select(pl.len()).collect().item()
    print(f"[test] France re-blocked: {n_fr:,} pairs (v8 France: {n_v8_fr:,}); total short-list {n_all:,}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "blocking":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--split", default="test")
        sys.exit(cmd_blocking_france(ap.parse_args()) or 0)
    if "--split" in sys.argv and sys.argv[sys.argv.index("--split") + 1] == "train":
        raise SystemExit("v11 only rebuilds the TEST side (France has no training rows); train artefacts are v8's")
    sys.exit(p8.main())
