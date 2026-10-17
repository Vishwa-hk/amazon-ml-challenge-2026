"""
v8 = v7b judge features (twin-decoy features, +21 offset) on short-lists v8
(= short-lists v6 + a record-side name slot for no-address records, see blocking_v8.py)
and with domain-style S2/S3 names split back into words (segment.py).

v6/v7 code is reused unchanged by pointing their module globals at the v8 folders:
  runs/blocking_v8/          candidates (+ id maps)
  runs/v8/base/<split>_features   v4 features (pipeline_v6 features code)
  runs/v8/<split>_features        + NEW7 (features_v7)
  runs/v8/<tag>/                  models, OOF, test predictions

Commands (in order):
  python pipeline_v8.py blocking --split train     # prints recall / oracle (+ no-address recall)
  python pipeline_v8.py blocking --split test
  python pipeline_v8.py features --split train     # base features, then NEW7
  python pipeline_v8.py features --split test [--fr-map]
  python pipeline_v8.py train --tag <tag>
  python pipeline_v8.py test --tag <tag> --out-dir <new folder> [--rule '<json>']
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

import pipeline_v6 as v6
import pipeline_v7 as v7
from blocking_v2 import prepare
from blocking_v8 import build_candidates
from features_v7 import DECOY_OFFSETS_B, pair_features, record_numbers
from run_blocking_v3 import oracle
from segment import segment_names, word_costs

PROJECT = Path(__file__).resolve().parents[3]
BL8, RUN8 = PROJECT / "runs" / "blocking_v8", PROJECT / "runs" / "v8"
RUN8FR = PROJECT / "runs" / "v8fr"   # test features with FR_MAP; its <tag>/ holds COPIES of the runs/v8/<tag> models
_load_raw_v6 = v6.load_raw


def load_raw_v8(split: str):
    """v6 inputs (transliterated S2/S3 names) + domain-style names split back into words (segment.py)."""
    s1raw, n_s2, oth_t, native = _load_raw_v6(split)
    oth_t, n_dom = segment_names(oth_t, word_costs(s1raw))
    print(f"[{split}] {n_dom:,} domain-style S2/S3 names segmented", flush=True)
    return s1raw, n_s2, oth_t, native


v6.load_raw = load_raw_v8   # every v6/v7 step reached from this module (blocking, features, stage 2) uses v8 inputs

CUT = "{split}_candidates_top100a20.parquet"   # same file name as v6 so pipeline_v6's features code finds it


def cmd_blocking(a):
    full, cut = BL8 / f"{a.split}_candidates.parquet", BL8 / CUT.format(split=a.split)
    v6.refuse(full)
    BL8.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s1raw, _, oth_t, native = load_raw_v8(a.split)
    # memory: keep only what the report needs, free the raw S2/S3 table before the candidate build
    pl.DataFrame({"entity_id": s1raw["entity_id"]}).write_parquet(BL8 / f"{a.split}_s1_ids.parquet")
    pl.DataFrame({"entity_id": oth_t["entity_id"]}).write_parquet(BL8 / f"{a.split}_oth_ids.parquet")
    ctry = s1raw.with_row_index("i1").select("i1", "country")
    n_s1 = s1raw.height
    held = {"s1": prepare(s1raw)}
    del s1raw, native
    held["oth"] = prepare(oth_t)
    del oth_t
    na = held["oth"].select(pl.col("i").alias("i2"), ((pl.col("ah").list.len() == 0) & (pl.col("nums").list.len() == 0)).alias("na")).filter(pl.col("na"))
    # pop: the builder then holds the only references, so its rebinding (add_rarest) frees the originals
    parts_dir = BL8 / f"{a.split}_candidate_parts"
    v6.refuse(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    build_candidates(held.pop("s1"), held.pop("oth"), top_n=150, addr_n=20, chunk=50_000, parts_dir=parts_dir)
    pl.scan_parquet(parts_dir / "part-*.parquet").sink_parquet(full, row_group_size=2_000_000)
    pl.scan_parquet(full).filter((pl.col("rank") <= 100) | (pl.col("arank") <= 20) | (pl.col("nrank") <= 5))         .drop("nrank").sink_parquet(cut, row_group_size=2_000_000)
    n_cut = pl.scan_parquet(cut).select(pl.len()).collect().item()
    print(f"[{a.split}] candidates saved; cut (top-100 + address-20 + no-address name slot 5): {n_cut:,} ({time.time()-t0:.0f}s)", flush=True)
    if a.split == "train":
        v6.BL6 = BL8
        pairs = v6.truth("train")
        c = pl.scan_parquet(cut).select("i1", "i2").collect()
        r, f = oracle(c, pairs, n_s1)
        print(f"[train] recall {r:.2%} | oracle F0.5 {f:.4f}   (short-lists v6: 97.62% / 0.9927)")
        hit = pairs.join(c, on=["i1", "i2"], how="semi").join(ctry, on="i1").group_by("country").len("hit")
        tot = pairs.join(ctry, on="i1").group_by("country").len("tot")
        for row in hit.join(tot, on="country").sort("country").iter_rows(named=True):
            print(f"   {row['country']:<6} recall {row['hit']/row['tot']:.2%}")
        p_na = pairs.join(na, on="i2", how="semi")
        print(f"   no-address true pairs: {p_na.height:,} | recall {p_na.join(c, on=['i1', 'i2'], how='semi').height / p_na.height:.2%}  (v6: ~70%)")


def point_globals(fr_map: bool = False):
    v6.BL6, v6.RUN6, v6.FEATURES_V4 = BL8, (RUN8FR if fr_map else RUN8), v7.FEATURES_V7


def cmd_features(a):
    base = RUN8 / "base"
    out = (RUN8FR if a.fr_map else RUN8) / f"{a.split}_features"
    v6.refuse(out)
    if not (base / f"{a.split}_features").exists():
        v6.BL6, v6.RUN6 = BL8, base
        v6.cmd_features(a)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    s1raw, _, oth_t, _ = v6.load_raw(a.split)   # v8 inputs (segmented names); FR_MAP + dotted-form collapse for France
    S, O = record_numbers(s1raw, "i1", a.fr_map, a.fr_map), record_numbers(oth_t, "i2", a.fr_map, a.fr_map)
    del s1raw, oth_t
    vocab = pl.read_parquet(v7.VOCAB)
    src = sorted((base / f"{a.split}_features").glob("part-*.parquet"))
    for k, p in enumerate(src):
        x = pl.read_parquet(p)
        lo, hi = x["i1"].min(), x["i1"].max()
        f = pair_features(x.select("i1", "i2"), S.filter(pl.col("i1").is_between(lo, hi)), O, vocab, DECOY_OFFSETS_B)
        y = x.join(f, on=["i1", "i2"], how="left")
        assert y.height == x.height
        y.write_parquet(out / p.name)
    print(f"[{a.split}] v8 features (+NEW7{', FR_MAP' if a.fr_map else ''}) done ({time.time()-t0:.0f}s) -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blocking", "features", "train", "test"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--tag", default="gpu_slot")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--n-train-entities", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--fr-map", action="store_true", help="translate French words before the decoy-word lookup (v7c)")
    ap.add_argument("--predict-only", action="store_true", help="test: cache stage-1/2 predictions, write no submission files")
    a = ap.parse_args()
    if a.cmd == "blocking":
        return cmd_blocking(a) or 0
    if a.cmd == "features":
        return cmd_features(a) or 0
    if a.cmd == "test" and not a.out_dir and not a.predict_only:
        ap.error("test needs --out-dir (a new folder)")
    point_globals(a.fr_map)
    {"train": v6.cmd_train, "test": v6.cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
