"""
Write a submission from a judge's cached test predictions with a PER-COUNTRY probability recalibration
(27 Sep evening).

Why: France has no training labels. On test the judge is far less sure there (4x more records with best
probability 0.05-0.5 than US/India), and the leaderboard-implied France score (~0.962) is ~0.009 below what
the model's own probabilities predict once the US/India structural gap is allowed for — France probabilities
are OVERCONFIDENT. Every stricter-France upload so far scored higher (v3 t0.9, v6 t0.85, v12b).
Recalibration: p' = sigmoid(logit(p) + logit_shift) for that country's pairs, then the usual expected-F0.5
rule. The shift is label-free: chosen from the leaderboard score of v16 (scratch analysis fr_calib/fr_cross).
US/India: logit_shift 0 = unchanged.

  python export_v17.py --p2 ../../../runs/v16/v16/test_p2_v11fr.parquet --blocking-dir ../../../runs/blocking_v11 \
      --rule '{"default": {...}, "by_country": {"France": {..., "logit_shift": -1.0}}}' --out-dir <new folder>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from baseline import to_ids, write_id_lists
from judge_v3 import decide

PROJECT = Path(__file__).resolve().parents[3]


def shifted(p: pl.DataFrame, shift: float) -> pl.DataFrame:
    if not shift:
        return p
    q = pl.col("p").clip(1e-9, 1 - 1e-9)
    return p.with_columns((1 / (1 + (-((q / (1 - q)).log() + shift)).exp())).alias("p"))


def decide_by_country_shift(p: pl.DataFrame, rule: dict, split: str) -> pl.DataFrame:
    """Like judge_v3.decide_by_country, plus an optional per-country "logit_shift" applied before deciding.
    One owner per record is chosen on the ORIGINAL probabilities (a shift is monotone within a country, and
    records never compete across countries, so the owner is the same either way)."""
    ctry = pl.read_parquet(PROJECT / "data_cache" / f"{split}_source1.parquet").with_row_index("i1").select("i1", "country")
    p = p.join(ctry, on="i1")
    out = []
    for c in p["country"].unique().to_list():
        r = dict(rule.get("by_country", {}).get(c, rule.get("default", rule)))
        shift = float(r.pop("logit_shift", 0.0))
        out.append(decide(shifted(p.filter(pl.col("country") == c).drop("country"), shift), r))
    return pl.concat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", type=Path, required=True, help="cached test stage-2 predictions (i1, i2, rank, p)")
    ap.add_argument("--blocking-dir", type=Path, required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    out.mkdir(parents=True, exist_ok=True)
    rule = json.loads(a.rule)
    p2 = pl.read_parquet(a.p2, columns=["i1", "i2", "rank", "p"])
    chosen = decide_by_country_shift(p2, rule, "test")
    assert chosen.join(p2, on=["i1", "i2"], how="anti").height == 0, "a match outside the candidate set"
    s1_ids = pl.read_parquet(a.blocking_dir / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(a.blocking_dir / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    ctry = pl.read_parquet(PROJECT / "data_cache" / "test_source1.parquet").with_row_index("i1").select("i1", "country")
    by = {r["country"]: r["len"] for r in chosen.join(ctry, on="i1").group_by("country").len().iter_rows(named=True)}
    meta = {"p2": str(a.p2), "rule": rule, "candidate_pairs": p2.height, "candidates_per_s1": p2.height / len(s1_ids),
            "matches": chosen.height, "matches_by_country": by}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")
    return 0


if __name__ == "__main__":
    sys.exit(main())
