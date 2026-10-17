"""
Write a submission with a SMALL candidate set (27 Sep rule update: candidate_pairs.tsv is part of the
final evaluation, and a smaller candidate set per Source-1 entity ranks higher).

Pipeline view:
  1. blocking keys            -> wide internal short-list (~97 pairs per S1; scalable inverted-index joins)
  2. stage-1 candidate scorer -> candidate_pairs.tsv = pairs with stage-1 probability >= --cand-p1
  3. stage-2 judge            -> matching_results.tsv, chosen ONLY among those candidates (matches ⊆ candidates)

Measured on train (v8, held-out, 1/3 of entities): full list 97.2 per S1 / F0.5 0.98309;
stage-1 p >= 0.05 -> 3.67 per S1 / 0.98309; p >= 0.01 -> 4.28 / 0.98310.

Reads the cached test predictions of a trained run (<run-dir>/test_p1.parquet, test_p2.parquet); runs in
about a minute. Never overwrites: --out-dir must not already hold submission files.

  python export_submission.py --run-dir ../../../runs/v8/gpu_slot --blocking-dir ../../../runs/blocking_v8 \
      --cand-p1 0.05 --out-dir ../../../runs/submissions/<new folder> [--rule '<json>']
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from baseline import to_ids, write_id_lists
from judge_v3 import decide_by_country

PROJECT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--blocking-dir", type=Path, required=True)
    ap.add_argument("--cand-p1", type=float, required=True, help="stage-1 probability cut for the submitted candidate set")
    ap.add_argument("--rule", help="decision rule JSON (default: the run's decision.json)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((a.run_dir / "decision.json").read_text())["best_stage2"]["rule"]
    p1 = pl.read_parquet(a.run_dir / "test_p1.parquet", columns=["i1", "i2", "p"]).rename({"p": "p1"})
    p2 = pl.read_parquet(a.run_dir / "test_p2.parquet", columns=["i1", "i2", "rank", "p"])
    n_all = p2.height
    cand = p2.join(p1.filter(pl.col("p1") >= a.cand_p1).select("i1", "i2"), on=["i1", "i2"], how="semi")
    del p1, p2
    chosen = decide_by_country(cand, rule, "test")
    s1_ids = pl.read_parquet(a.blocking_dir / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(a.blocking_dir / "test_oth_ids.parquet")["entity_id"]
    assert chosen.join(cand, on=["i1", "i2"], how="anti").height == 0, "a match outside the candidate set"
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(cand.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    meta = {"run_dir": str(a.run_dir), "cand_p1": a.cand_p1, "rule": rule, "internal_pairs": n_all,
            "candidate_pairs": cand.height, "candidates_per_s1": cand.height / len(s1_ids), "matches": chosen.height}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")
    return 0


if __name__ == "__main__":
    sys.exit(main())
