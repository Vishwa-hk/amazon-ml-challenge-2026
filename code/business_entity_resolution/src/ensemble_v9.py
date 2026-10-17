"""
Ensemble of judge_v9 stage-2 models (same region, features, universe and folds; different seeds /
sampling / tree sizes). The probability is the plain mean of the members.

  python ensemble_v9.py eval --tags v9_full v9_s1 v9_l511
  python ensemble_v9.py test --tags v9_full v9_s1 v9_l511 --v8-dir ../../../runs/v8fr --out-dir <new folder> [--rule '<json>']
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, load_region, region

RUN9 = PROJECT / "runs" / "v9"
RULES = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.6, 0.7)]
         + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.1, 1.25)])


def cmd_eval(a):
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    cut = region(src / "train_p1.parquet", kept, 0.01)
    ctry = pl.read_parquet(PROJECT / "data_cache" / "train_source1.parquet").with_row_index("i1").select("i1", "country")
    oofs = [pl.read_parquet(RUN9 / t / "train_p2.parquet").join(cut, on=["i1", "i2"], how="semi").sort("i1", "i2") for t in a.tags]
    base = oofs[0].select("i1", "i2", "rank")
    for o in oofs[1:]:
        assert o.height == base.height and (o["i2"] == base["i2"]).all()

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(pairs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    for t, o in zip(a.tags, oofs):
        rule = json.loads((RUN9 / t / "decision.json").read_text())["best_stage2"]["rule"]
        print(f"  member {t:<10} F0.5 {score(decide(owned(o, 1000), rule, already_owned=True))['f05']:.5f}", flush=True)
    ens = base.with_columns(pl.Series("p", np.mean([o["p"].to_numpy() for o in oofs], axis=0)))
    own = owned(ens, 1000)
    res = {json.dumps(r): score(decide(own, r, already_owned=True)) for r in RULES}
    for k_, f in res.items():
        print(f"  ensemble {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(own, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    print(f"BEST ensemble {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f}")
    out = RUN9 / ("ens_" + "+".join(a.tags) + ".json")
    out.write_text(json.dumps({"tags": a.tags, "best": {"rule": json.loads(best), **res[best]}, "by_country": by}, indent=2))


def cmd_test(a):
    t0 = time.time()
    v8d, out = a.v8_dir.resolve(), a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    rule = json.loads(a.rule) if a.rule else json.loads((RUN9 / ("ens_" + "+".join(a.tags) + ".json")).read_text())["best"]["rule"]
    n = pl.read_parquet(BL8 / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    cand = region(v8d / "gpu_slot" / "test_p1.parquet", kept, a.cand_p1)
    x = load_region(v8d / "test_features", cand, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet"))
    X = x.select(FEATS).to_numpy()
    preds = []
    for t in a.tags:
        for f in (0, 1):
            m = xgb.Booster(model_file=str(RUN9 / t / f"stage2_fold{f}.ubj"))
            preds.append(m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1)))
    p2 = x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean(preds, axis=0)))
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(BL8 / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(BL8 / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    p2.write_parquet(out / "test_p2_ensemble.parquet")
    meta = {"tags": a.tags, "features_from": str(v8d), "cand_p1": a.cand_p1, "rule": rule,
            "candidate_pairs": p2.height, "candidates_per_s1": p2.height / n, "matches": chosen.height}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), f"({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "test"])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--v8-dir", type=Path)
    ap.add_argument("--cand-p1", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "test" and not (a.out_dir and a.v8_dir):
        ap.error("test needs --out-dir and --v8-dir")
    {"eval": cmd_eval, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
