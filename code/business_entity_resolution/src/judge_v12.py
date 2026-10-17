"""
Judge v12 = judge v11 + WORD-SWAP features (language-independent).

Why (27 Sep): French decoys still in the uncertain band swap one generic word for another
("Pessac Ecole" -> "Pessac Parents", "Chevaux Pharmacie" -> "Chevaux Amicale"); the decoy-word
vocabulary is learned on US/India and knows no French words. Measured on TRAIN: when one CONTENT
word (legal forms, honorifics, generic suffixes and fillers excluded) is replaced by another content
word, the pair is a decoy 3-10x more often than for any other name edit (decoys per true pair:
swap 0.109 India / 0.117 US; word added 0.041 / 0.011; word dropped 0.050 / 0.025; typos 0.013 / 0.022).
The judge learns this on US/India and applies it to France automatically.

Content-word dictionary: tokens seen >= 20 times in the split's own S1 names (per country), minus
STOP words. New features: sw_add_c / sw_miss_c (content words added / missing), sw_add_o / sw_miss_o
(other tokens: typos, brand names), sw_swap (both a content word added and one missing).

  python judge_v12.py train --tag v12
  python judge_v12.py test --tag v12 --v8-dir ../../../runs/v11fr --blocking-dir ../../../runs/blocking_v11 --out-dir <new folder>
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

import pipeline_v8 as p8          # v8 name inputs (transliterated + segmented)
import pipeline_v6 as v6
from baseline import to_ids, write_id_lists
from judge_v3 import decide, decide_by_country, fold_of, macro_f05, owned
from judge_v4 import kept_entities
from judge_v9 import BL8, FEATS, PROJECT, RUN8, XPARAMS, load_region, region
from judge_v11 import CNT, copy_count_logratio, count_features
from normalize import DROP_TOKENS
from normalize_v2 import GENERIC, HONORIFICS
from translit import latin_tokens

RUN12 = PROJECT / "runs" / "v12"
SWF = ["sw_add_c", "sw_miss_c", "sw_add_o", "sw_miss_o", "sw_swap"]
STOP_VARIANT = "a"   # "b" (set by --stop-variant b) also excludes STOP_FR
STOP = (set(DROP_TOKENS) | set(GENERIC) | set(HONORIFICS)
        | {"pvt", "private", "limited", "ltd", "llp", "inc", "llc", "corp", "co", "company", "pc", "pllc", "lp", "the", "and", "of",
           "de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "au", "aux", "a", "sur", "sa", "sas", "sasu", "sarl", "eurl",
           "sci", "snc", "cie", "ei", "fils"})
# v12b: French equivalents of the English GENERIC words (associates, partners, brothers, group, center, company,
# enterprises, holdings) — in v12 they counted as content words, so "X Sportif SARL -> X SARL & Associés"
# looked like a word swap although it is the French form of "X -> X Partners".
STOP_FR = {"associes", "associe", "associees", "partenaires", "freres", "groupe", "centre", "compagnie", "societe",
           "entreprise", "entreprises", "participations", "holding", "services", "service", "international", "developpement"}


def swap_features(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """pairs (i1, i2) -> (i1, i2, SWF)."""
    s1raw, _, oth, _ = p8.load_raw_v8(split)
    S = s1raw.with_row_index("i1").select("i1", "country", latin_tokens("business_name").list.unique().alias("t1"))
    O = oth.with_row_index("i2").select("i2", latin_tokens("business_name").list.unique().alias("t2"))
    del s1raw, oth
    stop = STOP | (STOP_FR if STOP_VARIANT == "b" else set())
    dic = (S.explode("t1").group_by("country", "t1").len("c").filter((pl.col("c") >= 20) & ~pl.col("t1").is_in(list(stop)))
           .select("country", pl.col("t1").alias("w"), pl.lit(True).alias("isdict")))
    x = (pairs.select("i1", "i2").join(S, on="i1").join(O, on="i2")
         .with_columns(pl.col("t2").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("t2").alias("miss"))
         .select("i1", "i2", "country", "add", "miss"))
    del S, O

    def count(col, name):
        e = (x.select("i1", "i2", "country", pl.col(col).alias("w")).explode("w").drop_nulls()
             .join(dic, on=["country", "w"], how="left").with_columns(pl.col("isdict").fill_null(False)))
        return e.group_by("i1", "i2").agg(pl.col("isdict").sum().alias(f"sw_{name}_c"), (~pl.col("isdict")).sum().alias(f"sw_{name}_o"))
    out = (x.select("i1", "i2").join(count("add", "add"), on=["i1", "i2"], how="left").join(count("miss", "miss"), on=["i1", "i2"], how="left")
           .fill_null(0).with_columns(((pl.col("sw_add_c") >= 1) & (pl.col("sw_miss_c") >= 1)).alias("sw_swap")))
    return out.select("i1", "i2", *[pl.col(c).cast(pl.Float32) for c in SWF])


def cmd_train(a):
    t0 = time.time()
    d = RUN12 / a.tag
    v6.refuse(d)
    d.mkdir(parents=True, exist_ok=True)
    v6.BL6 = BL8
    src = RUN8 / "gpu_slot"
    n_s1 = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
    kept = kept_entities(n_s1, 0.19)
    truth = v6.truth("train").join(kept, on="i1", how="semi")
    lr = copy_count_logratio()
    p1 = pl.read_parquet(src / "train_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005).join(kept, on="i1", how="semi")
    cf = count_features(p1, lr)
    sw = swap_features(p1.select("i1", "i2"), "train")
    print(f"count + swap features ({time.time()-t0:.0f}s); swap rate {sw['sw_swap'].mean():.3%}", flush=True)
    df = load_region(RUN8 / "train_features", p1.select("i1", "i2"), kept, str(src / "train_ctx.parquet"), str(src / "train_s2" / "part-*.parquet"))
    df = (df.join(cf, on=["i1", "i2"], how="left").join(sw, on=["i1", "i2"], how="left")
          .join(truth.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["i1", "i2"], how="left")
          .with_columns(pl.col("y").fill_null(0), fold_of().alias("fold")))
    del cf, sw
    F = FEATS + CNT + SWF
    oof = []
    for fold in (0, 1):
        tr = df.filter(pl.col("fold") != fold)
        va_e = tr.select("i1").unique().sample(fraction=0.05, seed=99)
        va, tr = tr.join(va_e, on="i1", how="semi"), tr.join(va_e, on="i1", how="anti")
        dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["y"].to_numpy(), max_bin=XPARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(va.select(F).to_numpy(), va["y"].to_numpy(), ref=dtr)
        m = xgb.train(XPARAMS, dtr, num_boost_round=6000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=1000)
        m.save_model(str(d / f"stage2_fold{fold}.ubj"))
        held = df.filter(pl.col("fold") == fold)
        oof.append(held.select("i1", "i2", "rank").with_columns(
            pl.Series("p", m.inplace_predict(held.select(F).to_numpy(), iteration_range=(0, m.best_iteration + 1)))))
        print(f"[fold {fold}] {m.best_iteration + 1} trees ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, tr, va
    oof = pl.concat(oof)
    oof.write_parquet(d / "train_p2.parquet")
    ctry = pl.read_parquet(PROJECT / "data_cache" / "train_source1.parquet").with_row_index("i1").select("i1", "country")

    def score(pairs, sub=None):
        k = kept if sub is None else kept.join(ctry.filter(pl.col("country") == sub), on="i1", how="semi")
        r = k.with_row_index("j")
        return macro_f05(pairs.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"),
                         truth.join(r, on="i1").select(pl.col("j").alias("i1"), "i2"), k.height)
    cut = region(src / "train_p1.parquet", kept, 0.01)
    mine = owned(oof.join(cut, on=["i1", "i2"], how="semi"), 1000)
    rules = ([{"kind": "threshold", "t": t, "max_rank": 1000} for t in (0.6, 0.7)]
             + [{"kind": "expected_f05", "t_scale": s, "max_rank": 1000, "floor": 0.01} for s in (0.9, 1.0, 1.1, 1.25)])
    res = {json.dumps(r): score(decide(mine, r, already_owned=True)) for r in rules}
    for k_, f in res.items():
        print(f"  v12 {k_:<72} F0.5 {f['f05']:.5f} | P {f['precision']:.4f} R {f['recall']:.4f}", flush=True)
    best = max(res, key=lambda k_: res[k_]["f05"])
    ch = decide(mine, json.loads(best), already_owned=True)
    by = {c: score(ch, c)["f05"] for c in ("India", "US")}
    (d / "decision.json").write_text(json.dumps({"tag": a.tag, "lr": lr.tolist(), "best_stage2": {"rule": json.loads(best), **res[best]},
                                                 "by_country": by}, indent=2))
    print(f"\nBEST v12 {res[best]['f05']:.5f} {best} | India {by['India']:.5f} | US {by['US']:.5f} (v11: 0.98425) done ({time.time()-t0:.0f}s)")


def cmd_test(a):
    t0 = time.time()
    d, v8d, bl = RUN12 / a.tag, a.v8_dir.resolve(), a.blocking_dir.resolve()
    out = a.out_dir.resolve()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if (out / f).exists():
            raise SystemExit(f"refusing to overwrite existing {out / f}")
    dec = json.loads((d / "decision.json").read_text())
    rule = json.loads(a.rule) if a.rule else dec["best_stage2"]["rule"]
    n = pl.read_parquet(bl / "test_s1_ids.parquet").height
    kept = pl.DataFrame({"i1": pl.arange(0, n, eager=True, dtype=pl.UInt32)})
    p1 = pl.read_parquet(v8d / "gpu_slot" / "test_p1.parquet", columns=["i1", "i2", "p"]).filter(pl.col("p") >= 0.005)
    cf = count_features(p1, np.array(dec["lr"]))
    cand = p1.filter(pl.col("p") >= a.cand_p1).select("i1", "i2")
    sw = swap_features(cand, "test")
    x = (load_region(v8d / "test_features", cand, kept, None, str(v8d / "gpu_slot" / "test_s2" / "part-*.parquet"))
         .join(cf, on=["i1", "i2"], how="left").join(sw, on=["i1", "i2"], how="left"))
    X = x.select(FEATS + CNT + SWF).to_numpy()
    ms = [xgb.Booster(model_file=str(d / f"stage2_fold{f}.ubj")) for f in (0, 1)]
    p2 = x.select("i1", "i2", "rank").with_columns(pl.Series("p", np.mean([m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1)) for m in ms], axis=0)))
    p2.write_parquet(d / f"test_p2_{v8d.name}.parquet")
    chosen = decide_by_country(p2, rule, "test")
    s1_ids = pl.read_parquet(bl / "test_s1_ids.parquet")["entity_id"]
    o_ids = pl.read_parquet(bl / "test_oth_ids.parquet")["entity_id"]
    write_id_lists(out / "matching_results.tsv", s1_ids, to_ids(chosen, s1_ids, o_ids), "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, to_ids(p2.select("i1", "i2"), s1_ids, o_ids), "candidate_entity_ids")
    meta = {"tag": a.tag, "features_from": str(v8d), "blocking": str(bl), "cand_p1": a.cand_p1, "rule": rule,
            "candidate_pairs": p2.height, "candidates_per_s1": p2.height / n, "matches": chosen.height,
            "swap_rate_by_country": x.join(pl.read_parquet(PROJECT / "data_cache" / "test_source1.parquet").with_row_index("i1").select("i1", "country"), on="i1")
                                     .group_by("country").agg(pl.col("sw_swap").mean()).sort("country").rows()}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), f"({time.time()-t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                   cwd=PROJECT / "student_resource")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--tag", default="v12")
    ap.add_argument("--v8-dir", type=Path)
    ap.add_argument("--blocking-dir", type=Path, default=BL8)
    ap.add_argument("--cand-p1", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--stop-variant", choices=["a", "b"], default="a", help="b = v12b (French generic words excluded too)")
    a = ap.parse_args()
    global STOP_VARIANT
    STOP_VARIANT = a.stop_variant
    if a.cmd == "test" and not (a.out_dir and a.v8_dir):
        ap.error("test needs --out-dir and --v8-dir")
    {"train": cmd_train, "test": cmd_test}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
