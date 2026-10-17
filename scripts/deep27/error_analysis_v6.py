"""
v3 error analysis on the out-of-fold (held-out) predictions — train only.

1. Points-lost decomposition (per-entity counterfactuals, macro-averaged):
     F_actual -> F_noFP (drop every false match) -> F_oracle (also add every true
     pair that was a candidate) -> 1.0 (also add pairs the short-lists missed)
   loss = FP part + judge-miss part + blocking part.
2. Error types:
     FP  decoy          chosen record belongs to no S1 entity
     FP  wrong owner    chosen record belongs to a different S1 entity
     FN  lost owner     true candidate removed by one-owner-per-record (another S1 won)
     FN  cut            true candidate kept but below the expected-F0.5 cut
3. Slices by country / source / singleton, and feature profiles of each error
   type vs correct matches, plus examples with real names/addresses.
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))
from judge_v3 import decide, macro_f05, owned  # noqa: E402
from judge_v4 import kept_entities  # noqa: E402
from pipeline_v6 import truth as truth_pairs  # noqa: E402

RUN, BL, C = ROOT / "runs" / "v6", ROOT / "runs" / "blocking_v6", ROOT / "data_cache"
OUT = ROOT / "runs" / "v6_error_analysis"
OUT.mkdir(parents=True, exist_ok=True)
rep = open(OUT / "report.txt", "w", encoding="utf-8")


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    rep.write(m + "\n")


t0 = time.time()
rule = json.loads((RUN / "gpu_translit" / "decision.json").read_text())["best_stage2"]["rule"]
oof = pl.read_parquet(RUN / "gpu_translit" / "train_p2.parquet")
_n = pl.read_parquet(BL / "train_s1_ids.parquet").height
kept = kept_entities(_n, 0.19)
truth = truth_pairs().join(kept, on="i1", how="semi").with_columns(pl.lit(True).alias("is_true"))
n_s1 = kept.height  # test-like universe: only kept entities are scored (removed ones don't exist here)
own = owned(oof, rule.get("max_rank"))
chosen = (decide(own, rule, already_owned=True)
          .join(own.select("i1", "i2", "p"), on=["i1", "i2"])   # decide() returns ids only
          .with_columns(pl.lit(True).alias("chosen")))
log(f"v6 stage-2 held-out, test-like universe ({n_s1:,} kept S1 entities, 40% decoy records) | rule {rule}")

# ---------------------------------------------------------------- 1. decomposition
cand_true = oof.select("i1", "i2").join(truth, on=["i1", "i2"], how="semi")
nt = truth.group_by("i1").len("nt")
per = (kept
       .join(nt, on="i1", how="left")
       .join(chosen.group_by("i1").len("np"), on="i1", how="left")
       .join(chosen.join(truth, on=["i1", "i2"], how="semi").group_by("i1").len("tp"), on="i1", how="left")
       .join(cand_true.group_by("i1").len("tc"), on="i1", how="left")
       .fill_null(0))
F = lambda tp, np_, nt_: pl.when((nt_ == 0) & (np_ == 0)).then(1.0).otherwise(
    1.25 * tp / (1.25 * tp + 0.25 * (nt_ - tp) + (np_ - tp)))
per = per.with_columns(
    F(pl.col("tp"), pl.col("np"), pl.col("nt")).alias("f_act"),
    F(pl.col("tp"), pl.col("tp"), pl.col("nt")).alias("f_nofp"),
    F(pl.col("tc"), pl.col("tc"), pl.col("nt")).alias("f_orc"),
)
s = per.select(pl.col("f_act").mean(), pl.col("f_nofp").mean(), pl.col("f_orc").mean()).row(0, named=True)
log("\n" + "=" * 78 + "\n1. WHERE THE POINTS GO (held-out, all train entities)\n" + "=" * 78)
log(f"  actual F0.5                         {s['f_act']:.5f}")
log(f"  lost to FALSE matches               {s['f_nofp']-s['f_act']:.5f}")
log(f"  lost to true candidates not chosen  {s['f_orc']-s['f_nofp']:.5f}")
log(f"  lost to short-list misses           {1-s['f_orc']:.5f}")
log(f"  total loss                          {1-s['f_act']:.5f}")
sing = per.filter(pl.col("nt") == 0)
log(f"  singletons: {sing.height:,} | wrongly given matches {int((sing['np'] > 0).sum()):,} "
    f"| points lost there {(1-sing['f_act']).sum()/n_s1:.5f}")

# ---------------------------------------------------------------- 2. error types
owner = truth.select(pl.col("i1").alias("true_owner"), "i2")
fp = chosen.join(truth, on=["i1", "i2"], how="anti").join(owner, on="i2", how="left")
fp = fp.join(oof.select(pl.col("i1").alias("true_owner"), "i2", pl.col("p").alias("p_true_owner")),
             on=["true_owner", "i2"], how="left")
tc = oof.join(truth, on=["i1", "i2"], how="semi")
fn = tc.join(chosen.select("i1", "i2"), on=["i1", "i2"], how="anti")
fn_lost = fn.join(own.select("i1", "i2"), on=["i1", "i2"], how="anti")
fn_cut = fn.join(own.select("i1", "i2"), on=["i1", "i2"], how="semi")
n_ch = chosen.height
log("\n" + "=" * 78 + "\n2. ERROR TYPES (pairs)\n" + "=" * 78)
log(f"  chosen pairs {n_ch:,} | true pairs {truth.height:,} | true pairs in short-lists {tc.height:,}")
log(f"  FP decoy                   {fp['true_owner'].is_null().sum():>9,}")
log(f"  FP wrong owner             {fp['true_owner'].is_not_null().sum():>9,}  "
    f"(true owner had it as candidate: {fp['p_true_owner'].is_not_null().sum():,})")
log(f"  FN lost owner competition  {fn_lost.height:>9,}")
log(f"  FN below cut               {fn_cut.height:>9,}   p quartiles {fn_cut['p'].quantile(0.25):.3f} / "
    f"{fn_cut['p'].quantile(0.5):.3f} / {fn_cut['p'].quantile(0.75):.3f}")
log(f"  FP p quartiles             {fp['p'].quantile(0.25):.3f} / {fp['p'].quantile(0.5):.3f} / {fp['p'].quantile(0.75):.3f}")

# ---------------------------------------------------------------- 3. slices + profiles
s1_ids = pl.read_parquet(BL / "train_s1_ids.parquet")["entity_id"]
o_ids = pl.read_parquet(BL / "train_oth_ids.parquet")["entity_id"]
s1raw = pl.read_parquet(C / "train_source1.parquet").with_row_index("i1")
othraw = pl.concat([pl.read_parquet(C / "train_source2.parquet"), pl.read_parquet(C / "train_source3.parquet")]).with_row_index("i2")
country = s1raw.select("i1", "country")
per = per.join(country, on="i1")
log("\n" + "=" * 78 + "\n3. SLICES\n" + "=" * 78)
for r in per.group_by("country").agg(pl.len(), pl.col("f_act").mean(), pl.col("f_nofp").mean(), pl.col("f_orc").mean()).sort("country").iter_rows(named=True):
    log(f"  {r['country']:<6} n={r['len']:>9,} | F {r['f_act']:.5f} | lost: FP {r['f_nofp']-r['f_act']:.5f}, "
        f"judge-miss {r['f_orc']-r['f_nofp']:.5f}, short-list {1-r['f_orc']:.5f}")
n_s2 = pl.read_parquet(C / "train_source2.parquet").height
for name, df in (("FP", fp), ("FN lost", fn_lost), ("FN cut", fn_cut), ("short-list miss", truth.join(oof, on=["i1", "i2"], how="anti"))):
    log(f"  {name:<16} share from S3: {(df['i2'] >= n_s2).mean():.1%}")

feats = ["name_tset", "name_ratio", "cp_jw", "addr_tset", "num_sh", "num_partial", "non_latin", "rank", "rank_r", "r_cands", "gap_r"]
F3 = pl.scan_parquet(RUN / "train_features" / "part-*.parquet").select(["i1", "i2"] + feats)
tp_s = chosen.join(truth, on=["i1", "i2"], how="semi").sample(200_000, seed=1)
groups = {"correct match": tp_s, "FP decoy": fp.filter(pl.col("true_owner").is_null()),
          "FP wrong owner": fp.filter(pl.col("true_owner").is_not_null()),
          "FN lost owner": fn_lost, "FN below cut": fn_cut}
log("\n  feature means by group:")
log("  " + f"{'group':<16}" + "".join(f"{f:>12}" for f in feats))
prof = {}
for g, df in groups.items():
    x = F3.join(df.select("i1", "i2").lazy(), on=["i1", "i2"], how="semi").collect()
    prof[g] = x
    log("  " + f"{g:<16}" + "".join(f"{x[f].mean():>12.3f}" for f in feats))

for g_, df_ in groups.items():
    df_.select([c for c in df_.columns if c in ("i1","i2","p","true_owner","p_true_owner")]).write_parquet(OUT / (g_.replace(" ","_") + ".parquet"))
per.write_parquet(OUT / "per_entity.parquet")
log("\n" + "=" * 78 + "\n4. EXAMPLES (train)\n" + "=" * 78)
N1 = s1raw.select("i1", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
N2 = othraw.select("i2", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
for g, df in groups.items():
    if g == "correct match":
        continue
    ex = df.sample(min(25, df.height), seed=3).join(N1, on="i1").join(N2, on="i2")
    if "true_owner" in ex.columns:
        ex = ex.join(N1.rename({"i1": "true_owner", "n1": "tn", "a1": "ta"}), on="true_owner", how="left")
    log(f"\n  --- {g}")
    for r in ex.iter_rows(named=True):
        log(f"      p={r['p']:.3f}  S1: {r['n1']}  |  {r['a1']}")
        log(f"               rec: {r['n2']}  |  {r['a2']}")
        if r.get("tn"):
            log(f"          true S1: {r['tn']}  |  {r['ta']}")
miss = truth.join(oof, on=["i1", "i2"], how="anti").sample(15, seed=5).join(N1, on="i1").join(N2, on="i2")
log("\n  --- short-list misses")
for r in miss.iter_rows(named=True):
    log(f"      S1: {r['n1']}  |  {r['a1']}")
    log(f"      rec: {r['n2']}  |  {r['a2']}")
log(f"\ndone ({time.time()-t0:.0f}s) -> {OUT/'report.txt'}")
rep.close()
