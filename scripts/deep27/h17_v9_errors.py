# v9 held-out errors: category -> pairs and F0.5 points recoverable if that category were fixed.
import sys, json; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from pathlib import Path
import pipeline_v6 as v6
from judge_v3 import decide, owned
from judge_v4 import kept_entities
from judge_v9 import region
BL8 = Path("runs/blocking_v8"); v6.BL6 = BL8
C = "data_cache/"
n = pl.read_parquet(BL8 / "train_s1_ids.parquet").height
kept = kept_entities(n, 0.19)
truth = v6.truth("train").join(kept, on="i1", how="semi").with_columns(pl.lit(True).alias("y"))
cut = region(Path("runs/v8/gpu_slot/train_p1.parquet"), kept, 0.01)
rule = json.loads(Path("runs/v9/v9_full/decision.json").read_text())["best_stage2"]["rule"]
oof = pl.read_parquet("runs/v9/v9_full/train_p2.parquet").join(cut, on=["i1", "i2"], how="semi")
own = owned(oof, 1000)
chosen = decide(own, rule, already_owned=True).with_columns(pl.lit(True).alias("ch"))
cand = oof.join(truth, on=["i1", "i2"], how="left").join(chosen, on=["i1", "i2"], how="left").with_columns(
    pl.col("y").fill_null(False), pl.col("ch").fill_null(False))
cand = cand.join(own.select("i1", "i2").with_columns(pl.lit(True).alias("owner")), on=["i1", "i2"], how="left").with_columns(pl.col("owner").fill_null(False))
# F bookkeeping per entity
nt = truth.group_by("i1").len("nt")
def macro(tp, np_, nt_):
    return pl.when((nt_ == 0) & (np_ == 0)).then(1.0).otherwise(1.25 * tp / (1.25 * tp + 0.25 * (nt_ - tp) + (np_ - tp)))
base = kept.join(nt, on="i1", how="left").join(cand.filter("ch").group_by("i1").agg(pl.len().alias("np"), pl.col("y").sum().alias("tp")), on="i1", how="left").fill_null(0)
F0 = base.select(macro(pl.col("tp"), pl.col("np"), pl.col("nt")).mean()).item()
print(f"v9 held-out on candidates: F0.5 {F0:.5f}  (kept {kept.height:,})")
# features for categorising
feat = pl.concat([pl.read_parquet(p, columns=["i1", "i2", "tok_jac", "addr_jac", "num_sh", "non_latin", "off_first", "off_first_set", "off_any_set", "add_lo_max", "miss_lo_max", "n_added", "n_missing"])
                  .join(cand.filter(pl.col("y") != pl.col("ch")).select("i1", "i2"), on=["i1", "i2"], how="semi")
                  for p in sorted(Path("runs/v8/train_features").glob("part-*.parquet"))])
err = cand.filter(pl.col("y") != pl.col("ch")).join(feat, on=["i1", "i2"], how="left")
oth = pl.concat([pl.read_parquet(C + "train_source2.parquet"), pl.read_parquet(C + "train_source3.parquet")]).with_row_index("i2").select("i2", "business_address")
err = err.join(oth, on="i2", how="left").with_columns(
    (pl.col("business_address").is_null() | pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None", "null", "<NULL>", "N/A"])).alias("noaddr"))
err = err.with_columns(
    pl.when(pl.col("y")).then(pl.when(~pl.col("owner")).then(pl.lit("FN lost-owner")).otherwise(pl.lit("FN below-cut")))
      .otherwise(pl.lit("FP")).alias("kind"),
    pl.when(pl.col("noaddr")).then(pl.lit("no-addr")).when(pl.col("off_first").is_null()).then(pl.lit("no-num"))
      .when(pl.col("off_first") == 0).then(pl.lit("num=")).when(pl.col("off_first_set") == 1).then(pl.lit("decoy-off"))
      .when(pl.col("off_first").abs() <= 15).then(pl.lit("small-off")).otherwise(pl.lit("far/other")).alias("num"),
    pl.when(pl.col("tok_jac") == 0).then(pl.lit("name0")).when(pl.col("tok_jac") < 0.5).then(pl.lit("name<.5"))
      .when(pl.col("tok_jac") < 1).then(pl.lit("name<1")).otherwise(pl.lit("name=1")).alias("name"),
)
def gain(sub):
    # fix every error in `sub`: FP removed, FN added
    fp = sub.filter(~pl.col("y")).group_by("i1").len("dfp")
    fn = sub.filter(pl.col("y")).group_by("i1").len("dfn")
    b = base.join(fp, on="i1", how="left").join(fn, on="i1", how="left").fill_null(0)
    return b.select(macro(pl.col("tp") + pl.col("dfn"), pl.col("np") - pl.col("dfp") + pl.col("dfn"), pl.col("nt")).mean()).item() - F0
print(f"total recoverable if every candidate-level error fixed: {gain(err):.5f}")
rows = []
for keys in (["kind"], ["kind", "num"], ["kind", "name"], ["kind", "num", "name"]):
    for g, sub in err.group_by(keys):
        rows.append((" / ".join(map(str, g)), sub.height, gain(sub)))
for r in sorted(rows, key=lambda r: -r[2])[:45]:
    print(f"  {r[0]:<42} pairs {r[1]:>8,}  gain {r[2]:.5f}")
err.write_parquet("runs/v9/v9_full/errors_heldout.parquet")
