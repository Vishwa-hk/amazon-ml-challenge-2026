"""
Rule-based baseline for the Business Entity Resolution challenge (no ML model).

Pipeline
  1. normalize names -> token-set key + compact key; extract house numbers and
     address tokens (see normalize.py)
  2. BLOCKING (candidate_pairs.tsv): an S1 entity and an S2/S3 record become a
     candidate pair when
        same country  AND  (same token-set key OR same compact key)
        AND their house numbers do not conflict
            (they share a number, or one side has no number)
     Empty keys never block (a name made only of legal words carries no signal).
  3. MATCHING RULES (matching_results.tsv), per candidate pair:
        accept if they share >= 1 house number
        accept if one side has no number AND the record has exactly one S1
               candidate (nothing to confuse it with)          [toggle]
  4. ONE-TO-ONE: the data guarantees each S2/S3 record belongs to at most one
     S1 entity. A record accepted for several S1 entities keeps the best one
     (most shared numbers, then highest address-token Jaccard); an exact tie
     drops the record entirely (F0.5 punishes a wrong merge more than a miss).

Usage (run from this src/ directory or anywhere):
  python baseline.py --split train   # writes train predictions and scores them
  python baseline.py --split test    # writes output/matching_results.tsv + candidate_pairs.tsv
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

from normalize import add_features, clean_text
from scorer import parse_id_list_file, print_report, read_s1_country, score

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parents[2]  # .../amazon-ml-challenge-2026


def load_source(data_dir: Path, cache_dir: Path, split: str, name: str) -> pl.DataFrame:
    """Read one TSV (all-string columns), using a Parquet cache when available."""
    pq = cache_dir / f"{split}_{name}.parquet"
    if pq.exists():
        return pl.read_parquet(pq)
    df = pl.read_csv(data_dir / split / f"{split}_{name}.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(pq)
    return df


def addr_tokens(col: str) -> pl.Expr:
    """Address word tokens (no digits, len >= 3), hashed to u64 to keep joins light."""
    return (clean_text(pl.col(col)).str.replace_all(r"[^\p{L}]+", " ").str.strip_chars()
            .str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).hash())
            .list.unique())


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """Features keyed by an integer row index (i) instead of the string entity_id."""
    df = add_features(df).with_columns(addr_tokens("business_address").alias("atok"))
    return df.with_row_index("i").select("i", "country", "key_ts", "key_cp", "nums", "atok")


def build_candidates(s1: pl.DataFrame, oth: pl.DataFrame) -> pl.DataFrame:
    """Blocking: same country + same name key + non-conflicting house numbers.

    Works on integer row indices (i1 = S1 row, i2 = S2/S3 row). Each name-key join
    is filtered for number conflicts immediately, so the ~100M raw pairs never
    have to sit in memory together.
    """
    A = s1.rename({"i": "i1", "nums": "n1"})
    B = oth.rename({"i": "i2", "nums": "n2"})
    parts = []
    for k in ("key_ts", "key_cp"):
        j = (
            A.lazy().filter(pl.col(k) != "").select("i1", "country", k, "n1")
            .join(B.lazy().filter(pl.col(k) != "").select("i2", "country", k, "n2"), on=["country", k])
            .select(
                "i1", "i2",
                pl.col("n1").list.set_intersection("n2").list.len().alias("shared"),
                ((pl.col("n1").list.len() == 0) | (pl.col("n2").list.len() == 0)).alias("one_empty"),
            )
            .filter((pl.col("shared") >= 1) | pl.col("one_empty"))
            .collect()
        )
        parts.append(j)
    cand = pl.concat(parts).unique(["i1", "i2"])
    del parts
    # address similarity only for the surviving pairs
    cand = (cand.join(s1.select(pl.col("i").alias("i1"), pl.col("atok").alias("a1")), on="i1")
                .join(oth.select(pl.col("i").alias("i2"), pl.col("atok").alias("a2")), on="i2"))
    inter = pl.col("a1").list.set_intersection("a2").list.len()
    union = pl.col("a1").list.set_union("a2").list.len()
    cand = cand.with_columns(
        pl.when(union > 0).then(inter / union).otherwise(0.0).alias("ajac"),
    ).drop("a1", "a2")
    return cand.with_columns(pl.col("i1").n_unique().over("i2").alias("r_cands"))


def match(cand: pl.DataFrame, accept_single_candidate: bool = True) -> pl.DataFrame:
    """Apply the rules, then enforce one S1 per S2/S3 record."""
    rule = pl.col("shared") >= 1
    if accept_single_candidate:
        rule = rule | (pl.col("one_empty") & (pl.col("r_cands") == 1))
    acc = cand.filter(rule)
    # best S1 per record: most shared numbers, then highest address Jaccard
    acc = acc.filter(pl.col("shared") == pl.col("shared").max().over("i2"))
    acc = acc.filter(pl.col("ajac") == pl.col("ajac").max().over("i2"))
    return acc.filter(pl.len().over("i2") == 1).select("i1", "i2")  # exact tie -> drop


def to_ids(pairs: pl.DataFrame, s1_ids: pl.Series, oth_ids: pl.Series) -> pl.DataFrame:
    """Map integer row indices back to entity_id strings."""
    return pairs.select(
        pl.lit(s1_ids).gather(pl.col("i1")).alias("s1"),
        pl.lit(oth_ids).gather(pl.col("i2")).alias("r"),
    )


def write_id_lists(path: Path, all_s1: pl.Series, pairs: pl.DataFrame, list_col: str) -> None:
    """One row per S1 entity (empty list when none), TAB-separated, LF newlines."""
    lists = pairs.group_by("s1").agg(pl.col("r").unique().sort().str.join(",").alias(list_col))
    out = (pl.DataFrame({"source1_entity_id": all_s1})
           .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
           .with_columns(pl.col(list_col).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")


def run(split: str, data_dir: Path, cache_dir: Path, out_dir: Path, accept_single: bool) -> None:
    t0 = time.time()
    s1_raw = load_source(data_dir, cache_dir, split, "source1")
    oth_raw = pl.concat([load_source(data_dir, cache_dir, split, "source2"),
                         load_source(data_dir, cache_dir, split, "source3")])
    s1_ids, oth_ids = s1_raw["entity_id"], oth_raw["entity_id"]
    s1, oth = prepare(s1_raw), prepare(oth_raw)
    del oth_raw
    print(f"[{split}] features: {time.time()-t0:.0f}s")
    cand = build_candidates(s1, oth)
    del s1, oth
    print(f"[{split}] candidate pairs: {cand.height:,} ({time.time()-t0:.0f}s)")
    pairs = match(cand, accept_single_candidate=accept_single)
    print(f"[{split}] matched pairs: {pairs.height:,} ({time.time()-t0:.0f}s)")

    prefix = "" if split == "test" else f"{split}_"
    m_path = out_dir / f"{prefix}matching_results.tsv"
    c_path = out_dir / f"{prefix}candidate_pairs.tsv"
    write_id_lists(m_path, s1_ids, to_ids(pairs, s1_ids, oth_ids), "matched_entity_ids")
    write_id_lists(c_path, s1_ids, to_ids(cand.select("i1", "i2"), s1_ids, oth_ids), "candidate_entity_ids")
    print(f"[{split}] wrote {m_path} and {c_path} ({time.time()-t0:.0f}s)")

    if split == "train":  # score by re-reading the written file: tests the writer too
        gt = parse_id_list_file(str(data_dir / "train" / "train_ground_truth.tsv"), "matched_entity_ids")
        pred = parse_id_list_file(str(m_path), "matched_entity_ids")
        cands = parse_id_list_file(str(c_path), "candidate_entity_ids")
        found = sum(len(cands.get(k, set()) & v) for k, v in gt.items())
        total = sum(len(v) for v in gt.values())
        print(f"[train] blocking recall ceiling: {found/total:.2%} of true pairs are candidates")
        print_report(score(pred, gt, read_s1_country(str(data_dir / "train" / "train_source1.tsv"))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "student_resource" / "dataset")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "output")
    ap.add_argument("--no-single-candidate-rule", action="store_true",
                    help="disable: accept no-number pairs when the record has one S1 candidate")
    a = ap.parse_args()
    run(a.split, a.data_dir, a.cache_dir, a.out_dir, accept_single=not a.no_single_candidate_rule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
