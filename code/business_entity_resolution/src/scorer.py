"""
Local F_0.5 scorer for the Amazon ML Challenge 2026 (Business Entity Resolution).

Implements the metric exactly as specified in the problem statement:

  * F_beta with beta = 0.5, computed PER Source-1 entity, then MACRO-averaged
    over every Source-1 entity in the evaluation set (the ground truth).
  * Precision/recall per entity use set semantics (duplicate IDs count once).
  * Singletons: true list empty -> 1.0 if the prediction is empty, else 0.0.
  * A non-empty truth with an empty prediction scores 0.0.

Per entity this is algebraically identical to the published formula
    F_0.5 = 1.25*P*R / (0.25*P + R)
written in count form so the zero cases need no special handling:
    F_0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP)

Files are parsed with the same rules as utils/validate_submission.py: first
tab splits the S1 id from the list, the list splits on commas, empty list =
no match. Stdlib only, so it runs anywhere.

The official public leaderboard scores a hidden SUBSET of the test set, so a
local score on a held-out train slice is an estimate, not a prediction of the
exact leaderboard number.

Usage:
    python scorer.py --pred matching_results.tsv --gt train_ground_truth.tsv \
                     [--s1 train_source1.tsv]      # adds a per-country breakdown
    python scorer.py --selftest
"""
import argparse
import sys
from collections import defaultdict

DELIM = "\t"
BETA2 = 0.5 ** 2  # beta squared = 0.25


def f05(pred: set, true: set) -> float:
    """F_0.5 for one Source-1 entity."""
    if not pred and not true:
        return 1.0  # correctly predicted singleton
    tp = len(pred & true)
    fp = len(pred) - tp
    fn = len(true) - tp
    return (1 + BETA2) * tp / ((1 + BETA2) * tp + BETA2 * fn + fp)


def parse_id_list_file(path: str, list_col: str) -> dict:
    """Read a results-style TSV into {source1_entity_id: set(ids)}.

    Mirrors validate_submission.py parsing. Raises on a wrong header.
    """
    with open(path, "rb") as fb:
        if b"\r\n" in fb.read(1 << 16):
            print(f"WARNING: {path} uses CRLF line endings; write LF ('\\n') to be safe.",
                  file=sys.stderr)
    mapping = {}
    dup_rows = 0
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        expected = ["source1_entity_id", list_col]
        if cols != expected:
            raise ValueError(f"{path}: header {cols}, expected {expected}")
        for line in f:
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                continue
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if s1 in mapping:
                dup_rows += 1
            mapping[s1] = set(ids)
    if dup_rows:
        print(f"WARNING: {path} has {dup_rows} duplicate S1 rows (the real scorer rejects this).",
              file=sys.stderr)
    return mapping


def read_s1_country(path: str) -> dict:
    """{entity_id: country} from a source-1 TSV (columns: id, name, address, country)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) >= 4:
                out[parts[0]] = parts[3]
    return out


def score(pred: dict, gt: dict, s1_country: dict = None) -> dict:
    """Macro F_0.5 of predictions against ground truth, plus diagnostics.

    Entities in gt but missing from pred are scored as empty predictions (the
    real scorer would REJECT such a file; we count them so you notice).
    Entities in pred but not in gt are ignored and counted.
    """
    total = 0.0
    n = len(gt)
    missing = 0
    sing_n = sing_sum = sing_fp = 0
    ns_n = ns_sum = 0
    tp_all = fp_all = fn_all = pred_size = 0
    by_country = defaultdict(lambda: [0, 0.0])
    for s1, true in gt.items():
        p = pred.get(s1)
        if p is None:
            missing += 1
            p = set()
        s = f05(p, true)
        total += s
        tp = len(p & true)
        tp_all += tp
        fp_all += len(p) - tp
        fn_all += len(true) - tp
        pred_size += len(p)
        if true:
            ns_n += 1
            ns_sum += s
        else:
            sing_n += 1
            sing_sum += s
            sing_fp += bool(p)
        if s1_country is not None:
            c = by_country[s1_country.get(s1, "?")]
            c[0] += 1
            c[1] += s
    return {
        "macro_f05": total / n if n else 0.0,
        "entities": n,
        "missing_from_pred": missing,
        "extra_in_pred": len(set(pred) - set(gt)),
        "singletons": sing_n,
        "singleton_f05": sing_sum / sing_n if sing_n else float("nan"),
        "singletons_wrongly_matched": sing_fp,
        "nonsingleton_f05": ns_sum / ns_n if ns_n else float("nan"),
        "micro_precision": tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0,
        "micro_recall": tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0,
        "mean_pred_list_size": pred_size / n if n else 0.0,
        "by_country": {k: v[1] / v[0] for k, v in sorted(by_country.items())} if s1_country else None,
    }


def print_report(r: dict) -> None:
    print(f"macro F0.5            : {r['macro_f05']:.5f}   <- the leaderboard metric")
    print(f"entities scored       : {r['entities']:,}  (missing from pred: {r['missing_from_pred']:,}, "
          f"extra ignored: {r['extra_in_pred']:,})")
    print(f"non-singleton F0.5    : {r['nonsingleton_f05']:.5f}")
    print(f"singleton F0.5        : {r['singleton_f05']:.5f}  ({r['singletons']:,} singletons, "
          f"{r['singletons_wrongly_matched']:,} wrongly given matches)")
    print(f"micro precision/recall: {r['micro_precision']:.4f} / {r['micro_recall']:.4f}")
    print(f"mean predicted list   : {r['mean_pred_list_size']:.2f}")
    if r["by_country"]:
        print("by country            : " + ", ".join(f"{k}={v:.5f}" for k, v in r["by_country"].items()))


def selftest() -> None:
    """Check the implementation against the worked example and edge cases."""
    # Worked example from the problem statement: expected 0.714
    s = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(s - 0.7142857) < 1e-6, s
    # Must agree with the published P/R formula on a spread of cases
    for pred, true in [({"a"}, {"a", "b", "c", "d"}), ({"a", "x"}, {"a", "b"}), ({"a", "b"}, {"a", "b"})]:
        tp = len(pred & true)
        P, R = tp / len(pred), tp / len(true)
        ref = 1.25 * P * R / (0.25 * P + R)
        assert abs(f05(pred, true) - ref) < 1e-12, (pred, true)
    assert f05(set(), set()) == 1.0          # correct singleton
    assert f05({"S2-1"}, set()) == 0.0       # false merge on a singleton
    assert f05(set(), {"S2-1"}) == 0.0       # missed everything
    assert f05({"S2-9"}, {"S2-1"}) == 0.0    # wrong match only
    assert abs(f05({"S2-1"}, {"S2-1", "S2-2", "S2-3", "S2-4"}) - 0.625) < 1e-12
    # Macro average over entities, singletons included
    r = score({"A": {"x"}, "B": set(), "C": {"y"}}, {"A": {"x"}, "B": set(), "C": set()})
    assert abs(r["macro_f05"] - 2 / 3) < 1e-12
    print("selftest PASS: worked example = 0.714, edge cases and macro-average correct")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="matching_results.tsv to score")
    ap.add_argument("--gt", help="ground truth TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--s1", help="optional source-1 TSV for a per-country breakdown")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not (a.pred and a.gt):
        ap.error("--pred and --gt are required (or use --selftest)")
    pred = parse_id_list_file(a.pred, "matched_entity_ids")
    gt = parse_id_list_file(a.gt, "matched_entity_ids")
    print_report(score(pred, gt, read_s1_country(a.s1) if a.s1 else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
