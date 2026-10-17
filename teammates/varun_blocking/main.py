import argparse
import sys
import os
import subprocess
from pathlib import Path

from pipeline import run_pipeline
from config import OUTPUT_DIR, PROJECT_ROOT

VALIDATE_SCRIPT = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\utils\validate_submission.py")
TEST_DATASET_DIR = Path(r"C:\Users\Varun T\.gemini\antigravity\scratch\student_resource\student_resource\dataset\test")

def main():
    parser = argparse.ArgumentParser(description="ML Challenge 2026: Business Entity Resolution Pipeline")
    parser.add_argument("--sample-train", type=int, default=None, help="Cap train S1 entities for quick dev run")
    parser.add_argument("--sample-test", type=int, default=None, help="Cap test S1 entities for quick dev run")
    parser.add_argument("--skip-test", action="store_true", help="Skip test set processing")
    parser.add_argument("--validate-only", action="store_true", help="Run official validator script only")

    args = parser.parse_args()

    matching_file = OUTPUT_DIR / "matching_results.tsv"
    candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"

    if args.validate_only:
        print("=== Running Official Submission Validator ===", flush=True)
        cmd = [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(TEST_DATASET_DIR)
        ]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    print("=== Launching Business Entity Resolution Pipeline ===", flush=True)
    metrics = run_pipeline(
        train_sample_size=args.sample_train,
        test_sample_size=args.sample_test,
        skip_test=args.skip_test
    )

    print("\n=== Pipeline Execution Summary ===", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)

    if not args.skip_test and matching_file.exists():
        print("\n=== Validating Final Output Files with Official Validator ===", flush=True)
        cmd = [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(TEST_DATASET_DIR)
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("\nSUCCESS: All output files passed official submission validation!", flush=True)
        else:
            print("\nWARNING: Validator reported issues. Check output above.", flush=True)

if __name__ == "__main__":
    main()
