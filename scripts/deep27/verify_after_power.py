# After the 27 Sep power cut: check every parquet/model written today is readable.
import sys, time, glob, os
import pyarrow.parquet as pq
import xgboost as xgb
T0 = time.mktime(time.strptime("2026-09-27 10:30", "%Y-%m-%d %H:%M"))
RECENT = time.mktime(time.strptime("2026-09-27 13:20", "%Y-%m-%d %H:%M"))
bad = []
files = [f for f in glob.glob("runs/**/*.parquet", recursive=True) if os.path.getmtime(f) > T0]
print(f"{len(files)} parquet files written today")
for f in sorted(files):
    try:
        p = pq.ParquetFile(f)
        n = p.metadata.num_rows
        if os.path.getmtime(f) > RECENT:          # full decode of recent files, one row group at a time
            m = 0
            for g in range(p.num_row_groups):
                m += p.read_row_group(g).num_rows
            assert m == n, f"rows {m} != {n}"
            print(f"  FULL OK  {f}  rows={n:,}")
    except Exception as e:
        bad.append(f); print(f"  BAD      {f}: {type(e).__name__}: {e}")
for f in glob.glob("runs/**/*.ubj", recursive=True):
    if os.path.getmtime(f) > T0:
        try:
            b = xgb.Booster(model_file=f); print(f"  model OK {f} ({b.num_boosted_rounds()} trees)")
        except Exception as e:
            bad.append(f); print(f"  BAD model {f}: {e}")
print("BAD FILES:", bad if bad else "none")
