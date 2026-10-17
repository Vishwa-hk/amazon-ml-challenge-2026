import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator, Union

def read_tsv_in_chunks(
    filepath: Union[str, Path],
    chunksize: int = 100000,
    usecols: list = None
) -> Iterator[pd.DataFrame]:
    """Read a tab-separated TSV file in memory-efficient chunks."""
    return pd.read_csv(
        filepath,
        sep="\t",
        chunksize=chunksize,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="skip"
    )

def read_full_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read full TSV into a DataFrame with string dtypes."""
    return pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )

def save_parquet(df: pd.DataFrame, filepath: Union[str, Path]):
    """Save a pandas DataFrame as a Parquet file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression="snappy")

def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame."""
    return pq.read_table(filepath).to_pandas()
