"""
Temporal train / validation / test split (70 / 15 / 15).

Split is time-ordered, not random, to prevent data leakage across time.
Test set  = most recent 15 % of rows (by timestamp).
Val set   = the 15 % immediately before the test set.
Train set = everything earlier.

Usage:
    python -m ml.splits
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("ml/data/processed")


def split(src: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load historical.parquet and return (train, val, test) DataFrames.

    No row appears in more than one split.  The split boundary is computed
    globally across all spots so that the test set always covers the most
    recent calendar period (avoids per-spot contamination).
    """
    if src is None:
        src = PROCESSED / "historical.parquet"

    df = pd.read_parquet(src).sort_values("timestamp").reset_index(drop=True)
    n  = len(df)

    test_start = int(n * 0.85)
    val_start  = int(n * 0.70)

    train = df.iloc[:val_start].copy()
    val   = df.iloc[val_start:test_start].copy()
    test  = df.iloc[test_start:].copy()

    return train, val, test


def save_splits(src: Path | None = None, dst: Path | None = None) -> None:
    """Write train / val / test parquet files to dst directory."""
    if dst is None:
        dst = PROCESSED
    dst.mkdir(parents=True, exist_ok=True)

    train, val, test = split(src)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = dst / f"{name}.parquet"
        df.to_parquet(path, index=False)
        earliest = df["timestamp"].min()
        latest   = df["timestamp"].max()
        print(f"  {name}: {len(df):>7,} rows  {earliest.date()} → {latest.date()}  → {path}")


if __name__ == "__main__":
    print("Splitting historical.parquet …")
    save_splits()
