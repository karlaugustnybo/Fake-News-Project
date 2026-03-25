from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


REAL_LABELS = ["reliable"]
DROP_LABELS = ["unknown", "nan", "2018-02-10 13:43:39.521661"]


def standard_labels(
    df: pl.DataFrame,
    type_column: str = "type",
    real_labels: list[str] | None = None,
    drop_labels: list[str] | None = None,
) -> pl.DataFrame:
    real_labels = real_labels or REAL_LABELS
    drop_labels = drop_labels or DROP_LABELS

    return df.filter(~pl.col(type_column).is_in(drop_labels)).with_columns(
        pl.when(pl.col(type_column).is_in(real_labels))
        .then(pl.lit("real"))
        .otherwise(pl.lit("fake"))
        .alias("label")
    )


def split_data(
    df: pl.DataFrame,
    label_column: str = "label",
    random_state: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = df.shape[0]
    indices = np.arange(n)
    labels = df[label_column].to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=labels[temp_idx],
    )

    return (
        df[list(train_idx)],
        df[list(val_idx)],
        df[list(test_idx)],
    )


def label_distribution(
    split: pl.DataFrame, label_column: str = "label"
) -> pl.DataFrame:
    return (
        split.group_by(label_column)
        .agg(pl.len().alias("count"))
        .sort(label_column)
        .with_columns(
            (pl.col("count").cast(pl.Float64) / pl.col("count").sum() * 100)
            .round(1)
            .alias("pct")
        )
    )


def binary_labels(split: pl.DataFrame, label_column: str = "label") -> np.ndarray:
    return (split[label_column] == "fake").cast(pl.Int8).to_numpy()
