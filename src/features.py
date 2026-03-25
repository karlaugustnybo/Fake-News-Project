from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _docs_from_input(
    data: pd.DataFrame | pl.DataFrame | pl.Series | list[str],
) -> list[str]:
    if isinstance(data, pd.DataFrame):
        return data["content"].astype(str).tolist()
    if isinstance(data, pl.DataFrame):
        return data["content"].cast(pl.String).to_list()
    if isinstance(data, pl.Series):
        return data.cast(pl.String).to_list()
    return [str(item) for item in data]


def count_vectorizer(
    max_features: int = int(1e6),
    ngram_range: tuple[int, int] = (1, 3),
) -> CountVectorizer:
    return CountVectorizer(max_features=max_features, ngram_range=ngram_range)


def tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
        token_pattern=r"[a-zA-Z]{2,}",
        dtype=np.float64,
    )


def bow(
    df: pd.DataFrame | pl.DataFrame,
    vectorizer: CountVectorizer | TfidfVectorizer,
    fit_transform: bool = True,
    save_as: str | None = None,
) -> csr_matrix:
    docs = _docs_from_input(df)
    matrix = (
        vectorizer.fit_transform(docs) if fit_transform else vectorizer.transform(docs)
    )

    if save_as is not None:
        scipy.sparse.save_npz(save_as, scipy.sparse.csr_matrix(matrix))

    return scipy.sparse.csr_matrix(matrix)


def transform_in_chunks(
    series: pl.Series,
    vectorizer: TfidfVectorizer,
    chunk_size: int = 50_000,
) -> csr_matrix:
    chunks: list[csr_matrix] = []
    n_rows = len(series)

    for start in range(0, n_rows, chunk_size):
        length = min(chunk_size, n_rows - start)
        chunk_series = series.slice(start, length)
        chunk_matrix = scipy.sparse.csr_matrix(vectorizer.transform(chunk_series))
        chunks.append(chunk_matrix)

    return scipy.sparse.csr_matrix(vstack(chunks, format="csr"))


def save_sparse_matrix(matrix: csr_matrix, filepath: str) -> None:
    scipy.sparse.save_npz(filepath, matrix)


def load_sparse_matrix(filepath: str) -> csr_matrix:
    return scipy.sparse.csr_matrix(scipy.sparse.load_npz(filepath))


def save_vectorizer(
    vectorizer: CountVectorizer | TfidfVectorizer, filepath: str
) -> None:
    joblib.dump(vectorizer, filepath)


def load_vectorizer(filepath: str) -> CountVectorizer | TfidfVectorizer:
    return joblib.load(filepath)
