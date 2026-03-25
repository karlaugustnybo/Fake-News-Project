from __future__ import annotations

import os
import re
import sys

import Stemmer
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


natural_language_cols = [
    "content",
    "title",
    "authors",
    "meta_keywords",
    "meta_description",
]

email_str = r'[^ \n,"]+@[^ \n,"]+\.[^ \n,"]+'
date_str = r"[0-9]{2,4}[-/][0-9]{2,4}[-/][0-9]{2,4}"
url_str = r'(?:http)?s?(?://)?[^ \n,"]+\.[a-z]{2,}[^ \n,"]+'
num_str = r"[0-9]+[,.]?[0-9]*"
special_str = "[^a-zA-Z\\-<>'\u2019\u2018\u02bc\u2032\uff07]"
apostrephe_str = "['\u2019\u2018\u02bc\u2032\uff07]"
ws_str = r"\s+"

email_re = re.compile(email_str)
date_re = re.compile(date_str)
url_re = re.compile(url_str)
num_re = re.compile(num_str)
special_re = re.compile(special_str)
apostrephe_re = re.compile(apostrephe_str)
ws_re = re.compile(ws_str)

_stop_words = set(stopwords.words("english"))
_porter_stemmer = PorterStemmer()
_pystemmer = Stemmer.Stemmer("english")


def clean_text(doc: str) -> str:
    lower_case = str(doc).lower()
    substituted = email_re.sub(" <EMAIL> ", lower_case)
    substituted = date_re.sub(" <DATE> ", substituted)
    substituted = url_re.sub(" <URL> ", substituted)
    substituted = num_re.sub(" <NUM> ", substituted)
    no_specials = special_re.sub(" ", substituted)
    no_apostrophes = apostrephe_re.sub("", no_specials)
    cleaned = ws_re.sub(" ", no_apostrophes).strip()
    return cleaned


def clean(df: pl.DataFrame, nlc: list[str] = natural_language_cols) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col)
            .map_elements(lambda x: clean_text(x), return_dtype=pl.String)
            .alias(col)
            for col in nlc
            if col in df.columns
        ]
    )


def rm_stop_words(
    df: pl.DataFrame, nlc: list[str] = natural_language_cols
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col)
            .map_elements(
                lambda x: " ".join([w for w in str(x).split() if w not in _stop_words]),
                return_dtype=pl.String,
            )
            .alias(col)
            for col in nlc
            if col in df.columns
        ]
    )


def stem(df: pl.DataFrame, nlc: list[str] = natural_language_cols) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col)
            .map_elements(
                lambda x: " ".join([_porter_stemmer.stem(w) for w in str(x).split()]),
                return_dtype=pl.String,
            )
            .alias(col)
            for col in nlc
            if col in df.columns
        ]
    )


def preprocess(
    df: pl.DataFrame, nlc: list[str] = natural_language_cols
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col)
            .map_elements(
                lambda x: " ".join(
                    [
                        _porter_stemmer.stem(w)
                        for w in clean_text(str(x)).split()
                        if w not in _stop_words
                    ]
                ),
                return_dtype=pl.String,
            )
            .alias(col)
            for col in nlc
            if col in df.columns
        ]
    )


def stem_and_filter(text: str) -> str:
    words = [w for w in str(text).split() if w not in _stop_words]
    return " ".join(_pystemmer.stemWords(words))


def preprocess_text(text: str) -> str:
    return stem_and_filter(clean_text(text))


def preprocess_chunk(chunk: pl.DataFrame) -> pl.DataFrame:
    chunk = chunk.fill_null("nan")
    if "meta_keywords" in chunk.columns:
        chunk = chunk.with_columns(
            pl.when(pl.col("meta_keywords") == "['']")
            .then(pl.lit("nan"))
            .otherwise(pl.col("meta_keywords"))
            .alias("meta_keywords")
        )

    available_cols = [col for col in natural_language_cols if col in chunk.columns]
    if not available_cols:
        return chunk

    chunk = chunk.with_columns(
        [
            pl.col(col)
            .str.to_lowercase()
            .str.replace_all(email_str, " <EMAIL> ")
            .str.replace_all(date_str, " <DATE> ")
            .str.replace_all(url_str, " <URL> ")
            .str.replace_all(num_str, " <NUM> ")
            .str.replace_all(special_str, " ")
            .str.replace_all(apostrephe_str, "")
            .str.replace_all(ws_str, " ")
            .str.strip_chars()
            for col in available_cols
        ]
    )
    chunk = chunk.with_columns(
        [
            pl.col(col).map_elements(stem_and_filter, return_dtype=pl.String).alias(col)
            for col in available_cols
        ]
    )
    return chunk


def preprocess_file(
    filepath: str,
    outpath: str,
    chunk_size: int = 50_000,
    drop_columns: list[str] | None = None,
) -> str:
    drop_columns = drop_columns or ["id", "Unnamed: 0", "keywords", "summary"]

    if os.path.exists(outpath):
        os.remove(outpath)

    lf = pl.scan_csv(
        filepath,
        infer_schema_length=10000,
        truncate_ragged_lines=True,
        schema_overrides={"Unnamed: 0": pl.String},
    )
    total_rows = lf.select(pl.len()).collect().item()
    n_chunks = (total_rows + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        offset = i * chunk_size
        chunk = lf.slice(offset, chunk_size).collect()
        existing_drop_columns = [col for col in drop_columns if col in chunk.columns]
        if existing_drop_columns:
            chunk = chunk.drop(existing_drop_columns)
        chunk = preprocess_chunk(chunk)

        if i == 0:
            chunk.write_csv(outpath)
        else:
            with open(outpath, "a", encoding="utf-8") as handle:
                chunk.write_csv(handle, include_header=False)

        sys.stdout.write(
            f"\r{i + 1}/{n_chunks} chunks processed - {offset + len(chunk)} rows so far"
        )
        sys.stdout.flush()

    print(f"\nDone. Preprocessed data saved to {outpath}")
    return outpath
