import marimo

__generated_with = "0.20.2"
app = marimo.App(sql_output="lazy-polars")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><h1 style='text-align: center;'>Fake News</h1><br>
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    import numpy as np
    import re
    import contractions
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer, alt, contractions, np, pl, re


@app.cell
def _(pl):
    filepath = "news/data/995,000_rows.csv"

    schema = {
        "type": pl.Utf8,
        "url": pl.Utf8,
        "content": pl.Utf8,
        "scraped_at": pl.Utf8,
        "inserted_at": pl.Utf8,
        "updated_at": pl.Utf8,
        "title": pl.Utf8,
        "authors": pl.Utf8,
        "keywords": pl.Utf8,
        "meta_keywords": pl.Utf8,
        "meta_description": pl.Utf8,
        "tags": pl.Utf8,
        "summary": pl.Utf8,
        "source": pl.Utf8,
        "id": pl.Utf8,
        "domain": pl.Utf8,
        "Unnamed: 0": pl.Utf8,
    }

    df_raw = pl.read_csv(filepath, schema_overrides=schema)
    drop_cols = [
        c
        for c in [
            "id",
            "domain",
            "Unnamed: 0",
            "url",
            "scraped_at",
            "inserted_at",
            "updated_at",
            "keywords",
            "summary",
            "tags",
            "source",
        ]
        if c in df_raw.columns
    ]
    df_raw = df_raw.drop(drop_cols).head(100_000)
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    print("Schema:")
    for name, dtype in df_raw.schema.items():
        print(f"  {name}: {dtype}")
    print(f"\nShape: {df_raw.shape}")
    print(f"Null counts:\n{df_raw.null_count()}")
    return


@app.cell
def _(df_raw, pl):
    n_before = df_raw.shape[0]
    df_deduped = df_raw.unique(subset=["content"])
    n_after = df_deduped.shape[0]
    print(
        f"Duplicates removed: {n_before - n_after} "
        f"({n_before} -> {n_after})"
    )

    df = df_deduped.drop(["keywords", "summary"]).with_columns(
        pl.when(pl.col("meta_keywords") == "['']")
        .then(pl.lit(None))
        .otherwise(pl.col("meta_keywords"))
        .alias("meta_keywords")
    )
    return (df,)


@app.cell
def _(df, pl):
    if "type" in df.columns:
        label_col = "type"
    elif "label" in df.columns:
        label_col = "label"
    else:
        label_col = None

    if label_col:
        print(f"Label column: '{label_col}'")
        print(
            df.group_by(label_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
    else:
        print("No label column found. Available columns:", df.columns)
    return


@app.cell
def _(contractions):
    c_patterns = [
        k.lower() for k in contractions.contractions_dict.keys()
    ]
    c_replacements = [
        v.lower() for v in contractions.contractions_dict.values()
    ]
    return c_patterns, c_replacements


@app.cell
def _(c_patterns, c_replacements, df, pl):
    text_cols = ["content", "title", "meta_description"]
    other_cols = ["authors", "meta_keywords"]

    df_cleaned = df.with_columns(
        *[
            pl.col(col)
            .str.to_lowercase()
            .str.replace_many(c_patterns, c_replacements)
            .str.strip_chars()
            .alias(col)
            for col in text_cols
        ],
        *[
            pl.col(col)
            .str.to_lowercase()
            .str.strip_chars()
            .alias(col)
            for col in other_cols
        ],
    )

    df_cleaned = df_cleaned.filter(
        pl.col("content").is_not_null()
        & (pl.col("content").str.len_chars() > 0)
    )

    df_cleaned.filter(pl.col("meta_keywords").is_not_null()).select(
        text_cols + other_cols
    ).head()
    return (df_cleaned,)


@app.cell
def _(TfidfVectorizer, df_cleaned, re):
    # TODO: This currently fits TF-IDF on the full dataset. Before modeling,
    # split into train/test first, then fit_transform on train and transform
    # on test to avoid data leakage through IDF weights.

    _url_pattern = re.compile(r"https?://\S+|www\.\S+")
    _token_pattern = re.compile(r"[a-zA-Z]{2,}")

    def custom_tokenizer(text: str) -> list[str]:
        text = _url_pattern.sub("", text)
        return _token_pattern.findall(text)

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        min_df=5,
        max_df=0.95,
        stop_words="english",
        tokenizer=custom_tokenizer,
        token_pattern=None,
    )

    content = df_cleaned["content"].to_list()
    tfidf_matrix = vectorizer.fit_transform(content)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF shape:    {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


@app.cell
def _(alt, np, pl, tfidf_matrix, vectorizer):
    # Sum TF-IDF scores per term across all documents.
    tfidf_sums = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    vocab = (
        pl.DataFrame({
            "word": feature_names,
            "tfidf_sum": tfidf_sums,
        })
        .sort("tfidf_sum", descending=True)
        .head(50)
    )

    chart = (
        alt.Chart(vocab)
        .mark_bar()
        .encode(
            x=alt.X("word:N", sort=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("tfidf_sum:Q", title="Aggregate TF-IDF score"),
        )
        .properties(width=900, height=300)
    )
    chart
    return


if __name__ == "__main__":
    app.run()
