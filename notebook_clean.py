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
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer, alt, np, pl


@app.cell
def _(pl):
    filepath = "news/data/995,000_rows.csv"

    # Force all columns to string so Polars doesn't mis-infer types
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

    # Drop columns irrelevant to fake-news classification
    drop_cols = [
        c
        for c in [
            "id",
            "Unnamed: 0",
            "url",
            "inserted_at",
            "updated_at",
            "keywords",
            "summary",
            "source",
        ]
        if c in df_raw.columns
    ]
    df_raw = df_raw.drop(drop_cols)#.head(10_000)
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
    print(f"Duplicates removed: {n_before - n_after} ({n_before} -> {n_after})")

    df = df_deduped.with_columns(
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
def _(df, pl):
    text_cols = ["content", "title", "meta_description"]
    other_cols = ["authors", "meta_keywords"]

    df_cleaned = df.with_columns(
        *[
            pl.col(col)
            .str.replace_all(r"<[^>]+>", " ")          # strip HTML
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"[\u2018\u2019\u201C\u201D]", "'")  # normalise quotes
            .str.to_lowercase()
            .str.strip_chars()
            .alias(col)
            for col in text_cols
        ],
    ).filter(
        pl.col("content").is_not_null() & (pl.col("content").str.len_chars() > 0)
    )

    # Second pass filter (safety net) to ensure no empty content slips through
    df_cleaned = df_cleaned.filter(
        pl.col("content").is_not_null() & (pl.col("content").str.len_chars() > 0)
    )

    # Preview rows that have meta_keywords for a quick sanity check
    df_cleaned.filter(pl.col("meta_keywords").is_not_null()).select(
        text_cols + other_cols
    ).head()
    return (df_cleaned,)


@app.cell
def _(df_cleaned, np):
    from sklearn.model_selection import train_test_split

    n = df_cleaned.shape[0]
    indices = np.arange(n)

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )

    df_train = df_cleaned[train_idx.tolist()]
    df_val = df_cleaned[val_idx.tolist()]
    df_test = df_cleaned[test_idx.tolist()]

    print(f"Train: {df_train.shape[0]}  ({df_train.shape[0]/n:.0%})")
    print(f"Val:   {df_val.shape[0]}  ({df_val.shape[0]/n:.0%})")
    print(f"Test:  {df_test.shape[0]}  ({df_test.shape[0]/n:.0%})")
    return df_test, df_train, df_val


@app.cell
def _(TfidfVectorizer, df_test, df_train, df_val):
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        min_df=5,
        max_df=0.95,
        stop_words="english",
        token_pattern=r"[a-zA-Z]{2,}",
    )

    # fit on train, transform all three splits
    tfidf_train = vectorizer.fit_transform(df_train["content"].to_list())
    tfidf_val = vectorizer.transform(df_val["content"].to_list())
    tfidf_test = vectorizer.transform(df_test["content"].to_list())

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF train: {tfidf_train.shape}")
    print(f"TF-IDF val:   {tfidf_val.shape}")
    print(f"TF-IDF test:  {tfidf_test.shape}")
    return tfidf_train, vectorizer


@app.cell
def _(alt, np, pl, tfidf_train, vectorizer):
    # Sum TF-IDF scores per term across all training documents
    tfidf_sums = np.asarray(tfidf_train.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    # Build a small DataFrame of the top 50 terms for plotting
    vocab = (
        pl.DataFrame(
            {
                "word": feature_names,
                "tfidf_sum": tfidf_sums,
            }
        )
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
