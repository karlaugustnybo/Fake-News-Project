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
    import re
    import contractions
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer, alt, contractions, pl, re


@app.cell
def _(pl):
    filepath = "news/data/news_sample.csv"

    df_raw = pl.read_csv(filepath)
    df_raw = df_raw.drop(["id", "domain"])
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
def _(contractions, re):
    def clean_text(text: str | None) -> str | None:
        if text is None:
            return None
        text = text.lower()
        text = contractions.fix(text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [w for w in text.split() if len(w) > 1]
        return " ".join(tokens)

    return (clean_text,)


@app.cell
def _(clean_text, df, pl):
    text_cols = ["content", "title", "meta_description"]
    other_cols = ["authors", "meta_keywords"]

    df_cleaned = df.with_columns(
        *[
            pl.col(col)
            .map_elements(clean_text, return_dtype=pl.Utf8)
            .alias(col)
            for col in text_cols + other_cols
        ],
    )

    df_cleaned = df_cleaned.filter(pl.col("content").is_not_null())

    df_cleaned.filter(pl.col("meta_keywords").is_not_null()).select(
        text_cols + other_cols
    ).head()
    return (df_cleaned,)


@app.cell
def _(TfidfVectorizer, df_cleaned):
    # TODO: This currently fits TF-IDF on the full dataset. Before modeling,
    # split into train/test first, then fit_transform on train and transform
    # on test to avoid data leakage through IDF weights.

    tfidf_params = dict(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )

    vectorizer = TfidfVectorizer(**tfidf_params)

    content = df_cleaned["content"].to_list()

    tfidf_matrix = vectorizer.fit_transform(content)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF shape:    {tfidf_matrix.shape}")
    return content, vectorizer


@app.cell
def _(alt, content, pl, vectorizer):
    from collections import Counter

    feature_names = set(vectorizer.get_feature_names_out())
    all_tokens = " ".join(content).split()
    filtered_counts = Counter(t for t in all_tokens if t in feature_names)

    vocab = (
        pl.DataFrame({
            "word": list(filtered_counts.keys()),
            "count": list(filtered_counts.values()),
        })
        .sort("count", descending=True)
        .head(50)
    )

    chart = (
        alt.Chart(vocab)
        .mark_bar()
        .encode(
            x=alt.X("word:N", sort=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("count:Q"),
        )
        .properties(width=900, height=300)
    )
    chart
    return


if __name__ == "__main__":
    app.run()