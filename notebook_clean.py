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
    import Stemmer

    return Stemmer, TfidfVectorizer, alt, contractions, pl, re


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
def _(Stemmer, contractions, re):
    stemmer = Stemmer.Stemmer("english")

    def clean_text(text: str | None, stem: bool = True) -> str | None:
        if text is None:
            return None
        text = text.lower()
        text = contractions.fix(text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [w for w in text.split() if len(w) > 1]
        if stem:
            tokens = stemmer.stemWords(tokens)
        return " ".join(tokens)

    return clean_text, stemmer


@app.cell
def _(clean_text, df, pl):
    stem_cols = ["content", "title", "meta_description"]
    no_stem_cols = ["authors", "meta_keywords"]

    df_cleaned = df.with_columns(
        *[
            pl.col(col)
            .map_elements(
                lambda x: clean_text(x, stem=True), return_dtype=pl.Utf8
            )
            .alias(col)
            for col in stem_cols
        ],
        *[
            pl.col(col)
            .map_elements(
                lambda x: clean_text(x, stem=False), return_dtype=pl.Utf8
            )
            .alias(f"{col}_unstemmed")
            for col in stem_cols
        ],
        *[
            pl.col(col)
            .map_elements(
                lambda x: clean_text(x, stem=False), return_dtype=pl.Utf8
            )
            .alias(col)
            for col in no_stem_cols
        ],
    )

    df_cleaned = df_cleaned.filter(pl.col("content").is_not_null())

    df_cleaned.filter(pl.col("meta_keywords").is_not_null()).select(
        stem_cols + [f"{c}_unstemmed" for c in stem_cols] + no_stem_cols
    ).head()
    return (df_cleaned,)


@app.cell
def _(TfidfVectorizer, df_cleaned, stemmer):
    # TODO: This currently fits TF-IDF on the full dataset. Before modeling,
    # split into train/test first, then fit_transform on train and transform
    # on test to avoid data leakage through IDF weights.

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stemmed_stopwords = list(
        set(stemmer.stemWords(list(ENGLISH_STOP_WORDS)))
    )

    tfidf_params = dict(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )

    vectorizer_stemmed = TfidfVectorizer(
        stop_words=stemmed_stopwords, **tfidf_params
    )
    vectorizer_unstemmed = TfidfVectorizer(
        stop_words="english", **tfidf_params
    )

    content_stemmed = df_cleaned["content"].to_list()
    content_unstemmed = df_cleaned["content_unstemmed"].to_list()

    tfidf_stemmed = vectorizer_stemmed.fit_transform(content_stemmed)
    tfidf_unstemmed = vectorizer_unstemmed.fit_transform(content_unstemmed)

    vocab_stemmed = len(vectorizer_stemmed.vocabulary_)
    vocab_unstemmed = len(vectorizer_unstemmed.vocabulary_)
    reduction = (1 - vocab_stemmed / vocab_unstemmed) * 100

    print(f"Stemmed vocabulary size:   {vocab_stemmed}")
    print(f"Unstemmed vocabulary size:  {vocab_unstemmed}")
    print(f"Reduction from stemming:    {reduction:.1f}%")
    print(f"\nStemmed TF-IDF shape:   {tfidf_stemmed.shape}")
    print(f"Unstemmed TF-IDF shape: {tfidf_unstemmed.shape}")
    return content_stemmed, vectorizer_stemmed


@app.cell
def _(alt, content_stemmed, pl, vectorizer_stemmed):
    from collections import Counter

    feature_names = set(vectorizer_stemmed.get_feature_names_out())
    all_tokens = " ".join(content_stemmed).split()
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


@app.cell(hide_code=True)
def _(clean_text, pl):
    edge_cases = [
        "Don't worry, it's fine!",
        "The U.S. economy grew 3.5% in Q4.",
        "Contact john.doe@example.com or visit https://example.com",
        "COVID-19 and H1N1 are both viruses.",
        "She paid $1,500.00 for the car.",
        "The 1st, 2nd, and 3rd place winners.",
        "John's car is better than Mary's.",
        "well-known self-driving state-of-the-art",
        "Running runs ran runner",
        "I'm you're we've they'd",
        "   multiple   spaces   and\n\nnewlines\t\ttabs",
        "Cafe, resume, naive, facade",
        "emoji test and symbols",
    ]

    edge_case_df = pl.DataFrame({
        "original": edge_cases,
        "cleaned_stemmed": [clean_text(t, stem=True) for t in edge_cases],
        "cleaned_unstemmed": [
            clean_text(t, stem=False) for t in edge_cases
        ],
    })
    edge_case_df
    return


if __name__ == "__main__":
    app.run()