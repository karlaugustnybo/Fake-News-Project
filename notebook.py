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

    return alt, pl, re


@app.cell
def _(pl):
    filepath = "news/data/news_sample.csv"

    df_raw = pl.read_csv(filepath)
    drop_cols = [
        c for i, c in enumerate(df_raw.columns) if c == "id" or i == 1
    ]
    df_raw = df_raw.drop(drop_cols)
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
    df = df_raw.drop(["keywords", "summary"]).with_columns(
        pl.when(pl.col("meta_keywords") == "['']")
        .then(pl.lit(None))
        .otherwise(pl.col("meta_keywords"))
        .alias("meta_keywords")
    )
    return (df,)


@app.cell
def _(df, pl):
    desc = df.describe()
    desc.filter(~pl.col("statistic").is_in(["count", "null_count"]))
    return


@app.cell
def _(re):
    _DATE_PTN = re.compile(
        r"\b\d{1,2}[/ -]\d{1,2}[/ -]\d{2,4}\b"
        r"|\b\d{4}[/ -]\d{1,2}[/ -]\d{1,2}\b"
    )
    _EMAIL_PTN = re.compile(r"[^\s\n:;,.]+@[a-z0-9.]+\.[^\s\n:;,.]+")
    _URL_PTN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
    _NUM_PTN = re.compile(r"\b\d[\d,]*\.?\d*\b")
    _PUNCT_RE = re.compile(r"[^\w\s<>-]")
    _MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

    _SPECIAL_TOKENS = {"<NUM>", "<DATE>", "<EMAIL>", "<URL>", "<OTHER>"}

    SINGLE_CHAR_PATTERN = r"\b\w\b"

    def clean_text(doc: str) -> str:
        """Clean a document string and replace entities with special tokens.

        Token contract
        --------------
        <DATE>   - date-like patterns  (e.g. 01/02/2024, 2024-01-02)
        <EMAIL>  - email addresses
        <URL>    - http(s) / www URLs
        <NUM>    - standalone numbers (applied last so dates/emails/URLs
                   are already replaced)
        <OTHER>  - any remaining non-alphabetic, non-special token
        """
        lines = doc.split("\n")
        out = []

        for line in lines:
            tokens = line.split()
            tokens = [_PUNCT_RE.sub("", t).lower() for t in tokens]
            tokens = [t for t in tokens if len(t) > 0]

            joined = " ".join(tokens)

            joined = _DATE_PTN.sub("<DATE>", joined)
            joined = _EMAIL_PTN.sub("<EMAIL>", joined)
            joined = _URL_PTN.sub("<URL>", joined)
            joined = _NUM_PTN.sub("<NUM>", joined)

            final_tokens = []
            for t in joined.split():
                if t in _SPECIAL_TOKENS:
                    final_tokens.append(t)
                elif t.replace("-", "").isalpha():
                    final_tokens.append(t)
                else:
                    final_tokens.append("<OTHER>")

            out.append(" ".join(final_tokens))

        result = "\n".join(out)
        result = _MULTI_NEWLINE_RE.sub("\n", result)
        return result

    return SINGLE_CHAR_PATTERN, clean_text


@app.cell
def _(df, df2, pl):
    """Show the original tokens that got replaced by <OTHER>."""
    from collections import Counter

    other_counter = Counter()

    raw_texts = df["content"].drop_nulls().head(500).to_list()
    clean_texts = df2["content"].drop_nulls().head(500).to_list()

    for raw, cleaned in zip(raw_texts, clean_texts):
        raw_lines = raw.split("\n")
        clean_lines = cleaned.split("\n")

        for raw_line, clean_line in zip(raw_lines, clean_lines):
            raw_tokens = raw_line.split()
            clean_tokens = clean_line.split()

            if len(raw_tokens) != len(clean_tokens):
                continue

            for original, replaced in zip(raw_tokens, clean_tokens):
                if replaced == "<OTHER>":
                    other_counter[original] += 1

    other_log = (
        pl.DataFrame(
            {
                "original_token": list(other_counter.keys()),
                "count": list(other_counter.values()),
            }
        )
        .sort("count", descending=True)
        .head(60)
    )

    print(f"Unique tokens mapped to <OTHER>: {other_log.height}")
    other_log
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><br><br>
    <h1 style='text-align : center;'>CHECK OTHER</h1>
    <br><br><br>
    """)
    return


@app.cell
def _(clean_text, df, pl):
    natural_language_cols = [
        "content",
        "title",
        "authors",
        "meta_keywords",
        "meta_description",
    ]

    _new_columns = {}
    for _col in natural_language_cols:
        _raw_values = df[_col].to_list()
        _new_columns[_col] = [
            clean_text(v) if v is not None else None for v in _raw_values
        ]

    df2 = df.with_columns(
        [
            pl.Series(_col, _new_columns[_col])
            for _col in natural_language_cols
        ]
    )

    df2.filter(pl.col("meta_keywords").is_not_null()).select(
        natural_language_cols
    ).head()
    return df2, natural_language_cols


@app.cell
def _(SINGLE_CHAR_PATTERN, df2, natural_language_cols, pl):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    stemmed_stop_words = set(
        stemmer.stem(w) for w in stopwords.words("english")
    )

    def remove_stopwords_and_stem(text: str) -> str:
        """Stem every token and drop stemmed stopwords in one pass."""
        return " ".join(
            stemmer.stem(w)
            for w in text.split()
            if stemmer.stem(w) not in stemmed_stop_words
        )

    _new_columns = {}
    for _col in natural_language_cols:
        _raw_values = df2[_col].to_list()
        _new_columns[_col] = [
            remove_stopwords_and_stem(v) if v is not None else None
            for v in _raw_values
        ]

    df3 = df2.with_columns(
        [
            pl.Series(_col, _new_columns[_col])
            for _col in natural_language_cols
        ]
    )

    df3 = df3.with_columns(
        [
            pl.col(_col)
            .str.replace_all(SINGLE_CHAR_PATTERN, "")
            .str.replace_all(r" {2,}", " ")
            .str.strip_chars(" ")
            .alias(_col)
            for _col in natural_language_cols
        ]
    )

    df3.select(natural_language_cols)
    return (df3,)


@app.cell
def _(alt, df3, natural_language_cols, pl):
    all_text = " ".join(
        df3.select(natural_language_cols)
        .unpivot()
        .drop_nulls("value")
        .get_column("value")
        .to_list()
    )
    vocab = (
        pl.Series("word", all_text.split())
        .value_counts()
        .sort("count", descending=True)
        .head(50)
    )

    chart = (
        alt.Chart(vocab)
        .mark_line(point=True)
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
