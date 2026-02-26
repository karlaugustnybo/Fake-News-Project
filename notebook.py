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
    #!uv add polars numpy matplotlib seaborn nltk
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import altair as alt
    import nltk
    import re

    return alt, pl, re


@app.cell
def _(pl):
    filepath = "news/data/news_sample.csv"

    df_raw = pl.read_csv(filepath).drop(
        [
            c
            for i, c in enumerate(pl.read_csv(filepath, n_rows=0).columns)
            if c == "id" or i == 1
        ]
    )
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
    df = (
        df_raw.drop(["keywords", "summary"])
        .fill_null("nan")
        .with_columns(
            pl.when(pl.col("meta_keywords") == "['']")
            .then(pl.lit("nan"))
            .otherwise(pl.col("meta_keywords"))
            .alias("meta_keywords")
        )
    )
    return (df,)


@app.cell
def _(df, pl):
    desc = df.describe()
    desc.filter(~pl.col("statistic").is_in(["count", "null_count"]))
    return


@app.cell
def _(re):
    # clean function
    def clean_text(doc: str):
        special_tokens = ["<NUM>", "<DATE>", "<EMAIL>", "<URL>"]
        lines = doc.split("\n")

        date_ptn = re.compile(r"[0-9]{1,2}[ -/]?[0-9]{1,2}[ -/]?[0-9]{2,4}")
        email_ptn = re.compile(r"[^ \n:;,.]+@[a-z0-9\.]+\.[^ \n:;,.]+")
        url_ptn = re.compile(r"[htps]{,5}:?/{,2}[a-zA-Z0-9]+\.[^ ]+")
        num_ptn = re.compile(r"[0-9][0-9,^]*\.?[0-9]*")

        out = []
        for line in lines:
            # simplifying
            line = line.split()
            line = [x.replace("'", "") for x in line]  # apostrophes
            line = [
                x.strip(",.;:()!@#$%^&*/[]-_\"?*+-=<>{} `~|¡ºª¢€ħðþ‘'“”\n\t–")
                for x in line
            ]
            line = [
                x for x in line if len(x) > 0
            ]  # remove empty string (includes double spaces)
            line = [x.lower() for x in line]

            # pattern matching
            line = re.sub(date_ptn, "<DATE>", " ".join(line))
            line = re.sub(email_ptn, "<EMAIL>", line)
            line = re.sub(url_ptn, "<URL>", line)
            line = (
                re.sub(num_ptn, "<NUM>", line).split()
            )  # <NUM> last because dates,emails, and urls may contain numbers

            # joining back and adding <OTHER> token
            # decided to keep word1-word2 as a new word
            line = " ".join(
                [
                    x
                    if x.replace("-", "")
                    .replace("'", "")
                    .replace("'", "")
                    .isalpha()
                    or x in special_tokens
                    else "<OTHER>"
                    for x in line
                ]
            )
            out.append(line)

        return "\n".join(out).replace("\n\n", "\n")

    return (clean_text,)


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
    df2 = df.with_columns(
        [
            pl.col(col).map_elements(clean_text, return_dtype=pl.Utf8).alias(col)
            for col in natural_language_cols
        ]
    )
    df2.filter(pl.col("meta_keywords") != "nan").select(
        natural_language_cols
    ).head()
    return df2, natural_language_cols


@app.cell
def _(df2, natural_language_cols, pl):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stop_words = list(set(stopwords.words("english")))
    stemmer = PorterStemmer()


    def remove_stopwords(text: str) -> str:
        return " ".join([w for w in text.split() if w not in stop_words])


    def stem_text(text: str) -> str:
        return " ".join([stemmer.stem(w) for w in text.split()])


    df3 = df2.with_columns(
        [
            pl.col(col)
            .map_elements(remove_stopwords, return_dtype=pl.Utf8)
            .map_elements(stem_text, return_dtype=pl.Utf8)
            .alias(col)
            for col in natural_language_cols
        ]
    )
    df3.select(natural_language_cols)
    return (df3,)


@app.cell
def _(alt, df3, natural_language_cols, pl):
    all_text = " ".join(
        df3.select(natural_language_cols).unpivot().get_column("value").to_list()
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
