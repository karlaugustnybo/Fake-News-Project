import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><h1 style='text-align: center;'>Fake News</h1><br>
    # Part 1 - Data Processing
    ### Task 1 - Small sample
    """)
    return


@app.cell
def _():
    #!uv add polars numpy matplotlib seaborn nltk
    return


@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    # nltk.download('stopwords')
    plt.style.use("ggplot")
    return PorterStemmer, pl, plt, re, stopwords


@app.cell
def _(pl):
    _filepath = "news/data/news_sample.csv"
    df = pl.read_csv(_filepath).drop([""])  # drop the unnamed first column
    # Read data
    df
    return (df,)


@app.cell
def _(df):
    print(df.schema)
    print(f"Shape: {df.shape}")
    print(f"Null counts:\n{df.null_count()}")
    return


@app.cell
def _(df, pl):
    df_1 = (
        df.drop(["keywords", "summary"])
        .fill_null("nan")
        .with_columns(
            pl.when(pl.col("meta_keywords") == "['']")
            .then(pl.lit("nan"))
            .otherwise(pl.col("meta_keywords"))
            .alias("meta_keywords")
        )
    )
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1.describe()
    return


@app.cell
def _(re):
    # Helper function to collect tokens labeled as <OTHER>
    def _collect_other(token, log):
        if log is not None:
            log.append(token)
        return "<OTHER>"


    special_tokens = ["<NUM>", "<DATE>", "<EMAIL>", "<URL>"]
    date_ptn = re.compile("[0-9]{1,2}[ -/]?[0-9]{1,2}[ -/]?[0-9]{2,4}")
    # Compile regex and init data structures before function to reduce overhead
    email_ptn = re.compile("[^ \\n:;,.]+@[a-z0-9\\.]+\\.[^ \\n:;,.]+")
    url_ptn = re.compile("[htps]{,5}:?/{,2}[a-zA-Z0-9]+\\.[^ ]+")
    num_ptn = re.compile("[0-9][0-9,^]*\\.?[0-9]*")

    # Pattern strings — shared with Task 2 (Polars uses these directly)
    email_pat = r'[^ \n,"]+@[^ \n,"]+\.[^ \n,"]+'
    date_pat = r'[0-9]{2,4}[-/][0-9]{2,4}[-/][0-9]{2,4}'
    url_pat = r'(?:http)?s?(?://)?[^ \n,"]+\.[a-z]{2,}[^ \n,"]+'
    num_pat = r'[0-9]+[,.]?[0-9]*'
    special_pat = r"""[.,/—–!\\"#$%&'()*+:;=?@\[\\\]^_`{|}~€£¥§±×÷°•¶©®™¢∞≠≈≤≥√∑πµω∆∫]"""
    ws_pat = r'\s+'

    # Compile from the shared strings
    _email_re = re.compile(email_pat)
    _date_re = re.compile(date_pat)
    _url_re = re.compile(url_pat)
    _num_re = re.compile(num_pat)
    _special_re = re.compile(special_pat)
    _ws_re = re.compile(ws_pat)


    def clean_text(doc: str):
        lower_case = doc.lower()
        substituted = _email_re.sub(" <EMAIL> ", lower_case)
        substituted = _date_re.sub(" <DATE> ", substituted)
        substituted = _url_re.sub(" <URL> ", substituted)
        substituted = _num_re.sub(" <NUM> ", substituted)
        no_specials = _special_re.sub(" ", substituted)
        cleaned = _ws_re.sub(' ', no_specials).strip()
        return cleaned

    return (
        clean_text,
        date_pat,
        email_pat,
        num_pat,
        special_pat,
        url_pat,
        ws_pat,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><br><br>
    <h1 style='text-align : center;'>CHECK OTHER</h1>
    <br><br><br>
    """)
    return


@app.cell
def _():
    # # Check for most common tokens replaced by <OTHER>
    # other_tokens = []
    # for doc in df["content"].dropna().head(500):
    #     clean_text(doc, other_log=other_tokens)

    # # Show most common tokens replaced by <OTHER>
    # other_counts = pd.Series(other_tokens).value_counts()[:20]

    # print(f"Unique tokens mapped to <OTHER>: {len(other_counts)}")
    # other_counts
    return


@app.cell
def _(clean_text, df_1, pl):
    natural_language_cols = [
        "content",
        "title",
        "authors",
        "meta_keywords",
        "meta_description",
    ]


    def clean(df: pl.DataFrame, nlc: list[str] = natural_language_cols):
        # Given a DataFrame and potentially
        #   1. a list, nlc, of natural language columns (default: "natural_language_cols")
        # Returns new DataFrame with nlc cleaned.
        return df.with_columns(
            [
                pl.col(col)
                .map_elements(lambda x: clean_text(x), return_dtype=pl.String)
                .alias(col)
                for col in nlc
            ]
        )


    # df = clean(df)
    df_1.filter(pl.col("meta_keywords") != "nan").select(
        natural_language_cols
    ).head()
    return (natural_language_cols,)


@app.cell
def _(PorterStemmer, clean_text, df_1, natural_language_cols, pl, stopwords):
    def rm_stop_words(df: pl.DataFrame, nlc: list[str] = natural_language_cols):
        #Removes stop words from DataFrame with i/o as for the clean function.
        _stop_words = set(stopwords.words("english"))
        return df.with_columns(
            [
                pl.col(col)
                .map_elements(
                    lambda x: " ".join(
                        [w for w in x.split() if w not in _stop_words]
                    ),
                    return_dtype=pl.String,
                )
                .alias(col)
                for col in nlc
            ]
        )

    def stem(df: pl.DataFrame, nlc: list[str] = natural_language_cols):
        #Stems DataFrame with i/o as for the clean function.
        _stemmer = PorterStemmer()
        return df.with_columns(
            [
                pl.col(col)
                .map_elements(
                    lambda x: " ".join([_stemmer.stem(w) for w in x.split()]),
                    return_dtype=pl.String,
                )
                .alias(col)
                for col in nlc
            ]
        )

    def preprocess(
        df: pl.DataFrame, nlc: list[str] = natural_language_cols, progbar=False
    ):
        # General preprocessing function — faster than combining clean, rm_stop_words, stem.
        _stop_words = set(stopwords.words("english"))
        _stemmer = PorterStemmer()
        return df.with_columns(
            [
                pl.col(col)
                .map_elements(
                    lambda x: " ".join(
                        [
                            _stemmer.stem(w)
                            for w in clean_text(x).split()
                            if w not in _stop_words
                        ]
                    ),
                    return_dtype=pl.String,
                )
                .alias(col)
                for col in nlc
            ]
        )

    df_clean = preprocess(df_1)
    df_clean.select(natural_language_cols)
    return (df_clean,)


@app.cell
def _(df_clean, natural_language_cols, plt):
    # Make Zipf's law plot of the 50 most common tokens after preprocessing
    # Gather all text from NL columns, split into tokens, count
    all_text = " ".join(
        df_clean.select(natural_language_cols)
        .fill_null("")
        .unpivot()
        .get_column("value")
        .to_list()
    )
    from collections import Counter

    token_counts = Counter(all_text.split())
    most_common = token_counts.most_common(50)
    words, counts = zip(*most_common)

    plt.figure(figsize=(24, 8))
    plt.plot(list(words), list(counts))
    plt.xticks(rotation=35, ha="right")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 2 - Bigger subset
    """)
    return


@app.cell
def _(
    date_pat,
    email_pat,
    natural_language_cols,
    num_pat,
    pl,
    special_pat,
    stopwords,
    url_pat,
    ws_pat,
):
    import os
    import Stemmer  # PyStemmer (~10× faster than NLTK)

    _filepath = "news/data/995,000_rows.csv"
    _outpath = "news/data/995,000_rows_preprocessed.csv"
    _chunk_size = 50_000

    # Preprocess resources (initialized once)
    _stop_words = set(stopwords.words("english"))
    _stemmer = Stemmer.Stemmer("english")

    def _stem_and_filter(x: str) -> str:
        words = [w for w in x.split() if w not in _stop_words]
        return " ".join(_stemmer.stemWords(words))

    # Remove old output file if it exists so we can append fresh
    if os.path.exists(_outpath):
        os.remove(_outpath)

    # Use scan_csv (lazy) and process in chunks to keep memory low
    _lf = pl.scan_csv(_filepath)
    _total_rows = _lf.select(pl.len()).collect().item()
    _n_chunks = (_total_rows + _chunk_size - 1) // _chunk_size
    print(
        f"Processing {_total_rows} rows in {_n_chunks} chunks of {_chunk_size}..."
    )

    for _i in range(_n_chunks):
        _offset = _i * _chunk_size
        # Collect only this chunk from the lazy frame
        _chunk = (
            _lf.slice(_offset, _chunk_size)
            .drop(["id", "Unnamed: 0", "keywords", "summary"])
            .collect()
        )
        # Fill nulls
        _chunk = _chunk.fill_null("nan")
        _chunk = _chunk.with_columns(
            pl.when(pl.col("meta_keywords") == "['']")
            .then(pl.lit("nan"))
            .otherwise(pl.col("meta_keywords"))
            .alias("meta_keywords")
        )
        # --- Regex cleaning via native Polars (runs in Rust, not Python) ---
        _chunk = _chunk.with_columns(
            [
                pl.col(col)
                .str.to_lowercase()
                .str.replace_all(email_pat, " <EMAIL> ")
                .str.replace_all(date_pat, " <DATE> ")
                .str.replace_all(url_pat, " <URL> ")
                .str.replace_all(num_pat, " <NUM> ")
                .str.replace_all(special_pat, " ")
                .str.replace_all(ws_pat, " ")
                .str.strip_chars()
                for col in natural_language_cols
            ]
        )
        # --- Stop-word removal + stemming (PyStemmer is C-based, ~10× faster) ---
        _chunk = _chunk.with_columns(
            [
                pl.col(col)
                .map_elements(_stem_and_filter, return_dtype=pl.String)
                .alias(col)
                for col in natural_language_cols
            ]
        )
        # Write: first chunk with header (creates file), rest append without header
        if _i == 0:
            _chunk.write_csv(_outpath)
        else:
            with open(_outpath, "a") as _f:
                _chunk.write_csv(_f, include_header=False)
        print(
            f"  Chunk {_i + 1}/{_n_chunks} done ({_offset + len(_chunk)} rows so far)"
        )
        del _chunk  # Free memory immediately

    print("Done. Preprocessed data saved to", _outpath)
    return


@app.cell
def _(pl):
    df_final = pl.read_csv("news/data/995,000_rows_preprocessed.csv")
    df_final
    return


if __name__ == "__main__":
    app.run()
