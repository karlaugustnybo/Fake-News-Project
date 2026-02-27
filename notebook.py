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
    CONTRACTIONS = {
        "didn't": "did not",
        "don't": "do not",
        "won't": "will not",
        "can't": "cannot",
        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "there's": "there is",
        "we'll": "we will",
        "they'll": "they will",
        "i'll": "i will",
        "you'll": "you will",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "let's": "let us",
        "who's": "who is",
        "what's": "what is",
        "here's": "here is",
        "how's": "how is",
        "where's": "where is",
    }

    CONTRACTION_PTN = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
        re.IGNORECASE,
    )

    ABBREVIATIONS = {
        "u.s.": "united states",
        "u.k.": "united kingdom",
        "u.n.": "united nations",
        "e.u.": "european union",
        "d.c.": "district of columbia",
        "p.o.": "post office",
    }

    ABBREVIATION_PTN = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in ABBREVIATIONS) + r")\b",
        re.IGNORECASE,
    )

    POSSESSIVE_PTN = re.compile(r"'s\b")
    DASH_SPLIT_RE = re.compile(r"[\u2013\u2014]+")

    URL_PTN = re.compile(
        r"https?://[^\s]+"
        r"|www\.[^\s]+"
        r"|pic\.twitter\.com/[^\s]+"
    )
    EMAIL_PTN = re.compile(
        r"[^\s\n:;,.]+@[a-z0-9.]+\.[^\s\n:;,.]+", re.IGNORECASE
    )
    MONEY_PTN = re.compile(r"[\$\u00a3\u20ac]\s?\d[\d,]*\.?\d*")
    DATE_PTN = re.compile(
        r"\b\d{1,2}[/ -]\d{1,2}[/ -]\d{2,4}\b"
        r"|\b\d{4}[/ -]\d{1,2}[/ -]\d{1,2}\b"
    )
    ORDINAL_PTN = re.compile(r"\b\d+(st|nd|rd|th)\b", re.IGNORECASE)

    NUM_PTN = re.compile(r"\b\d[\d,]*\.?\d*\b")
    # No longer preserve < > in general text
    PUNCT_RE = re.compile(r"[^\w\s-]")
    MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
    SINGLE_CHAR_RE = re.compile(r"\b\w\b")

    # Pattern for mixed alphanumeric tokens like co2, mi6, l2tp
    ALNUM_RE = re.compile(r"^[a-z]+\d+[a-z0-9]*$|^\d+[a-z]+[a-z0-9]*$")

    SPECIAL_TOKENS = {
        "<NUM>",
        "<DATE>",
        "<EMAIL>",
        "<URL>",
        "<MONEY>",
        "<OTHER>",
    }

    other_log_list: list[str] = []

    def _expand_contraction(match: re.Match) -> str:
        word = match.group(0).lower()
        replacement = CONTRACTIONS[word]
        if match.group(0)[0].isupper():
            return replacement.capitalize()
        return replacement

    def _expand_abbreviation(match: re.Match) -> str:
        return ABBREVIATIONS[match.group(0).lower()]

    def _pad_special_token(pattern, replacement, text):
        """Replace pattern and ensure whitespace around the token."""
        def _repl(m):
            return f" {replacement} "
        return pattern.sub(_repl, text)

    def _expand_hyphenated(token: str) -> list[str]:
        parts = token.split("-")
        # If every non-empty part is purely alphabetic, keep as-is
        if all(p.isalpha() for p in parts if p):
            return [token]

        out = []
        for p in parts:
            if not p:
                continue
            if p in SPECIAL_TOKENS:
                out.append(p)
            elif p == "<num>":
                # Recover lowercased <NUM>
                out.append("<NUM>")
            elif p.isdigit() or NUM_PTN.fullmatch(p):
                out.append("<NUM>")
            elif p.isalpha():
                out.append(p)
            elif ALNUM_RE.match(p):
                # Keep well-known alphanumeric identifiers intact
                out.append(p)
            else:
                m = re.match(r"(\d+)([a-zA-Z]+)", p)
                if m:
                    out.append("<NUM>")
                    out.append(m.group(2))
                else:
                    other_log_list.append(token)
                    out.append("<OTHER>")
        return out if out else ["<OTHER>"]

    def _classify_token(t: str) -> str:
        """Classify a single post-cleanup token."""
        if t in SPECIAL_TOKENS:
            return t
        if t.replace("-", "").isalpha():
            return t
        if ALNUM_RE.match(t):
            # Alphanumeric identifiers: co2, mi6, l2tp, cas9, h1, etc.
            return t
        if "-" in t:
            return None  # signal: needs hyphen expansion
        if NUM_PTN.fullmatch(t):
            return "<NUM>"
        # Last resort: try splitting leading digits from trailing alpha
        m = re.match(r"^(\d+)([a-zA-Z]+)$", t)
        if m:
            return None  # signal: needs splitting
        other_log_list.append(t)
        return "<OTHER>"

    def clean_text(doc: str) -> str:
        lines = doc.split("\n")
        out = []

        for line in lines:
            # 1. Expand contractions
            line = CONTRACTION_PTN.sub(_expand_contraction, line)

            # 2. Strip possessives
            line = POSSESSIVE_PTN.sub("", line)

            # 3. Split em/en dashes
            line = DASH_SPLIT_RE.sub(" ", line)

            # 4. Replace entities with padded special tokens
            line = _pad_special_token(URL_PTN, "<URL>", line)
            line = _pad_special_token(EMAIL_PTN, "<EMAIL>", line)
            line = _pad_special_token(MONEY_PTN, "<MONEY>", line)
            line = _pad_special_token(DATE_PTN, "<DATE>", line)
            line = ORDINAL_PTN.sub(" <NUM> ", line)

            # 5. Replace abbreviations
            line = ABBREVIATION_PTN.sub(_expand_abbreviation, line)

            # 6. Strip punctuation (now removes < > too) and lowercase
            tokens = line.split()
            clean_tokens = []
            for t in tokens:
                if t in SPECIAL_TOKENS:
                    clean_tokens.append(t)
                else:
                    cleaned = PUNCT_RE.sub("", t).lower().strip("-")
                    if cleaned:
                        clean_tokens.append(cleaned)
            tokens = clean_tokens

            # 7. Replace bare numbers
            joined = " ".join(tokens)
            joined = NUM_PTN.sub("<NUM>", joined)
            tokens = joined.split()

            # 8. Classify and expand each token
            final_tokens = []
            for t in tokens:
                label = _classify_token(t)
                if label is not None:
                    final_tokens.append(label)
                elif "-" in t:
                    final_tokens.extend(_expand_hyphenated(t))
                else:
                    # digit+alpha glued tokens like "tonnes17"
                    m = re.match(r"^(\d+)([a-zA-Z]+)$", t)
                    if m:
                        final_tokens.append("<NUM>")
                        final_tokens.append(m.group(2))
                    else:
                        m2 = re.match(r"^([a-zA-Z]+)(\d+)$", t)
                        if m2:
                            final_tokens.append(m2.group(1))
                            final_tokens.append("<NUM>")
                        else:
                            other_log_list.append(t)
                            final_tokens.append("<OTHER>")

            out.append(" ".join(final_tokens))

        result = "\n".join(out)
        result = MULTI_NEWLINE_RE.sub("\n", result)
        return result

    return SINGLE_CHAR_RE, SPECIAL_TOKENS, clean_text, other_log_list


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
def _(other_log_list: list[str], pl):
    from collections import Counter

    other_counter = Counter(other_log_list)
    other_log = (
        pl.DataFrame(
            {
                "original_token": list(other_counter.keys()),
                "count": list(other_counter.values()),
            }
        )
        .sort("count", descending=True)
    )
    other_log
    return


@app.cell
def _(SINGLE_CHAR_RE, SPECIAL_TOKENS, df2, natural_language_cols, pl):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    stemmed_stop_words = set(
        stemmer.stem(w) for w in stopwords.words("english")
    )


    def remove_stopwords_and_stem(text: str) -> str:
        tokens = []
        for w in text.split():
            if w in SPECIAL_TOKENS:
                tokens.append(w)
                continue
            s = stemmer.stem(w)
            if s not in stemmed_stop_words:
                tokens.append(s)
        return " ".join(tokens)


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
            .str.replace_all(SINGLE_CHAR_RE.pattern, "")
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
            x=alt.X(
                "word:N", sort=None, axis=alt.Axis(labelAngle=-35)
            ),
            y=alt.Y("count:Q"),
        )
        .properties(width=900, height=300)
    )
    chart
    return


if __name__ == "__main__":
    app.run()
