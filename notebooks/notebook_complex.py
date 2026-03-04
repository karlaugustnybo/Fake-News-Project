import marimo

__generated_with = "0.19.2"
app = marimo.App(sql_output="lazy-polars")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><h1 style='text-align: center;'>Fake News Complex</h1><br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dependencies

    | Library | Purpose |
    |---|---|
    | **polars** | High-performance DataFrame library (Rust-backed), used for all tabular data manipulation |
    | **altair** | Declarative statistical visualization library built on Vega-Lite |
    | **numpy** | Numerical array operations, used for index manipulation and sparse-matrix helpers |
    | **scikit-learn (`TfidfVectorizer`)** | Converts raw text into TF-IDF feature vectors for machine-learning |
    """)
    return


@app.cell
def _():
    # Core dependencies used across nearly every cell
    import polars as pl  # DataFrame library (fast, Rust-backed)
    import altair as alt  # Declarative charting
    import numpy as np  # Numerical operations
    from sklearn.feature_extraction.text import (
        TfidfVectorizer,
    )  # Text → TF-IDF features
    return TfidfVectorizer, alt, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Load Raw Data

    Read the FakeNewsCorpus CSV (≈ 995 k rows). All columns are forced to `Utf8`
    to avoid Polars type-inference errors on messy data. Columns that carry no
    predictive signal (IDs, URLs, timestamps, etc.) are dropped immediately,
    and only the first **10 000** rows are kept to speed up development.
    """)
    return


@app.cell
def _(pl):
    filepath = "news/data/995,000_rows.csv"

    # Force every column to string to prevent type-inference failures
    # on the heterogeneous CSV data
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

    # Drop columns with no predictive value (IDs, URLs, timestamps, etc.)
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
    # Keep only the first 10 000 rows for fast iteration
    df_raw = df_raw.drop(drop_cols).head(10_000)
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    # Quick sanity-check: print schema, dimensions, and per-column null counts
    print("Schema:")
    for _name, _dtype in df_raw.schema.items():
        print(f"  {_name}: {_dtype}")
    print(f"\nShape: {df_raw.shape}")
    print(f"Null counts:\n{df_raw.null_count()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Label Consolidation

    The raw `type` column contains many fine-grained categories
    ("reliable", "political", "fake", "conspiracy", …). We collapse them into
    a binary label:

    * **real** ← `reliable`, `political`
    * **fake** ← everything else that is not in the ambiguous/noisy set

    Ambiguous categories (`unknown`, `satire`, `bias`, `clickbait`,
    `unreliable`, `state`) are **dropped entirely** because they would add
    label noise.
    """)
    return


@app.cell
def _(df_raw, pl):
    # Categories we trust as "real" news
    REAL_LABELS = {"reliable", "political"}
    # Ambiguous / noisy categories — dropped to reduce label noise
    DROP_LABELS = {"unknown", "satire", "bias", "clickbait", "unreliable", "state"}

    # Filter out nulls and ambiguous types, then map to binary label
    df_labeled = df_raw.filter(
        pl.col("type").is_not_null() & ~pl.col("type").is_in(list(DROP_LABELS))
    ).with_columns(
        pl.when(pl.col("type").is_in(list(REAL_LABELS)))
        .then(pl.lit("real"))
        .otherwise(pl.lit("fake"))
        .alias("label")
    )

    print("Label distribution after binary consolidation:")
    print(
        df_labeled.group_by("label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print("\nOriginal type → label mapping:")
    print(
        df_labeled.group_by("type", "label")
        .agg(pl.len().alias("count"))
        .sort("type")
    )
    return (df_labeled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Deduplication (exact + near-duplicate)

    Duplicate articles inflate metrics and leak information between splits.

    1. **Exact dedup** — Polars `.unique()` on the `content` column.
    2. **Near-duplicate dedup** — Uses **MinHash LSH** (`datasketch` library)
       with 128 permutations and a Jaccard-similarity threshold of 0.8.
       Articles whose shingle sets are ≥ 80 % similar are treated as
       near-duplicates and removed.

    > `datasketch` provides probabilistic data structures for fast
    > approximate nearest-neighbour / similarity queries.
    """)
    return


@app.cell
def _(df_labeled, pl):
    # datasketch — probabilistic data structures for near-duplicate detection
    from datasketch import MinHash, MinHashLSH

    _n_before = df_labeled.shape[0]

    # Step 1: exact dedup on content text
    _df_exact = df_labeled.unique(subset=["content"])
    _n_exact = _df_exact.shape[0]

    # Step 2: near-duplicate removal via MinHash LSH
    _NUM_PERM = 128  # number of hash permutations (more = more accurate, slower)
    _THRESHOLD = 0.8  # Jaccard similarity threshold for "near-duplicate"
    _lsh = MinHashLSH(threshold=_THRESHOLD, num_perm=_NUM_PERM)
    _keep_indices: list[int] = []
    _texts = _df_exact["content"].to_list()

    for _i, _text in enumerate(_texts):
        if _text is None:
            continue
        # Build a MinHash signature from the word-level shingles
        _mh = MinHash(num_perm=_NUM_PERM)
        for _word in _text.split():
            _mh.update(_word.encode("utf-8"))
        # If no similar document is already in the index, keep this one
        if not _lsh.query(_mh):
            _lsh.insert(str(_i), _mh)
            _keep_indices.append(_i)

    df_deduped = _df_exact[_keep_indices]
    _n_near = _n_exact - df_deduped.shape[0]

    # Clean up placeholder meta_keywords values ("['']" → null)
    df_deduped = df_deduped.with_columns(
        pl.when(pl.col("meta_keywords") == "['']")
        .then(pl.lit(None))
        .otherwise(pl.col("meta_keywords"))
        .alias("meta_keywords")
    )

    print(f"Exact duplicates removed:  {_n_before - _n_exact}")
    print(f"Near-duplicates removed:   {_n_near}")
    print(f"Rows remaining:            {df_deduped.shape[0]}")
    return (df_deduped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Language Filtering

    The dataset may contain non-English articles. We use **`langdetect`**
    (a port of Google's language-detection library) to identify each article's
    language from its first 200 characters and keep only English (`en`) texts.
    """)
    return


@app.cell
def _(df_deduped, pl):
    # langdetect — fast language identification
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException


    def _detect_lang(text: str) -> str:
        """Detect language from first 200 chars for speed."""
        try:
            return detect(text[:200])
        except LangDetectException, Exception:
            return "unknown"


    _n_before = df_deduped.shape[0]

    # Tag each row with its detected language, keep only English, then drop helper col
    df_lang = (
        df_deduped.with_columns(
            pl.col("content")
            .map_elements(_detect_lang, return_dtype=pl.Utf8)
            .alias("_lang")
        )
        .filter(pl.col("_lang") == "en")
        .drop("_lang")
    )

    _n_removed = _n_before - df_lang.shape[0]
    print(f"Non-English articles removed: {_n_removed}")
    print(f"Rows remaining:               {df_lang.shape[0]}")
    return (df_lang,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Feature Engineering & Text Cleaning

    This cell does two things **before** lowercasing the text:

    ### a) Stylometric features (computed on *raw* text)
    These capture writing-style signals that differ between real and fake news:

    | Feature | What it measures |
    |---|---|
    | `caps_ratio` | Fraction of characters that are uppercase |
    | `exclamation_density` | Exclamation marks per word |
    | `question_density` | Question marks per word |
    | `avg_sentence_len` | Words per sentence (approximated) |

    ### b) Text normalisation pipeline
    Applied to `content`, `title`, and `meta_description`:

    1. Strip HTML tags
    2. Remove URLs and e-mail addresses
    3. Remove "BREAKING:" prefixes
    4. Normalise smart quotes → ASCII quotes
    5. Collapse whitespace & non-breaking spaces
    6. Lowercase + strip
    7. Expand contractions ("don't" → "do not") using the `contractions` library
    """)
    return


@app.cell
def _(df_lang, pl):
    # contractions — expands English contractions (e.g. "don't" → "do not")
    import contractions

    text_cols = ["content", "title", "meta_description"]

    # --- a) Compute stylometric features BEFORE lowercasing ---
    df_features = df_lang.with_columns(
        # Ratio of uppercase characters to total characters
        (
            pl.col("content").str.count_matches(r"[A-Z]").cast(pl.Float64)
            / (pl.col("content").str.len_chars().cast(pl.Float64) + 1.0)
        ).alias("caps_ratio"),
        # Exclamation marks per word
        (
            pl.col("content").str.count_matches(r"!").cast(pl.Float64)
            / (pl.col("content").str.count_matches(r"\S+").cast(pl.Float64) + 1.0)
        ).alias("exclamation_density"),
        # Question marks per word
        (
            pl.col("content").str.count_matches(r"\?").cast(pl.Float64)
            / (pl.col("content").str.count_matches(r"\S+").cast(pl.Float64) + 1.0)
        ).alias("question_density"),
        # Average sentence length (words per sentence)
        (
            pl.col("content").str.count_matches(r"\S+").cast(pl.Float64)
            / (
                pl.col("content").str.count_matches(r"[.!?]+").cast(pl.Float64)
                + 1.0
            )
        ).alias("avg_sentence_len"),
    )

    # --- b) Text normalisation pipeline ---
    df_cleaned = df_features.with_columns(
        *[
            pl.col(col)
            .str.replace_all(r"<[^>]+>", " ")  # strip HTML tags
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"\S+@\S+\.\S+", "")  # remove emails
            .str.replace_all(r"(?i)^breaking:\s*", "")  # strip "BREAKING:" prefix
            .str.replace_all(
                r"[\u2018\u2019\u201C\u201D]",
                "'",  # smart quotes → ASCII
            )
            .str.replace_all(r"\u00a0", " ")  # non-breaking space → space
            .str.replace_all(r"\s+", " ")  # collapse whitespace
            .str.to_lowercase()  # lowercase
            .str.strip_chars()  # trim leading/trailing whitespace
            .alias(col)
            for col in text_cols
        ],
    )

    # Expand contractions (e.g. "don't" → "do not")
    df_cleaned = df_cleaned.with_columns(
        pl.col("content")
        .map_elements(contractions.fix, return_dtype=pl.Utf8)
        .alias("content")
    )

    # Drop rows where content is null or empty after cleaning
    df_cleaned = df_cleaned.filter(
        pl.col("content").is_not_null() & (pl.col("content").str.len_chars() > 0)
    )

    print(f"Rows after text cleaning: {df_cleaned.shape[0]}")
    df_cleaned.select(text_cols).head(3)
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Length-based Outlier Removal

    Very short articles (< 50 words) are likely stubs or scraping artefacts.
    Very long articles (> 50 000 chars) are likely concatenated pages or OCR dumps.
    Both are removed to keep the data distribution clean.
    """)
    return


@app.cell
def _(df_cleaned, pl):
    MIN_WORDS = 50  # Minimum word count to keep
    MAX_CHARS = 50_000  # Maximum character count to keep

    _n_before = df_cleaned.shape[0]
    df_filtered = (
        df_cleaned.with_columns(
            pl.col("content").str.count_matches(r"\S+").alias("_wc")
        )
        .filter(
            (pl.col("_wc") >= MIN_WORDS)
            & (pl.col("content").str.len_chars() <= MAX_CHARS)
        )
        .drop("_wc")
    )

    _n_removed = _n_before - df_filtered.shape[0]
    print(
        f"Outlier documents removed: {_n_removed}"
        f"  (< {MIN_WORDS} words or > {MAX_CHARS:,} chars)"
    )
    print(f"Rows remaining:            {df_filtered.shape[0]}")
    return (df_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Cleanlab Label-Quality Audit

    **Goal:** automatically find and remove mislabelled or anomalous samples.

    | Dependency | Role |
    |---|---|
    | `cleanlab.Datalab` | Data-centric AI toolkit that scores every sample for label quality and outlierness |
    | `LogisticRegression` | Simple classifier used to produce the predicted probabilities Cleanlab needs |
    | `StratifiedKFold` + `cross_val_predict` | Generate out-of-fold probability estimates so no sample is scored on data it was trained on |

    **Steps:**
    1. Fit a quick TF-IDF (10 k features) + Logistic Regression via 5-fold stratified CV.
    2. Feed the out-of-fold predicted probabilities to Cleanlab's `Datalab`.
    3. Remove rows whose **outlier score < 0.01** (extreme outliers) or whose
       **label score < 0.15** (likely mislabelled).
    """)
    return


@app.cell
def _(df_filtered, np, pl):
    # scikit-learn utilities for the audit classifier
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # cleanlab — data-centric AI toolkit for finding label issues
    from cleanlab import Datalab

    # Build a lightweight TF-IDF representation for the audit classifier
    _vec = _TV(max_features=10_000, min_df=2, max_df=0.95, sublinear_tf=True)
    X_audit = _vec.fit_transform(df_filtered["content"].to_list())
    y_audit = df_filtered["label"].to_numpy()

    # Convert string labels to integer indices (cleanlab requires ints)
    _label_map = {lbl: i for i, lbl in enumerate(sorted(set(y_audit)))}
    y_int = np.array([_label_map[l] for l in y_audit])

    # Determine safe number of CV folds (must be ≤ minority-class count)
    min_class = min(
        df_filtered.group_by("label").agg(pl.len().alias("n"))["n"].to_list()
    )
    n_splits = min(5, min_class)

    if n_splits < 2:
        print(
            f" Skipping cleanlab audit: minority class has only {min_class} "
            f"sample(s), which is too few for cross-validation."
        )

    # Stratified K-Fold ensures each fold has the same label ratio
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Out-of-fold predicted probabilities — each sample is scored by a model
    # that never saw it during training
    pred_probs = cross_val_predict(
        LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
        X_audit,
        y_int,
        cv=skf,
        method="predict_proba",
    )

    # Run Cleanlab's automated data-quality checks
    lab = Datalab(data={"label": y_int.tolist()}, label_name="label")
    lab.find_issues(pred_probs=pred_probs)
    lab.report()

    # Remove samples flagged as outliers or mislabelled
    _issues = lab.get_issues()
    _n_before = df_filtered.shape[0]

    _outlier_mask = (
        _issues["outlier_score"].values >= 0.01
    )  # keep if NOT extreme outlier
    _label_mask = (
        _issues["label_score"].values >= 0.15
    )  # keep if label looks correct
    _keep_mask = _outlier_mask & _label_mask

    _keep_indices = np.where(_keep_mask)[0].tolist()
    df_audited = df_filtered[_keep_indices]

    _n_outlier_rm = int((~_outlier_mask).sum())
    _n_label_rm = int((~_label_mask & _outlier_mask).sum())
    print(f"\n── Cleanlab-based removal ──")
    print(f"Severe outliers removed (score < 0.01): {_n_outlier_rm}")
    print(f"Mislabeled samples removed (score < 0.15): {_n_label_rm}")
    print(f"Rows remaining: {df_audited.shape[0]}  (was {_n_before})")
    return (df_audited,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Train / Validation / Test Split

    We split the audited data into three stratified subsets:

    * **Train (80 %)** — used to fit the final model
    * **Validation (10 %)** — used for hyperparameter tuning / early stopping
    * **Test (10 %)** — held out for final evaluation only

    `stratify=labels` ensures each split preserves the real/fake ratio.
    """)
    return


@app.cell
def _(df_audited, np):
    from sklearn.model_selection import train_test_split

    n = df_audited.shape[0]
    indices = np.arange(n)
    labels = df_audited["label"].to_numpy()

    # First split: 80 % train, 20 % temp (will become val + test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    # Second split: halve the temp set into 10 % val + 10 % test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx]
    )

    df_train = df_audited[train_idx.tolist()]
    df_val = df_audited[val_idx.tolist()]
    df_test = df_audited[test_idx.tolist()]

    print(f"Train: {df_train.shape[0]}  ({df_train.shape[0] / n:.0%})")
    print(f"Val:   {df_val.shape[0]}  ({df_val.shape[0] / n:.0%})")
    print(f"Test:  {df_test.shape[0]}  ({df_test.shape[0] / n:.0%})")
    return df_test, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · TF-IDF Vectorization + Stylometric Features

    Build the final feature matrices that a classifier will consume:

    1. **TF-IDF** — up to 80 000 uni- and bigram features, with sublinear TF
       scaling (`1 + log(tf)`) to dampen the effect of very frequent terms.
       `min_df=3` / `max_df=0.95` prune extremely rare or ubiquitous terms.
    2. **Stylometric columns** — the four writing-style features computed
       in §5 are converted to a sparse matrix and horizontally stacked
       onto the TF-IDF matrix via `scipy.sparse.hstack`.

    > **Important:** the vectorizer is **fit on the training set only** and
    > then applied (`transform`) to val/test to prevent data leakage.
    """)
    return


@app.cell
def _(TfidfVectorizer, df_test, df_train, df_val, np):
    # scipy.sparse — efficient storage for high-dimensional sparse feature matrices
    from scipy.sparse import csr_matrix, hstack

    # Configure the TF-IDF vectorizer with uni- and bigrams
    vectorizer = TfidfVectorizer(
        max_features=80_000,  # vocabulary cap
        ngram_range=(1, 2),  # unigrams + bigrams
        sublinear_tf=True,  # apply 1 + log(tf)
        min_df=3,  # ignore terms in < 3 documents
        max_df=0.95,  # ignore terms in > 95 % of documents
        strip_accents="unicode",  # normalise accented characters
        token_pattern=r"[a-zA-Z]{2,}",  # only alphabetic tokens ≥ 2 chars
    )

    # Fit on train, transform val/test (no leakage)
    tfidf_train = vectorizer.fit_transform(df_train["content"].to_list())
    tfidf_val = vectorizer.transform(df_val["content"].to_list())
    tfidf_test = vectorizer.transform(df_test["content"].to_list())

    # Stylometric feature columns to append alongside TF-IDF
    STYLE_COLS = [
        "caps_ratio",
        "exclamation_density",
        "question_density",
        "avg_sentence_len",
    ]


    def _style_matrix(df):
        """Convert stylometric columns to a sparse matrix (compatible with hstack)."""
        arr = df.select(STYLE_COLS).to_numpy().astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return csr_matrix(arr)


    # Horizontally stack TF-IDF + stylometric features
    X_train = hstack([tfidf_train, _style_matrix(df_train)])
    X_val = hstack([tfidf_val, _style_matrix(df_val)])
    X_test = hstack([tfidf_test, _style_matrix(df_test)])

    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"TF-IDF train: {tfidf_train.shape}")
    print(f"TF-IDF val:   {tfidf_val.shape}")
    print(f"TF-IDF test:  {tfidf_test.shape}")
    print(f"\nFinal feature matrix shapes:")
    print(
        f"  X_train: {X_train.shape}  (TF-IDF: {tfidf_train.shape[1]} + style: {len(STYLE_COLS)})"
    )
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    return tfidf_train, vectorizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Split Distribution Check

    Prints the label distribution (count + percentage) in each split to
    verify that stratification kept the real/fake ratio consistent.
    """)
    return


@app.cell
def _(df_test, df_train, df_val, pl):
    # Verify that stratified splitting preserved the label ratio
    for _name, _split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        _dist = (
            _split.group_by("label")
            .agg(pl.len().alias("count"))
            .sort("label")
            .with_columns(
                (pl.col("count").cast(pl.Float64) / pl.col("count").sum() * 100)
                .round(1)
                .alias("pct")
            )
        )
        print(f"\n{_name} split:")
        print(_dist)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Top-50 TF-IDF Terms (Visualization)

    Bar chart of the 50 terms with the highest aggregate TF-IDF score across
    the training set. Useful for spotting dominant words and potential
    topic-leakage terms (e.g. source names that correlate with labels).
    """)
    return


@app.cell
def _(alt, np, pl, tfidf_train, vectorizer):
    # Sum each term's TF-IDF value across all training documents
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

    # Altair bar chart — declarative visualization
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
