from __future__ import annotations

import polars as pl
from sklearn.metrics import accuracy_score, classification_report, f1_score


DEFAULT_SOURCE_NAMES = [
    "Normal Reuters",
    "Semi-political Reuters",
    "Very-political Reuters",
    "Non-political The New Yorker",
    "Satire The Onion",
    "Fake-news and political InfoWars",
]


def report(y_true, y_pred, target_names: list[str] | None = None) -> str:
    target_names = target_names or ["real", "fake"]
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    return f"Accuracy: {accuracy:.4f}\nF1: {f1:.4f}\n{report}"


def type_summary(types, guesses, scores) -> pl.DataFrame:
    return (
        pl.DataFrame({"type": types, "guess": guesses, "score": scores})
        .group_by("type")
        .agg(
            pl.len().alias("count"),
            pl.col("guess").mean().round(3).alias("mean_guess"),
            pl.col("score").mean().round(3).alias("mean_score"),
        )
        .sort("count", descending=True)
    )


def qualitative_test(
    filepath: str,
    model,
    vectorizer,
    preprocess_text,
    source_names: list[str] | None = None,
) -> list[dict[str, object]]:
    source_names = source_names or DEFAULT_SOURCE_NAMES

    with open(filepath, encoding="utf-8") as file:
        text = file.read()

    raw_articles = [chunk.strip() for chunk in text.split("---") if chunk.strip()]
    processed_texts = [preprocess_text(text) for text in raw_articles]
    x_custom = vectorizer.transform(processed_texts)
    preds = model.predict(x_custom)
    scores = model.decision_function(x_custom)

    rows = []
    for i, text in enumerate(raw_articles):
        rows.append(
            {
                "source": source_names[i]
                if i < len(source_names)
                else f"Article {i + 1}",
                "raw_text": text,
                "processed_text": processed_texts[i],
                "prediction": "FAKE" if preds[i] == 1 else "REAL",
                "score": float(scores[i]),
            }
        )
    return rows


def format_test_results(rows: list[dict[str, object]]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    str(row["source"]),
                    f"Raw Text:           {str(row['raw_text'])[:80].replace(chr(10), ' ')}...",
                    f"Processed Text:     {str(row['processed_text'])[:80]}...",
                    f"Prediction (SVM):   {row['prediction']}",
                    f"SVM Score:         {row['score']:.4f}",
                    "-" * 80,
                ]
            )
        )
    return "\n".join(blocks)
