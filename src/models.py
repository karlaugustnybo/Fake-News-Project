from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_logreg(
    X_train,
    y_train,
    random_state: int = 1000,
    max_iter: int = 2500,
) -> LogisticRegression:
    classifier = LogisticRegression(random_state=random_state, max_iter=max_iter)
    classifier.fit(X_train, y_train)
    return classifier


def train_svm(
    X_train,
    y_train,
    c: float = 0.3727593720314942,
    random_state: int = 1,
    max_iter: int = 10_000,
    dual: str = "auto",
    class_weight: str | None = "balanced",
) -> LinearSVC:
    model = LinearSVC(
        C=c,
        random_state=random_state,
        max_iter=max_iter,
        dual=dual,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath: str) -> None:
    joblib.dump(model, filepath)


def load_model(filepath: str):
    return joblib.load(filepath)
