"""
capstone/modules/m02_sentiment_classifier.py
SentimentClassifier — TF-IDF + LogisticRegression / MultinomialNB asosida
ikkilik sentiment tahlili (ijobiy / salbiy).
Shartnoma: capstone/contracts.py :: SentimentClassifier
P2 (3-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) ustiga quriladi.
Consumed by: M4 (FastAPI app.py), Day 16 (agent tool).
"""
from __future__ import annotations

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:  # paket sifatida import qilinganda
    from .m01_text_preprocessor import TextPreprocessor


class SentimentClassifier:
    """TF-IDF + LogReg yoki NaiveBayes asosida ikkilik sentiment tahlili.

    Consumed by: M4 (FastAPI), Day 16 (agent tool).
    """

    def __init__(self, model: str = "logreg") -> None:
        if model not in ("logreg", "nb"):
            raise ValueError("model 'logreg' yoki 'nb' bo'lishi kerak.")
        self._model_name = model
        self._pre = TextPreprocessor()
        self._vec = TfidfVectorizer()
        self._clf = (
            LogisticRegression(max_iter=1000)
            if model == "logreg"
            else MultinomialNB(alpha=1.0)
        )
        self._fitted = False

    def _prep(self, text: str) -> str:
        """m01 bilan tozalab, TF-IDF uchun bo'sh-joy bilan birlashtiradi."""
        if not isinstance(text, str) or not text.strip():
            return ""
        return " ".join(self._pre.preprocess(text))

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Modelni o'qitadi. labels: 'ijobiy' yoki 'salbiy'."""
        if len(texts) != len(labels):
            raise ValueError("texts va labels uzunligi teng bo'lishi kerak.")
        X = self._vec.fit_transform([self._prep(t) for t in texts])
        self._clf.fit(X, labels)
        self._fitted = True

    def predict(self, text: str) -> str:
        """Bitta matn uchun 'ijobiy' yoki 'salbiy' qaytaradi."""
        if not self._fitted:
            raise ValueError("Avval fit() ni chaqiring.")
        X = self._vec.transform([self._prep(text)])
        return str(self._clf.predict(X)[0])

    def predict_proba(self, text: str) -> dict[str, float]:
        """Ehtimolliklar: {'ijobiy': 0.82, 'salbiy': 0.18}."""
        if not self._fitted:
            raise ValueError("Avval fit() ni chaqiring.")
        X = self._vec.transform([self._prep(text)])
        probs = self._clf.predict_proba(X)[0]
        return {str(c): float(p) for c, p in zip(self._clf.classes_, probs)}

    def save(self, path: str) -> None:
        """Vektorlashtiruvchi va modelni pickle orqali saqlaydi."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vec": self._vec,
                    "clf": self._clf,
                    "model_name": self._model_name,
                    "fitted": self._fitted,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._vec = state["vec"]
        self._clf = state["clf"]
        self._model_name = state["model_name"]
        self._fitted = state["fitted"]
