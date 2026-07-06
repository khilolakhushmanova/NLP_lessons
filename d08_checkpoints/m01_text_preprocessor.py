"""
capstone/modules/m01_text_preprocessor.py
TextPreprocessor — O'zbek matni uchun preprocessing pipeline.
Shartnoma: capstone/contracts.py :: TextPreprocessor
P1 (2-kun amaliyoti) da qurilgan; m02-m09, m11, m15 tomonidan ishlatiladi.
"""
from __future__ import annotations

import re
from collections import Counter

_APOSTROPHE_RE = re.compile(r"['’‘ʼʻ`´]")
_TOKEN_RE = re.compile(r"[a-z][a-z']*")

_DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    "va", "bu", "bir", "bilan", "da", "ni", "ga", "dan",
    "ham", "uchun", "bo'lgan", "bo'lib", "bo'ldi", "o'z",
    "ular", "u", "men", "biz", "siz", "edi", "ekan", "deb",
    "lekin", "ammo", "yoki", "agar", "chunki", "hali",
    "ko'p", "oz", "shunday", "shu", "esa", "endi",
    "bor", "yo'q", "kerak", "mumkin", "bo'lsa", "bo'lishi",
})

_UZ_SUFFIXES: tuple[str, ...] = (
    "larning", "lardan", "lardan", "larimiz", "laringiz", "laridir",
    "larni", "larga", "larim", "laring", "lari",
    "imizdan", "ingizdan",
    "ning", "niki", "nchi",
    "lilik", "ligi",
    "roqqa", "roq",
    "ishda", "ishi", "ish",
    "chi", "lik",
    "lar", "ni", "da", "dan", "ga",
)


class TextPreprocessor:
    """O'zbek matni uchun tokenizatsiya + normalizatsiya + stemming pipeline.

    Consumed by: m02, m04, m05, m06, m07, m08, m09, m11, m15.
    """

    def __init__(self) -> None:
        self._stopwords: set[str] = set(_DEFAULT_STOPWORDS)

    # ─── ichki metodlar ───────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = _APOSTROPHE_RE.sub("'", text)
        return text.lower()

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(self._normalize(text))

    def _stem(self, token: str) -> str:
        for suffix in sorted(_UZ_SUFFIXES, key=len, reverse=True):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    # ─── ommaviy metodlar ─────────────────────────────────────────────────────

    def preprocess(self, text: str) -> list[str]:
        """Bitta matnni tokenize qilib, tozalab, stemlab qaytaradi.

        Args:
            text: Xom o'zbek matni (UTF-8, U+2019 yoki ASCII apostrof).

        Returns:
            Tozalangan tokenlar ro'yxati. Stopwordlar va tinywords (len<2)
            tashlanadi. Har token kichik harfda, stemmed.

        Raises:
            ValueError: Agar text bo'sh string bo'lsa.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "preprocess(): text bo'sh bo'lmasligi kerak. "
                "Bo'sh string qabul qilinmaydi."
            )
        result = []
        for t in self._tokenize(text):
            if len(t) < 2:
                continue
            if t in self._stopwords:
                continue
            result.append(self._stem(t))
        return result

    def preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        """preprocess() ni bir ro'yxat uchun qo'llaydi."""
        return [self.preprocess(t) for t in texts]

    def fit_stopwords(self, texts: list[str], max_df: float = 0.85) -> None:
        """Korpus-spesifik stopwordlarni chastota bo'yicha aniqlaydi.

        max_df ulushidan ortiq hujjatlarda uchraydigan so'zlar
        stopwords ga qo'shiladi.

        Args:
            texts:  Xom matnlar ro'yxati.
            max_df: [0,1] — hujjatlar ulushidan ortiq uchraydigan so'zlar
                    stopword sifatida belgilanadi. Default 0.85.
        """
        n_docs = len(texts)
        if n_docs == 0:
            return
        threshold = max(1, int(n_docs * max_df))
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(self._tokenize(text)))
        for word, count in df.items():
            if count >= threshold:
                self._stopwords.add(word)
