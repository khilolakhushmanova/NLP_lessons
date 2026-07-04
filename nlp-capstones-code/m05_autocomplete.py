"""
capstone/modules/m05_autocomplete.py
Autocomplete — N-gram til modeli asosida so'z/ibora to'ldirish.
Shartnoma: capstone/contracts.py :: Autocomplete
P5 (6-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) normalizatsiyasidan foydalanadi.
Consumed by: Day 16 (pipeline demo).

nltk SHART EMAS — n-gram toza-python bilan sanaladi. Til modeli uchun stop-so'zlar
SAQLANADI (autocomplete funksional so'zlarni ham bashorat qilishi kerak), shuning uchun
m01.preprocess (stopword/stemming) emas, balki m01 normalizatsiyasi (apostrof + kichik harf)
ishlatiladi.
"""
from __future__ import annotations

import math
import re
import pickle
from collections import Counter

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor

_TOK = re.compile(r"[a-z][a-z']*")


class Autocomplete:
    """N-gram til modeli asosida so'z/ibora to'ldirish.

    Consumed by: Day 16 (pipeline demo).
    """

    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()      # m01 — normalizatsiya uchun
        self.n = 2
        self.context_counts: Counter = Counter()    # (n-1)-gram kontekst soni
        self.ngram_counts: Counter = Counter()      # (kontekst, so'z) soni
        self.vocabulary: set = set()

    def _tokenize(self, text: str) -> list[str]:
        """m01 normalizatsiyasi (apostrof + kichik harf); barcha so'zlar saqlanadi."""
        return _TOK.findall(self.preprocessor._normalize(text))

    def train(self, texts: list[list[str]], n: int = 2) -> None:
        """N-gram modelini sanab o'qitadi. texts — tokenlangan jumlalar ro'yxati."""
        self.n = n
        self.context_counts.clear()
        self.ngram_counts.clear()
        self.vocabulary.clear()
        for tokens in texts:
            self.vocabulary.update(tokens)
            for index in range(n - 1, len(tokens)):
                context = tuple(tokens[index - n + 1:index])
                next_word = tokens[index]
                self.context_counts[context] += 1
                self.ngram_counts[(context, next_word)] += 1

    def _p_laplace(self, context: tuple, word: str) -> float:
        vocabulary_size = len(self.vocabulary)
        if not vocabulary_size:
            return 0.0
        numerator = self.ngram_counts[(context, word)] + 1
        denominator = self.context_counts[context] + vocabulary_size
        return numerator / denominator

    def complete(self, prefix: str, k: int = 3) -> list[str]:
        """Prefiksdan keyingi eng ehtimoliy k ta so'zni qaytaradi."""
        tokens = self._tokenize(prefix)
        context = tuple(tokens[-(self.n - 1):]) if self.n > 1 else ()
        candidates = sorted({word for (saved_context, word) in self.ngram_counts
                             if saved_context == context})
        if not candidates:
            candidates = sorted(self.vocabulary)
        candidates.sort(key=lambda word: (-self._p_laplace(context, word), word))
        return candidates[:k]

    def perplexity(self, text: str) -> float:
        """Matn uchun perplexity (add-1 smoothing bilan)."""
        tokens = self._tokenize(text)
        if len(tokens) < self.n:
            return float("inf")
        log_sum = 0.0
        bigram_count = 0
        for index in range(self.n - 1, len(tokens)):
            context = tuple(tokens[index - self.n + 1:index])
            word = tokens[index]
            log_sum += math.log(self._p_laplace(context, word))
            bigram_count += 1
        return math.exp(-log_sum / bigram_count) if bigram_count else float("inf")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "n": self.n,
                "ctx": self.context_counts,
                "ngram": self.ngram_counts,
                "vocab": self.vocabulary,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self.preprocessor = TextPreprocessor()
        self.n = s["n"]
        self.context_counts = s["ctx"]
        self.ngram_counts = s["ngram"]
        self.vocabulary = s["vocab"]
