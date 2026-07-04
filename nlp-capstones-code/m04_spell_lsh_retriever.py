"""
capstone/modules/m04_spell_lsh_retriever.py
SpellLSHRetriever — imlo tuzatish (Noisy Channel + Levenshtein) va
MinHash LSH asosida hujjat qidirish.
Shartnoma: capstone/contracts.py :: SpellLSHRetriever
P4 (5-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) ustiga quriladi.
Consumed by: m15 (agent tool: spell_correct), Day 16 (pipeline).

LSH toza-python/numpy MinHash bilan amalga oshirilgan (datasketch SHART EMAS).
Kaggle da datasketch.MinHashLSH ham ishlatilishi mumkin, lekin bu modul unga
bog'liq emas -- har joyda ishlaydi.
"""
from __future__ import annotations

import pickle
import zlib
from collections import Counter

import numpy as np

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:   # paket sifatida import qilinganda
    from .m01_text_preprocessor import TextPreprocessor


def _stable_hash(s: str) -> int:
    """Takrorlanuvchan (deterministik) token xeshi -- Python hash() tuzsiz."""
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


class SpellLSHRetriever:
    """Imlo tuzatish (Noisy Channel) + LSH asosida hujjat qidirish.

    Consumed by: m15 (agent tool: spell_correct), Day 16 (pipeline).
    """

    def __init__(self, num_perm: int = 64, bands: int = 16,
                 alpha_channel: float = 0.1) -> None:
        self._pre = TextPreprocessor()
        self._num_perm = num_perm
        self._bands = bands
        self._rows = num_perm // bands
        self._alpha = alpha_channel
        rng = np.random.RandomState(42)
        self._a = rng.randint(1, 2**31 - 1, num_perm).astype(np.int64)
        self._b = rng.randint(0, 2**31 - 1, num_perm).astype(np.int64)
        self._p = np.int64(2**31 - 1)
        self._freq: Counter = Counter()
        self._total = 0
        self._docs: list[str] = []
        self._doc_shingles: list[set] = []
        self._buckets: dict = {}

    # ─── imlo: Levenshtein + Noisy Channel ──────────────────────────────────────
    def edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein tahrir masofasi (dinamik dasturlash, 2D jadval)."""
        m, n = len(s1), len(s2)
        D = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            D[i][0] = i
        for j in range(n + 1):
            D[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    D[i][j] = D[i - 1][j - 1]
                else:
                    D[i][j] = 1 + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
        return D[m][n]

    def fit_dictionary(self, texts: list[str]) -> None:
        """Lug'at chastotalarini (til modeli P(w)) korpusdan o'rganadi."""
        for t in texts:
            self._freq.update(self._pre.preprocess(t))
        self._total = sum(self._freq.values())

    def correct(self, word: str) -> str:
        """Noisy channel: argmax_w P(w)*P(x|w); P(x|w)=alpha^edit_distance."""
        w = word.lower()
        if not self._total:
            return w
        if w in self._freq:        # lug'atda bor -> tuzatish shart emas
            return w
        best, best_score = w, -1.0
        for cand, c in self._freq.items():
            d = self.edit_distance(w, cand)
            if d > 2:
                continue
            score = (c / self._total) * (self._alpha ** d)
            if score > best_score:
                best, best_score = cand, score
        return best

    # ─── LSH: MinHash + banding ─────────────────────────────────────────────────
    def _shingles(self, text: str) -> set:
        return set(self._pre.preprocess(text)) if text.strip() else set()

    def _signature(self, shingles: set) -> tuple:
        if not shingles:
            return tuple([int(self._p)] * self._num_perm)
        hs = np.array([_stable_hash(t) for t in shingles], dtype=np.int64)
        M = (self._a[:, None] * hs[None, :] + self._b[:, None]) % self._p
        return tuple(int(x) for x in M.min(axis=1))

    def index_docs(self, texts: list[str]) -> None:
        """Hujjatlarni MinHash LSH indeksiga qo'shadi."""
        for text in texts:
            i = len(self._docs)
            sh = self._shingles(text)
            self._docs.append(text)
            self._doc_shingles.append(sh)
            sig = self._signature(sh)
            for band in range(self._bands):
                key = (band, sig[band * self._rows:(band + 1) * self._rows])
                self._buckets.setdefault(key, set()).add(i)

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def lsh_candidates(self, query: str) -> set:
        """LSH savatlaridan nomzod hujjat indekslarini qaytaradi (tezlik uchun)."""
        sig = self._signature(self._shingles(query))
        cand = set()
        for band in range(self._bands):
            key = (band, sig[band * self._rows:(band + 1) * self._rows])
            cand |= self._buckets.get(key, set())
        return cand

    def retrieve_lsh(self, query: str, k: int = 5) -> list[str]:
        """LSH orqali eng o'xshash k ta hujjatni qaytaradi."""
        qsh = self._shingles(query)
        candidates = self.lsh_candidates(query)
        if not candidates:
            candidates = set(range(len(self._docs)))
        scored = [
            (self._jaccard(qsh, self._doc_shingles[i]), self._docs[i])
            for i in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:k]]

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        state = {
            "num_perm": self._num_perm, "bands": self._bands, "rows": self._rows,
            "alpha": self._alpha, "a": self._a, "b": self._b,
            "freq": self._freq, "total": self._total,
            "docs": self._docs, "doc_shingles": self._doc_shingles,
            "buckets": self._buckets,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._pre = TextPreprocessor()
        self._num_perm, self._bands, self._rows = s["num_perm"], s["bands"], s["rows"]
        self._alpha, self._a, self._b = s["alpha"], s["a"], s["b"]
        self._p = np.int64(2**31 - 1)
        self._freq, self._total = s["freq"], s["total"]
        self._docs, self._doc_shingles = s["docs"], s["doc_shingles"]
        self._buckets = s["buckets"]
