"""
capstone/modules/m06_custom_word2vec.py
CustomWord2Vec — o'zbek korpusida noldan o'qitilgan CBOW so'z embeddinglari.
Shartnoma: capstone/contracts.py :: CustomWord2Vec
P6 (7-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) ustiga quriladi.
Consumed by: m08 (GRU/LSTM pretrained Embedding layer), m09 (generator).

gensim SHART EMAS (m03 namunasi):
  - Kaggle yo'li: gensim.models.Word2Vec(..., sg=0) bilan CBOW o'qitiladi, .kv saqlanadi.
  - Offline yo'l (gensimsiz): toza-numpy CBOW (negative sampling). Proyeksiya = kontekst
    vektorlari o'rtachasi (L6 [I2]), embed/most_similar kosinus bilan.
HAS_GENSIM bayrog'i faqat yo'lni tanlaydi -- modul har joyda ishlaydi.
"""
from __future__ import annotations

import pickle

import numpy as np

try:
    from gensim.models import Word2Vec, KeyedVectors   # faqat Kaggle/onlayn
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


class CustomWord2Vec:
    """Gensim bilan (yoki gensimsiz toza-numpy) o'qitilgan CBOW embeddinglar.

    Consumed by: m08 (GRU/LSTM pretrained Embedding layer), m09 (generator).
    """

    def __init__(self) -> None:
        self._gensim_model = None             # Kaggle: gensim Word2Vec
        self._words: list[str] = []           # offline: lug'at
        self._w2i: dict[str, int] = {}
        self._vec: np.ndarray | None = None   # offline: (V, dim) so'z vektorlari
        self._dim: int = 0

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def train(
        self,
        texts: list[list[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 10,
    ) -> None:
        """CBOW (sg=0) modelini o'qitadi. texts -- tokenlangan jumlalar ro'yxati."""
        self._dim = vector_size
        if HAS_GENSIM:
            # Kaggle yo'li: gensim CBOW (sg=0)
            self._gensim_model = Word2Vec(
                sentences=texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=0,            # 0 = CBOW
                epochs=epochs,
                workers=1,
                seed=42,
            )
        else:
            # Offline yo'li: toza-numpy CBOW (negative sampling)
            self._train_numpy(texts, vector_size, window, min_count, epochs)

    def _train_numpy(self, texts, dim, window, min_count, epochs) -> None:
        from collections import Counter

        rng = np.random.RandomState(42)
        # lug'at (min_count filtri)
        freq = Counter(w for s in texts for w in s)
        self._words = [w for w, c in freq.items() if c >= min_count]
        self._w2i = {w: i for i, w in enumerate(self._words)}
        V = len(self._words)
        if V == 0:
            self._vec = np.zeros((0, dim), dtype=np.float32)
            return

        # (kontekst, markaz) juftlari
        pairs = []
        for s in texts:
            idx = [self._w2i[w] for w in s if w in self._w2i]
            for i, center in enumerate(idx):
                lo, hi = max(0, i - window), min(len(idx), i + window + 1)
                ctx = [idx[j] for j in range(lo, hi) if j != i]
                if ctx:
                    pairs.append((ctx, center))

        # parametrlar
        W_in = (rng.randn(V, dim) * 0.01).astype(np.float32)    # so'z vektorlari
        W_out = (rng.randn(V, dim) * 0.01).astype(np.float32)   # chiqish vektorlari
        # negative sampling taqsimoti (unigram^0.75)
        p = np.array([freq[w] for w in self._words], dtype=np.float64) ** 0.75
        p /= p.sum()
        K, lr = 5, 0.05

        for _ in range(epochs):
            rng.shuffle(pairs)
            for ctx, center in pairs:
                h = W_in[ctx].mean(axis=0)                       # proyeksiya (L6 [I2])
                negs = rng.choice(V, size=K, p=p)
                targets = np.concatenate(([center], negs))
                labels = np.zeros(K + 1, dtype=np.float32); labels[0] = 1.0
                scores = W_out[targets] @ h                      # (K+1,)
                pred = 1.0 / (1.0 + np.exp(-scores))             # sigmoid
                grad = (pred - labels)                           # (K+1,)
                dh = grad[:, None] * W_out[targets]              # (K+1, dim)
                W_out[targets] -= lr * grad[:, None] * h[None, :]
                W_in[ctx] -= lr * dh.sum(axis=0)[None, :] / len(ctx)

        norms = np.linalg.norm(W_in, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vec = W_in.astype(np.float32)
        self._norm = (W_in / norms).astype(np.float32)

    # ─── asosiy metodlar ──────────────────────────────────────────────────────
    def embed(self, word: str) -> np.ndarray:
        """So'z vektori; OOV uchun nol-vektor (shape (vector_size,) float32)."""
        if HAS_GENSIM and self._gensim_model is not None:
            if word in self._gensim_model.wv:
                return np.asarray(self._gensim_model.wv[word], dtype=np.float32)
            return np.zeros(self._dim, dtype=np.float32)
        i = self._w2i.get(word)
        if i is None:
            return np.zeros(self._dim, dtype=np.float32)
        return self._vec[i].copy()

    def most_similar(self, word: str, n: int = 5) -> list[tuple[str, float]]:
        """Kosinus bo'yicha eng o'xshash n ta so'z: [(so'z, o'xshashlik), ...]."""
        if HAS_GENSIM and self._gensim_model is not None:
            if word not in self._gensim_model.wv:
                return []
            return [(w, float(s)) for w, s in self._gensim_model.wv.most_similar(word, topn=n)]
        i = self._w2i.get(word)
        if i is None:
            return []
        sims = self._norm @ self._norm[i]
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for j in order:
            if int(j) == i:
                continue
            out.append((self._words[int(j)], float(sims[int(j)])))
            if len(out) >= n:
                break
        return out

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Kaggle: .kv (gensim KeyedVectors); offline: pickle (lug'at + matritsa)."""
        if HAS_GENSIM and self._gensim_model is not None:
            self._gensim_model.wv.save(path)
        else:
            with open(path, "wb") as f:
                pickle.dump({"words": self._words, "vec": self._vec,
                             "norm": self._norm, "dim": self._dim}, f)

    def load(self, path: str) -> None:
        if HAS_GENSIM and path.endswith(".kv"):
            kv = KeyedVectors.load(path)
            self._words = list(kv.index_to_key)
            self._w2i = {w: i for i, w in enumerate(self._words)}
            self._vec = np.asarray(kv.vectors, dtype=np.float32)
            self._dim = self._vec.shape[1]
            norms = np.linalg.norm(self._vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._norm = (self._vec / norms).astype(np.float32)
            self._gensim_model = None
        else:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self._words, self._vec = s["words"], s["vec"]
            self._norm, self._dim = s["norm"], s["dim"]
            self._w2i = {w: i for i, w in enumerate(self._words)}
            self._gensim_model = None
