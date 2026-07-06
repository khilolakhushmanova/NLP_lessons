"""
capstone/modules/m07_rnn_classifier.py
RNNClassifier — PyTorch nn.RNN asosida matn (sentiment) tasniflash.
Shartnoma: capstone/contracts.py :: RNNClassifier
P7 (8-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) ustiga quriladi.
Consumed by: m08 (taqqoslash uchun baseline), Day 16 (pipeline demo).

torch SHART EMAS (m03/m06 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: nn.Embedding + nn.RNN + nn.Linear, CrossEntropyLoss + Adam.
  - Offline yo'l (torch'siz): toza-numpy RNN klassifikator (forward + BPTT/SGD).
HAS_TORCH bayrog'i yo'lni tanlaydi -- modul har joyda ishlaydi.
Yorliqlar: 'ijobiy' / 'salbiy' (qulflangan).
"""
from __future__ import annotations

import pickle

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle yo'li: PyTorch SimpleRNN klassifikatori
# ─────────────────────────────────────────────────────────────────────────────
if HAS_TORCH:
    class _TorchRNN(nn.Module):
        def __init__(self, vocab_size, dim, hidden, n_cls):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.rnn = nn.RNN(dim, hidden, batch_first=True, nonlinearity="tanh")
            self.out = nn.Linear(hidden, n_cls)

        def forward(self, x):
            e = self.emb(x)            # (B, T, dim)
            _, h = self.rnn(e)         # h: (1, B, hidden) -- oxirgi holat h_T
            return self.out(h[-1])     # (B, n_cls) = W_o h_T + b_o


class RNNClassifier:
    """nn.RNN (yoki toza-numpy) asosida ikkilik sentiment tasnifi.

    Consumed by: m08 (baseline comparison), Day 16 (pipeline demo).
    """

    def __init__(self, embed_dim: int = 32) -> None:
        self._pre = TextPreprocessor()
        self._dim = embed_dim
        self._w2i: dict[str, int] = {}     # 0 = PAD
        self._labels: list[str] = []
        self._model = None                 # torch model
        self._np = None                    # numpy parametrlar (offline)
        self._hidden = 64

    # ─── kodlash ──────────────────────────────────────────────────────────────
    def _encode(self, text: str) -> list[int]:
        toks = self._pre.preprocess(text) if text.strip() else []
        ids = [self._w2i[t] for t in toks if t in self._w2i]
        return ids if ids else [0]

    def _build_vocab(self, token_lists) -> None:
        vocab = sorted({t for toks in token_lists for t in toks})
        self._w2i = {w: i + 1 for i, w in enumerate(vocab)}   # 0 PAD uchun

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def fit(self, texts: list[str], labels: list[str],
            epochs: int = 5, hidden_size: int = 64, lr: float = 1e-3) -> None:
        """RNN klassifikatorini o'qitadi. labels: 'ijobiy'/'salbiy'."""
        self._hidden = hidden_size
        token_lists = [self._pre.preprocess(t) if t.strip() else [] for t in texts]
        self._build_vocab(token_lists)
        self._labels = sorted(set(labels))                    # ['ijobiy','salbiy']
        l2i = {lab: i for i, lab in enumerate(self._labels)}
        seqs = [[self._w2i[t] for t in toks if t in self._w2i] or [0] for toks in token_lists]
        ys = [l2i[lab] for lab in labels]

        if HAS_TORCH:
            self._fit_torch(seqs, ys, epochs, hidden_size, lr)
        else:
            self._fit_numpy(seqs, ys, epochs, hidden_size, lr)

    def _pad(self, seqs):
        T = max(len(s) for s in seqs)
        return [s + [0] * (T - len(s)) for s in seqs]

    def _fit_torch(self, seqs, ys, epochs, hidden, lr) -> None:
        torch.manual_seed(42)
        X = torch.tensor(self._pad(seqs), dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        self._model = _TorchRNN(len(self._w2i) + 1, self._dim, hidden, len(self._labels))
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self._model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        self._model.eval()

    def _fit_numpy(self, seqs, ys, epochs, hidden, lr) -> None:
        rng = np.random.RandomState(42)
        V, d, H, C = len(self._w2i) + 1, self._dim, hidden, len(self._labels)
        E = rng.randn(V, d) * 0.1
        Wx = rng.randn(H, d) * 0.1
        Wh = rng.randn(H, H) * 0.1
        bh = np.zeros(H)
        Wo = rng.randn(C, H) * 0.1
        bo = np.zeros(C)
        for _ in range(epochs):
            for s, y in zip(seqs, ys):
                hs = [np.zeros(H)]
                for tid in s:                                  # forward
                    a = Wx @ E[tid] + Wh @ hs[-1] + bh
                    hs.append(np.tanh(a))
                z = Wo @ hs[-1] + bo
                p = np.exp(z - z.max()); p /= p.sum()
                dz = p.copy(); dz[y] -= 1.0                     # softmax+CE grad
                Wo -= lr * np.outer(dz, hs[-1]); bo -= lr * dz
                dh = Wo.T @ dz
                for t in range(len(s), 0, -1):                 # BPTT
                    da = dh * (1 - hs[t] ** 2)
                    Wh -= lr * np.outer(da, hs[t - 1])
                    Wx -= lr * np.outer(da, E[s[t - 1]])
                    bh -= lr * da
                    E[s[t - 1]] -= lr * (Wx.T @ da)
                    dh = Wh.T @ da
        self._np = {"E": E, "Wx": Wx, "Wh": Wh, "bh": bh, "Wo": Wo, "bo": bo}

    # ─── bashorat ─────────────────────────────────────────────────────────────
    def _logits(self, text: str) -> np.ndarray:
        ids = self._encode(text)
        if HAS_TORCH and self._model is not None:
            with torch.no_grad():
                x = torch.tensor([ids], dtype=torch.long)
                return self._model(x)[0].numpy()
        p = self._np
        h = np.zeros(self._hidden)
        for tid in ids:
            h = np.tanh(p["Wx"] @ p["E"][tid] + p["Wh"] @ h + p["bh"])
        return p["Wo"] @ h + p["bo"]

    def predict(self, text: str) -> str:
        return self._labels[int(np.argmax(self._logits(text)))]

    def predict_proba(self, text: str) -> dict[str, float]:
        z = self._logits(text)
        e = np.exp(z - z.max()); e /= e.sum()
        return {lab: float(e[i]) for i, lab in enumerate(self._labels)}

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        state = {"w2i": self._w2i, "labels": self._labels, "dim": self._dim,
                 "hidden": self._hidden, "has_torch": HAS_TORCH}
        if HAS_TORCH and self._model is not None:
            state["torch"] = self._model.state_dict()
        else:
            state["np"] = self._np
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._w2i, self._labels = s["w2i"], s["labels"]
        self._dim, self._hidden = s["dim"], s["hidden"]
        if HAS_TORCH and "torch" in s:
            self._model = _TorchRNN(len(self._w2i) + 1, self._dim, self._hidden, len(self._labels))
            self._model.load_state_dict(s["torch"]); self._model.eval()
            self._np = None
        else:
            self._np = s.get("np")
            self._model = None
