"""
capstone/modules/m08_gru_lstm_classifier.py

GRUClassifier, LSTMClassifier, and GRULSTMClassifier for Uzbek sentiment
classification. The separate classes make the architecture distinction explicit;
GRULSTMClassifier remains the capstone contract wrapper used for comparison.

Torch path:
  nn.Embedding + nn.GRU/nn.LSTM + nn.Linear, CrossEntropyLoss + Adam.

Offline path:
  Random recurrent GRU/LSTM-style encoder (reservoir) + tiny softmax readout.
  This keeps the notebook runnable without torch or sklearn while preserving the
  gate dynamics students just studied.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _macro_f1(y_true, y_pred):
    scores = []
    for label in sorted(set(y_true) | set(y_pred)):
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


class _SoftmaxReadout:
    """Small sklearn-free multinomial readout for offline notebooks."""

    def __init__(self, epochs=250, lr=0.2):
        self.epochs = epochs
        self.lr = lr
        self.W = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_examples, n_features = X.shape
        n_classes = int(y.max()) + 1
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(n_examples), y] -= 1.0
            self.W -= self.lr * (X.T @ probs) / n_examples
            self.b -= self.lr * probs.mean(axis=0)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logits = X @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if HAS_TORCH:

    class _TorchSequenceNet(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, n_classes, arch):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
            self.recurrent = rnn_cls(
                embed_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, n_classes)
            self.arch = arch

        def forward(self, token_ids):
            embedded = self.embedding(token_ids)
            _, hidden = self.recurrent(embedded)
            last_hidden = hidden[0] if self.arch == "lstm" else hidden
            return self.output(last_hidden[-1])


@dataclass
class _TrainingConfig:
    epochs: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    lr: float = 1e-3


class _BaseSequenceClassifier:
    """Shared implementation for the concrete GRU and LSTM classifiers."""

    arch = ""

    def __init__(self, embed_dim: int = 32) -> None:
        self._pre = TextPreprocessor()
        self._dim = embed_dim
        self._w2i: dict[str, int] = {}
        self._labels: list[str] = []
        self._config = _TrainingConfig()
        self._model = None
        self._np = None
        self._train_cache = None

    def _tokenize(self, text: str) -> list[str]:
        return self._pre.preprocess(text) if text.strip() else []

    def _prepare(self, texts, labels):
        token_lists = [self._tokenize(text) for text in texts]
        vocab = sorted({token for tokens in token_lists for token in tokens})
        self._w2i = {word: index + 1 for index, word in enumerate(vocab)}
        self._labels = sorted(set(labels))
        label_to_index = {label: index for index, label in enumerate(self._labels)}
        sequences = [
            [self._w2i[token] for token in tokens if token in self._w2i] or [0]
            for tokens in token_lists
        ]
        y = [label_to_index[label] for label in labels]
        return sequences, y

    def _encode(self, text: str) -> list[int]:
        token_ids = [self._w2i[token] for token in self._tokenize(text) if token in self._w2i]
        return token_ids or [0]

    @staticmethod
    def _pad(sequences):
        max_len = max(len(sequence) for sequence in sequences)
        return [sequence + [0] * (max_len - len(sequence)) for sequence in sequences]

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        epochs: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 1e-3,
    ) -> None:
        self._config = _TrainingConfig(epochs, hidden_size, num_layers, lr)
        sequences, y = self._prepare(texts, labels)
        self._train_cache = (list(texts), list(labels))

        if HAS_TORCH:
            self._fit_torch(sequences, y)
        else:
            self._fit_numpy(sequences, y)

    def _fit_torch(self, sequences, y):
        torch.manual_seed(42)
        X = torch.tensor(self._pad(sequences), dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        self._model = _TorchSequenceNet(
            len(self._w2i) + 1,
            self._dim,
            self._config.hidden_size,
            self._config.num_layers,
            len(self._labels),
            self.arch,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        loss_fn = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(self._config.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._model(X), y_tensor)
            loss.backward()
            optimizer.step()
        self._model.eval()

    def _new_reservoir(self):
        rng = np.random.RandomState(42)
        vocab_size = len(self._w2i) + 1
        embed_dim = self._dim
        hidden_size = self._config.hidden_size
        n_gates = 4 if self.arch == "lstm" else 3
        return {
            "E": rng.randn(vocab_size, embed_dim) * 0.3,
            "W": rng.randn(n_gates, hidden_size, hidden_size + embed_dim)
            * (1.0 / np.sqrt(hidden_size + embed_dim)),
            "hidden_size": hidden_size,
            "arch": self.arch,
        }

    def _encode_state(self, reservoir, sequence):
        hidden_size = reservoir["hidden_size"]
        hidden_state = np.zeros(hidden_size)
        cell_state = np.zeros(hidden_size)

        for token_id in sequence:
            embedded_token = reservoir["E"][token_id]
            if reservoir["arch"] == "lstm":
                joined = np.concatenate([hidden_state, embedded_token])
                forget_gate = _sigmoid(reservoir["W"][0] @ joined)
                input_gate = _sigmoid(reservoir["W"][1] @ joined)
                output_gate = _sigmoid(reservoir["W"][2] @ joined)
                candidate = np.tanh(reservoir["W"][3] @ joined)
                cell_state = forget_gate * cell_state + input_gate * candidate
                hidden_state = output_gate * np.tanh(cell_state)
            else:
                joined = np.concatenate([hidden_state, embedded_token])
                update_gate = _sigmoid(reservoir["W"][0] @ joined)
                reset_gate = _sigmoid(reservoir["W"][1] @ joined)
                candidate_input = np.concatenate([reset_gate * hidden_state, embedded_token])
                candidate = np.tanh(reservoir["W"][2] @ candidate_input)
                hidden_state = (1 - update_gate) * hidden_state + update_gate * candidate

        return hidden_state

    def _fit_numpy(self, sequences, y):
        reservoir = self._new_reservoir()
        states = np.array([self._encode_state(reservoir, sequence) for sequence in sequences])
        readout = _SoftmaxReadout().fit(states, y)
        self._np = {"reservoir": reservoir, "readout": readout}
        self._model = None

    def _proba(self, text: str) -> np.ndarray:
        token_ids = self._encode(text)
        if HAS_TORCH and self._model is not None:
            with torch.no_grad():
                logits = self._model(torch.tensor([token_ids], dtype=torch.long))[0].numpy()
            probs = np.exp(logits - logits.max())
            return probs / probs.sum()

        state = self._encode_state(self._np["reservoir"], token_ids)
        return self._np["readout"].predict_proba([state])[0]

    def predict(self, text: str) -> str:
        return self._labels[int(np.argmax(self._proba(text)))]

    def predict_proba(self, text: str) -> dict[str, float]:
        probs = self._proba(text)
        return {label: float(probs[index]) for index, label in enumerate(self._labels)}

    def evaluate(self, texts: list[str], labels: list[str]) -> dict[str, float]:
        start = time.perf_counter()
        predictions = [self.predict(text) for text in texts]
        inference_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        accuracy = float(np.mean(np.array(predictions) == np.array(labels)))
        return {
            "f1": round(_macro_f1(labels, predictions), 4),
            "accuracy": round(accuracy, 4),
            "inference_time": round(inference_ms, 4),
        }

    def save(self, path: str) -> None:
        state = {
            "w2i": self._w2i,
            "labels": self._labels,
            "dim": self._dim,
            "config": self._config,
            "arch": self.arch,
            "has_torch": HAS_TORCH,
        }
        if HAS_TORCH and self._model is not None:
            state["torch"] = self._model.state_dict()
        else:
            state["np"] = self._np
        with open(path, "wb") as file:
            pickle.dump(state, file)

    def load(self, path: str) -> None:
        with open(path, "rb") as file:
            state = pickle.load(file)
        self._w2i = state["w2i"]
        self._labels = state["labels"]
        self._dim = state["dim"]
        self._config = state["config"]

        if HAS_TORCH and "torch" in state:
            self._model = _TorchSequenceNet(
                len(self._w2i) + 1,
                self._dim,
                self._config.hidden_size,
                self._config.num_layers,
                len(self._labels),
                self.arch,
            )
            self._model.load_state_dict(state["torch"])
            self._model.eval()
            self._np = None
        else:
            self._np = state.get("np")
            self._model = None


class GRUClassifier(_BaseSequenceClassifier):
    """GRU-only sentiment classifier."""

    arch = "gru"


class LSTMClassifier(_BaseSequenceClassifier):
    """LSTM-only sentiment classifier."""

    arch = "lstm"


class GRULSTMClassifier:
    """Capstone wrapper: train one architecture and compare GRU vs LSTM."""

    def __init__(self, embed_dim: int = 32) -> None:
        self._dim = embed_dim
        self._active = LSTMClassifier(embed_dim=embed_dim)
        self._train_args = None
        self._w2i: dict[str, int] = {}
        self._labels: list[str] = []
        self._arch = "lstm"

    def _new_model(self, arch: str):
        if arch == "lstm":
            return LSTMClassifier(embed_dim=self._dim)
        if arch == "gru":
            return GRUClassifier(embed_dim=self._dim)
        raise ValueError("arch must be 'lstm' or 'gru'")

    def _sync_public_state(self):
        self._w2i = self._active._w2i
        self._labels = self._active._labels
        self._arch = self._active.arch

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        arch: str = "lstm",
        epochs: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 1e-3,
    ) -> None:
        self._active = self._new_model(arch)
        self._active.fit(
            texts,
            labels,
            epochs=epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lr=lr,
        )
        self._train_args = (list(texts), list(labels), epochs, hidden_size, num_layers, lr)
        self._sync_public_state()

    def predict(self, text: str) -> str:
        return self._active.predict(text)

    def predict_proba(self, text: str) -> dict[str, float]:
        return self._active.predict_proba(text)

    def compare_report(self) -> dict:
        assert self._train_args is not None, "Avval fit() chaqiring."
        texts, labels, epochs, hidden_size, num_layers, lr = self._train_args
        report = {}
        for arch, model_cls in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
            model = model_cls(embed_dim=self._dim)
            model.fit(
                texts,
                labels,
                epochs=epochs,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lr=lr,
            )
            report[arch] = model.evaluate(texts, labels)
        return report

    def save(self, path: str) -> None:
        state = {
            "dim": self._dim,
            "arch": self._arch,
            "train_args": self._train_args,
            "active": self._active,
        }
        with open(path, "wb") as file:
            pickle.dump(state, file)

    def load(self, path: str) -> None:
        with open(path, "rb") as file:
            state = pickle.load(file)
        self._dim = state["dim"]
        self._train_args = state["train_args"]
        self._active = state["active"]
        self._sync_public_state()
