"""
m10_ner_tagger.py — Kapstone modul: BiLSTM asosidagi NER tagger.

Shartnoma:
    class NERTagger:
        TAGSET = ["O","B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG"]
        def __init__(embed_dim=32, hidden_size=32, epochs=8, lr=1e-2)
        def fit(tagged_sentences: list[list[tuple[str, str]]]) -> "NERTagger"
        def predict(text: str) -> list[tuple[str, str]]
        def entities(text: str) -> list[dict]
        def save(path: str) -> None
        def load(path: str) -> None

PyTorch bo'lsa BiLSTM tagger, aks holda token->tag lookup fallback.
"""
from __future__ import annotations
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


TAGSET = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]


if HAS_TORCH:
    class _BiLSTM_NER(nn.Module):
        """Ichki BiLSTM NER modeli."""
        def __init__(self, vocab_size, embed_dim, hidden_size, num_tags):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True,
                                 bidirectional=True)
            self.out = nn.Linear(hidden_size * 2, num_tags)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            return self.out(out)


class NERTagger:
    """BiLSTM NER tagger (PyTorch mavjud bo'lsa) yoki lookup fallback."""

    TAGSET = TAGSET

    def __init__(self, embed_dim: int = 32, hidden_size: int = 32,
                 epochs: int = 8, lr: float = 1e-2):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.tag_to_index = {tag: i for i, tag in enumerate(self.TAGSET)}
        self.index_to_tag = {i: tag for tag, i in self.tag_to_index.items()}
        self.lookup: Dict[str, str] = {}
        self.default_tag = "O"
        self.model = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.split()

    def _build_lookup(self, tagged_sentences):
        token_tag_counts: Dict[str, Counter] = defaultdict(Counter)
        all_tag_counts: Counter = Counter()
        for sent in tagged_sentences:
            for token, tag in sent:
                token_tag_counts[token][tag] += 1
                all_tag_counts[tag] += 1
        for token, tags in token_tag_counts.items():
            self.lookup[token] = tags.most_common(1)[0][0]
        self.default_tag = all_tag_counts.most_common(1)[0][0] if all_tag_counts else "O"

    def _build_vocab(self, tagged_sentences):
        for sent in tagged_sentences:
            for token, _ in sent:
                if token not in self.word_to_index:
                    self.word_to_index[token] = len(self.word_to_index)

    def _encode_tokens(self, tokens):
        return [self.word_to_index.get(t, self.word_to_index["<UNK>"]) for t in tokens]

    def fit(self, tagged_sentences):
        """[[(token, tag), ...], ...] shaklidagi ma'lumot."""
        if not tagged_sentences:
            raise ValueError("Bo'sh o'quv ma'lumoti")

        self._build_lookup(tagged_sentences)
        self._build_vocab(tagged_sentences)

        if not HAS_TORCH:
            return self

        torch.manual_seed(42)
        vocab_size = len(self.word_to_index)
        num_tags = len(self.TAGSET)

        self.model = _BiLSTM_NER(vocab_size, self.embed_dim, self.hidden_size, num_tags)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(self.epochs):
            for sent in tagged_sentences:
                tokens = [t for t, _ in sent]
                tags = [self.tag_to_index.get(tag, 0) for _, tag in sent]
                x = torch.tensor([self._encode_tokens(tokens)], dtype=torch.long)
                y = torch.tensor([tags], dtype=torch.long)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, num_tags), y.view(-1))
                loss.backward()
                optimizer.step()

        return self

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = self.tokenize(text)
        if not HAS_TORCH or self.model is None:
            return [(t, self.lookup.get(t, self.default_tag)) for t in tokens]

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([self._encode_tokens(tokens)], dtype=torch.long)
            logits = self.model(x)
            tag_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            # Ensure it's a list (not scalar)
            if not isinstance(tag_ids, list):
                tag_ids = [tag_ids]
            tags = [self.index_to_tag[tid] for tid in tag_ids]
        return list(zip(tokens, tags))

    def entities(self, text: str) -> List[Dict]:
        """IOB2 taglardan entity ro'yxatini ajratadi."""
        tagged = self.predict(text)
        tokens = self.tokenize(text)

        # Belgi (char) pozitsiyalarini hisoblaymiz
        char_offsets = []
        pos = 0
        for tok in tokens:
            idx = text.find(tok, pos)
            if idx < 0:
                idx = pos
            char_offsets.append((idx, idx + len(tok)))
            pos = idx + len(tok)

        results = []
        current = None
        for i, (tok, tag) in enumerate(tagged):
            start, end = char_offsets[i] if i < len(char_offsets) else (0, 0)
            if tag.startswith("B-"):
                if current:
                    results.append(current)
                current = {
                    "start": start, "end": end,
                    "label": tag[2:], "text": tok
                }
            elif tag.startswith("I-") and current and current["label"] == tag[2:]:
                current["end"] = end
                current["text"] = text[current["start"]:end]
            else:
                if current:
                    results.append(current)
                    current = None
        if current:
            results.append(current)
        return results

    def save(self, path: str) -> None:
        state = {
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "word_to_index": self.word_to_index,
            "lookup": self.lookup,
            "default_tag": self.default_tag,
            "torch_state": self.model.state_dict() if (HAS_TORCH and self.model) else None,
        }
        Path(path).write_bytes(pickle.dumps(state))

    def load(self, path: str) -> None:
        state = pickle.loads(Path(path).read_bytes())
        self.embed_dim = state["embed_dim"]
        self.hidden_size = state["hidden_size"]
        self.epochs = state["epochs"]
        self.lr = state["lr"]
        self.word_to_index = state["word_to_index"]
        self.lookup = state["lookup"]
        self.default_tag = state["default_tag"]

        if HAS_TORCH and state["torch_state"] is not None:
            self.model = _BiLSTM_NER(
                len(self.word_to_index), self.embed_dim,
                self.hidden_size, len(self.TAGSET)
            )
            self.model.load_state_dict(state["torch_state"])
        else:
            self.model = None
