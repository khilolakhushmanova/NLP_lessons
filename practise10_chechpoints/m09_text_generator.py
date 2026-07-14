"""
capstone/modules/m09_text_generator.py

TextGenerator for a small char-level text generation practice.

Torch path:
  nn.Embedding + nn.LSTM + nn.Linear trained to predict the next character.

Offline path:
  Character n-gram model with temperature sampling. This keeps the capstone
  runnable in local environments where torch is not installed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class _CharLSTM(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.recurrent = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, character_ids, hidden=None):
            embedded = self.embedding(character_ids)
            outputs, hidden = self.recurrent(embedded, hidden)
            return self.output(outputs), hidden


@dataclass
class _TrainingConfig:
    epochs: int = 20
    hidden_size: int = 128
    lr: float = 1e-3


class TextGenerator:
    """Char-level LSTM text generator with an n-gram fallback."""

    def __init__(self, ngram_order: int = 3, random_state: int = 42) -> None:
        self.ngram_order = ngram_order
        self.random_state = random_state
        self.characters: list[str] = []
        self.char_to_index: dict[str, int] = {}
        self.index_to_char: dict[int, str] = {}
        self.config = _TrainingConfig()
        self.model = None
        self.context_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.unigram_counts: Counter[str] = Counter()
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _softmax(logits, temperature: float = 1.0) -> np.ndarray:
        temperature = max(float(temperature), 1e-8)
        scaled = np.asarray(logits, dtype=float) / temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        return probabilities / probabilities.sum()

    def train(self, text: str, epochs: int = 20, hidden_size: int = 128) -> None:
        """Train the generator on one raw text corpus."""
        if not isinstance(text, str) or len(text) < 2:
            raise ValueError("TextGenerator.train() needs at least two characters.")

        self.config = _TrainingConfig(epochs=epochs, hidden_size=hidden_size)
        self.characters = sorted(set(text))
        self.char_to_index = {
            character: index for index, character in enumerate(self.characters)
        }
        self.index_to_char = {
            index: character for character, index in self.char_to_index.items()
        }

        self._fit_ngram(text)
        if HAS_TORCH:
            self._fit_torch(text)

    def _fit_ngram(self, text: str) -> None:
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter(text)
        padded_text = "\n" * self.ngram_order + text
        for position in range(self.ngram_order, len(padded_text)):
            next_character = padded_text[position]
            for context_size in range(1, self.ngram_order + 1):
                context = padded_text[position - context_size : position]
                self.context_counts[context][next_character] += 1

    def _fit_torch(self, text: str) -> None:
        torch.manual_seed(self.random_state)
        vocab_size = len(self.characters)
        sequence = torch.tensor(
            [self.char_to_index[character] for character in text],
            dtype=torch.long,
        )
        inputs = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        model = _CharLSTM(vocab_size, self.config.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        self.model = model

    def generate(self, seed: str, length: int = 200, temperature: float = 0.7) -> str:
        """Generate a character continuation from seed."""
        if not self.characters:
            raise RuntimeError("Call train() before generate().")
        if length < 0:
            raise ValueError("length must be non-negative.")
        if not seed:
            seed = self.unigram_counts.most_common(1)[0][0]

        if HAS_TORCH and self.model is not None:
            return self._generate_torch(seed, length, temperature)
        return self._generate_ngram(seed, length, temperature)

    def _generate_torch(self, seed: str, length: int, temperature: float) -> str:
        result = list(seed)
        known_seed = [self.char_to_index.get(character, 0) for character in seed]
        if not known_seed:
            known_seed = [0]

        with torch.no_grad():
            inputs = torch.tensor([known_seed], dtype=torch.long)
            logits, hidden = self.model(inputs)
            for _ in range(length):
                probabilities = self._softmax(logits[0, -1].numpy(), temperature)
                sampled_index = int(self.rng.choice(len(probabilities), p=probabilities))
                sampled_character = self.index_to_char[sampled_index]
                result.append(sampled_character)
                next_input = torch.tensor([[sampled_index]], dtype=torch.long)
                logits, hidden = self.model(next_input, hidden)
        return "".join(result)

    def _generate_ngram(self, seed: str, length: int, temperature: float) -> str:
        result = list(seed)
        for _ in range(length):
            next_character = self._sample_from_context("".join(result), temperature)
            result.append(next_character)
        return "".join(result)

    def _sample_from_context(self, prefix: str, temperature: float) -> str:
        for context_size in range(min(self.ngram_order, len(prefix)), 0, -1):
            context = prefix[-context_size:]
            if context in self.context_counts:
                return self._sample_counter(self.context_counts[context], temperature)
        return self._sample_counter(self.unigram_counts, temperature)

    def _sample_counter(self, counts: Counter[str], temperature: float) -> str:
        if not counts:
            counts = self.unigram_counts
        characters = np.array(sorted(counts))
        raw_counts = np.array([counts[character] for character in characters])
        probabilities = self._softmax(np.log(raw_counts), temperature)
        return str(self.rng.choice(characters, p=probabilities))
