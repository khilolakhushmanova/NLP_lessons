"""
capstone/modules/m05b_pos_tagger.py
POSTagger — Yashirin Markov Modeli + Viterbi orqali so'z turkumini teglash.
Shartnoma: capstone/contracts.py :: POSTagger
P5 (6-kun amaliyoti) da qurilgan. Pedagogik demo — yakuniy pipelineda ishlatilmaydi.

Viterbi log fazoda hisoblanadi (underflow oldini olish — L5 [L4]-slayd).
"""
from __future__ import annotations

import math
import pickle
from collections import Counter

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor

_EPS = 1e-12


class POSTagger:
    """Yashirin Markov Modeli + Viterbi orqali so'z turkumini teglash."""

    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()      # m01 — token normalizatsiyasi
        self.states: list[str] = []
        self.start_probs: dict = {}
        self.transition_probs: dict = {}            # (from, to) -> prob
        self.emission_probs: dict = {}              # (state, word) -> prob

    def train(self, tagged_sentences: list) -> None:
        """HMM parametrlarini (pi, A, B) sanab hisoblaydi."""
        start_counts = Counter()
        transition_counts = Counter()
        emission_counts = Counter()
        tag_counts = Counter()
        sentence_count = 0
        for tagged_sentence in tagged_sentences:
            if not tagged_sentence:
                continue
            sentence_count += 1
            first_tag = tagged_sentence[0][1]
            start_counts[first_tag] += 1
            for index, (word, tag) in enumerate(tagged_sentence):
                emission_counts[(tag, word.lower())] += 1
                tag_counts[tag] += 1
                if index > 0:
                    previous_tag = tagged_sentence[index - 1][1]
                    transition_counts[(previous_tag, tag)] += 1

        self.states = sorted(tag_counts)
        self.start_probs = (
            {state: start_counts[state] / sentence_count for state in self.states}
            if sentence_count else {}
        )

        transition_totals = Counter()
        for (from_state, _to_state), count in transition_counts.items():
            transition_totals[from_state] += count

        self.transition_probs = {}
        for from_state in self.states:
            for to_state in self.states:
                denominator = transition_totals[from_state]
                self.transition_probs[(from_state, to_state)] = (
                    transition_counts[(from_state, to_state)] / denominator
                    if denominator else 0.0
                )
        self.emission_probs = {
            (tag, word): count / tag_counts[tag]
            for (tag, word), count in emission_counts.items()
        }

    def _emit(self, state: str, word: str) -> float:
        return self.emission_probs.get((state, word.lower()), 1e-6)

    def tag(self, tokens: list[str]) -> list:
        """Viterbi (log fazo) bilan har token uchun teg bashorat qiladi."""
        if not tokens or not self.states:
            return [(w, "") for w in tokens]
        normalized_tokens = [self.preprocessor._normalize(token) for token in tokens]
        states = self.states

        def safe_log(probability: float) -> float:
            return math.log(probability + _EPS)

        delta_table = [{
            state: safe_log(self.start_probs.get(state, 0.0))
                   + safe_log(self._emit(state, normalized_tokens[0]))
            for state in states
        }]
        backpointer_table = [{}]

        for index in range(1, len(normalized_tokens)):
            current_delta = {}
            current_backpointers = {}
            for state in states:
                best_previous_state = max(
                    states,
                    key=lambda previous_state:
                        delta_table[index - 1][previous_state]
                        + safe_log(self.transition_probs.get((previous_state, state), 0.0))
                )
                current_delta[state] = (
                    delta_table[index - 1][best_previous_state]
                    + safe_log(self.transition_probs.get((best_previous_state, state), 0.0))
                    + safe_log(self._emit(state, normalized_tokens[index]))
                )
                current_backpointers[state] = best_previous_state
            delta_table.append(current_delta)
            backpointer_table.append(current_backpointers)

        last_state = max(states, key=lambda state: delta_table[-1][state])
        best_path = [last_state]
        for index in range(len(normalized_tokens) - 1, 0, -1):
            last_state = backpointer_table[index][last_state]
            best_path.insert(0, last_state)
        return list(zip(tokens, best_path))
