"""
m11_seq2seq_translator.py — Kapstone modul: LSTM + Attention seq2seq tarjimon.

Shartnoma:
    class Seq2SeqTranslator:
        def __init__(embed_dim=32, hidden_size=64)
        def train(src_texts, tgt_texts, epochs=10, max_len=50)
        def translate(text: str) -> str
        def bleu(references, hypotheses) -> float
        def save(path: str) -> None
        def load(path: str) -> None
        last_attention: torch.Tensor | None  # oxirgi tarjima uchun

O'zbekcha-inglizcha yoki har qanday til juftligi uchun ishlaydi.
"""
from __future__ import annotations
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


if HAS_TORCH:
    class _Encoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
            self.recurrent = nn.LSTM(embed_dim, hidden_size, batch_first=True)

        def forward(self, source_ids):
            embedded = self.embedding(source_ids)
            outputs, (hidden, cell) = self.recurrent(embedded)
            return outputs, hidden[-1], cell[-1]

    class _AdditiveAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
            self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
            self.score_layer = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, query, encoder_outputs):
            Q = self.query_layer(query).unsqueeze(1)
            K = self.key_layer(encoder_outputs)
            scores = self.score_layer(torch.tanh(Q + K)).squeeze(-1)
            alpha = torch.softmax(scores, dim=-1)
            context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
            return alpha, context

    class _DecoderStep(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, attention):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
            self.attention = attention
            self.cell = nn.LSTMCell(embed_dim + hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size * 2, vocab_size)

        def forward(self, previous_token, hidden, cell, encoder_outputs):
            alpha, context = self.attention(hidden, encoder_outputs)
            embedded = self.embedding(previous_token)
            combined = torch.cat([embedded, context], dim=-1)
            hidden, cell = self.cell(combined, (hidden, cell))
            output_input = torch.cat([hidden, context], dim=-1)
            logits = self.output(output_input)
            return logits, hidden, cell, alpha

    class _TinySeq2Seq(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=32, hidden_size=64):
            super().__init__()
            self.encoder = _Encoder(src_vocab_size, embed_dim, hidden_size)
            self.attention = _AdditiveAttention(hidden_size)
            self.decoder = _DecoderStep(tgt_vocab_size, embed_dim, hidden_size, self.attention)

        def forward(self, source_ids, decoder_input_ids):
            encoder_outputs, hidden, cell = self.encoder(source_ids)
            logits_by_step = []
            attention_by_step = []
            for step in range(decoder_input_ids.size(1)):
                prev = decoder_input_ids[:, step]
                logits, hidden, cell, alpha = self.decoder(prev, hidden, cell, encoder_outputs)
                logits_by_step.append(logits)
                attention_by_step.append(alpha)
            return torch.stack(logits_by_step, dim=1), torch.stack(attention_by_step, dim=1)


def _corpus_bleu(references, hypotheses, max_n=4):
    """Sodda BLEU-4 hisobi (brevity penalty siz)."""
    import math
    from collections import Counter
    weights = [1.0/max_n] * max_n
    total_score = 0.0
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            for ng, cnt in hyp_ngrams.items():
                matches += min(cnt, ref_ngrams.get(ng, 0))
                total += cnt
        p_n = matches / total if total else 0.0
        if p_n > 0:
            total_score += weights[n-1] * math.log(p_n)
        else:
            return 0.0
    return math.exp(total_score)


class Seq2SeqTranslator:
    """LSTM + Additive Attention asosidagi tarjimon."""

    def __init__(self, embed_dim: int = 32, hidden_size: int = 64):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.src_vocab = {}
        self.tgt_vocab = {}
        self.tgt_index_to_word = {}
        self.model = None
        self.last_attention = None

    def _build_vocab(self, texts):
        vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<SOS>": SOS_ID, "<EOS>": EOS_ID}
        for text in texts:
            for tok in text.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        return vocab

    def _encode(self, text, vocab):
        return [vocab.get(t, UNK_ID) for t in text.lower().split()]

    def train(self, src_texts, tgt_texts, epochs: int = 10, max_len: int = 50):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch talab qilinadi (m11 uchun).")
        if not src_texts or len(src_texts) != len(tgt_texts):
            raise ValueError("src_texts va tgt_texts uzunligi mos kelmadi.")

        self.src_vocab = self._build_vocab(src_texts)
        self.tgt_vocab = self._build_vocab(tgt_texts)
        self.tgt_index_to_word = {i: w for w, i in self.tgt_vocab.items()}

        def pad(seq, L):
            return seq[:L] + [PAD_ID] * max(0, L - len(seq))

        src_ids = [self._encode(t, self.src_vocab) for t in src_texts]
        tgt_ids_in = [[SOS_ID] + self._encode(t, self.tgt_vocab) for t in tgt_texts]
        tgt_ids_out = [self._encode(t, self.tgt_vocab) + [EOS_ID] for t in tgt_texts]

        L_src = min(max(len(s) for s in src_ids), max_len)
        L_tgt = min(max(len(s) for s in tgt_ids_out), max_len)

        X = torch.tensor([pad(s, L_src) for s in src_ids], dtype=torch.long)
        Y_in = torch.tensor([pad(s, L_tgt) for s in tgt_ids_in], dtype=torch.long)
        Y_out = torch.tensor([pad(s, L_tgt) for s in tgt_ids_out], dtype=torch.long)

        torch.manual_seed(42)
        self.model = _TinySeq2Seq(
            len(self.src_vocab), len(self.tgt_vocab),
            embed_dim=self.embed_dim, hidden_size=self.hidden_size
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = self.model(X, Y_in)
            loss = loss_fn(logits.reshape(-1, len(self.tgt_vocab)), Y_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return self

    def translate(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Avval train() ni chaqiring.")
        self.model.eval()
        src_ids = torch.tensor([self._encode(text, self.src_vocab)], dtype=torch.long)
        output_ids = []
        attention_rows = []
        with torch.no_grad():
            enc_out, hidden, cell = self.model.encoder(src_ids)
            prev = torch.tensor([SOS_ID], dtype=torch.long)
            for _ in range(20):
                logits, hidden, cell, alpha = self.model.decoder(prev, hidden, cell, enc_out)
                next_id = int(logits.argmax(dim=-1).item())
                attention_rows.append(alpha[0])
                if next_id == EOS_ID:
                    break
                output_ids.append(next_id)
                prev = torch.tensor([next_id], dtype=torch.long)
        self.last_attention = torch.stack(attention_rows) if attention_rows else None
        return " ".join(self.tgt_index_to_word.get(i, "<UNK>") for i in output_ids)

    def bleu(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        return _corpus_bleu(references, hypotheses)

    def save(self, path: str) -> None:
        state = {
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "src_vocab": self.src_vocab,
            "tgt_vocab": self.tgt_vocab,
            "tgt_index_to_word": self.tgt_index_to_word,
            "torch_state": self.model.state_dict() if (HAS_TORCH and self.model) else None,
        }
        Path(path).write_bytes(pickle.dumps(state))

    def load(self, path: str) -> None:
        state = pickle.loads(Path(path).read_bytes())
        self.embed_dim = state["embed_dim"]
        self.hidden_size = state["hidden_size"]
        self.src_vocab = state["src_vocab"]
        self.tgt_vocab = state["tgt_vocab"]
        self.tgt_index_to_word = state["tgt_index_to_word"]

        if HAS_TORCH and state["torch_state"] is not None:
            self.model = _TinySeq2Seq(
                len(self.src_vocab), len(self.tgt_vocab),
                embed_dim=self.embed_dim, hidden_size=self.hidden_size
            )
            self.model.load_state_dict(state["torch_state"])
        else:
            self.model = None
