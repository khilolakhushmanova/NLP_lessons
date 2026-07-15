"""
m12_transformer_summarizer.py — Kapstone modul: Mini Transformer matn qisqartiruvchi.

Shartnoma:
    class TransformerSummarizer:
        def train(src_texts, tgt_texts, epochs=10, d_model=128, nhead=4)
        def summarize(text, max_length=60) -> str
        def rouge1(references, hypotheses) -> dict[str, float]
        def save(path) -> None
        def load(path) -> None

d13_p12_transformer_summ.ipynb dan olingan. PyTorch mavjud bo'lmasa ValueError.
"""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PAD, BOS, EOS, UNK = 0, 1, 2, 3
_SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("`", "'").replace("’", "'").replace("‘", "'")
    return re.findall(r"[a-zA-Z0-9']+", text)


def _build_vocab(texts: list[str]) -> tuple[dict, dict]:
    counts: dict = {}
    for text in texts:
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = list(_SPECIAL) + [tok for tok in sorted(counts)]
    t2i = {tok: i for i, tok in enumerate(vocab)}
    i2t = {i: tok for tok, i in t2i.items()}
    return t2i, i2t


def _encode(text: str, t2i: dict, max_len: int, add_bos: bool = False, add_eos: bool = True) -> list[int]:
    ids = [t2i.get(tok, UNK) for tok in _tokenize(text)]
    if add_bos:
        ids = [BOS] + ids
    if add_eos:
        ids = ids + [EOS]
    ids = ids[:max_len]
    if add_eos and ids and ids[-1] != EOS:
        ids[-1] = EOS
    return ids


def _pad(seqs: list[list[int]]) -> "torch.Tensor":
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _decode(ids: list[int], i2t: dict) -> str:
    words = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS:
            break
        if idx in (PAD, BOS):
            continue
        words.append(i2t.get(idx, "<unk>"))
    return " ".join(words)


if HAS_TORCH:
    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 256):
            super().__init__()
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.pe[:, :x.size(1)]

    class _TinyTransformer(nn.Module):
        def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, nhead: int,
                     num_layers: int = 2, dim_ff: int = 256):
            super().__init__()
            self.d_model = d_model
            self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
            self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
            self.pe = _PositionalEncoding(d_model)
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead,
                num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                dim_feedforward=dim_ff, dropout=0.1, batch_first=True,
            )
            self.out_proj = nn.Linear(d_model, tgt_vocab)

        def forward(self, src: "torch.Tensor", tgt_in: "torch.Tensor") -> "torch.Tensor":
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_in.size(1)
            ).to(src.device)
            src_emb = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
            tgt_emb = self.pe(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
            hidden = self.transformer(
                src_emb, tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src.eq(PAD),
                tgt_key_padding_mask=tgt_in.eq(PAD),
                memory_key_padding_mask=src.eq(PAD),
            )
            return self.out_proj(hidden)


class TransformerSummarizer:
    """Mini Transformer Encoder-Decoder asosida matn qisqartirish.

    Corpus: Wikipedia uz lead-paragraph juftlari (CC-BY-SA).
    Consumed by: m15 (agent tool: summarize_text), Day 16 (pipeline).
    """

    _MAX_SRC = 64
    _MAX_TGT = 24

    def __init__(self) -> None:
        self._model = None
        self._src_t2i: dict = {}
        self._src_i2t: dict = {}
        self._tgt_t2i: dict = {}
        self._tgt_i2t: dict = {}
        self._d_model = 128
        self._nhead = 4
        self._device = "cpu"

    def train(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        epochs: int = 10,
        d_model: int = 128,
        nhead: int = 4,
    ) -> None:
        """Transformer modelini o'qitadi."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch talab qilinadi (m12 uchun).")
        if not src_texts:
            raise ValueError("src_texts bo'sh bo'lmasligi kerak.")
        if len(src_texts) != len(tgt_texts):
            raise ValueError(
                f"src_texts ({len(src_texts)}) va tgt_texts ({len(tgt_texts)}) uzunligi mos kelmadi."
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} nhead={nhead} ga bo'linishi kerak.")

        self._d_model = d_model
        self._nhead = nhead
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._src_t2i, self._src_i2t = _build_vocab(src_texts)
        self._tgt_t2i, self._tgt_i2t = _build_vocab(tgt_texts)

        src_ids = [
            _encode(t, self._src_t2i, self._MAX_SRC, add_bos=False, add_eos=True)
            for t in src_texts
        ]
        tgt_ids = [
            _encode(t, self._tgt_t2i, self._MAX_TGT, add_bos=True, add_eos=True)
            for t in tgt_texts
        ]

        src_batch = _pad(src_ids).to(self._device)
        tgt_batch = _pad(tgt_ids).to(self._device)
        dec_in = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]

        torch.manual_seed(42)
        self._model = _TinyTransformer(
            len(self._src_t2i), len(self._tgt_t2i),
            d_model=d_model, nhead=nhead,
        ).to(self._device)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.003)

        self._model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self._model(src_batch, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()

    def summarize(self, text: str, max_length: int = 60) -> str:
        """Berilgan matn uchun qisqa xulosa generatsiya qiladi."""
        if self._model is None:
            raise RuntimeError("Avval train() ni chaqiring.")
        if not text:
            raise ValueError("Matn bo'sh bo'lmasligi kerak.")

        self._model.eval()
        src = _pad(
            [_encode(text, self._src_t2i, self._MAX_SRC, add_bos=False, add_eos=True)]
        ).to(self._device)
        generated = torch.tensor([[BOS]], dtype=torch.long, device=self._device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self._model(src, generated)
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                if int(next_id.item()) == EOS:
                    break

        return _decode(generated[0].tolist(), self._tgt_i2t)

    def rouge1(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        """ROUGE-1 precision, recall, F1 ni qaytaradi."""
        total_p, total_r, n = 0.0, 0.0, 0
        for ref, hyp in zip(references, hypotheses):
            ref_toks = _tokenize(ref)
            hyp_toks = _tokenize(hyp)
            if not ref_toks or not hyp_toks:
                continue
            ref_cnt: dict = {}
            for t in ref_toks:
                ref_cnt[t] = ref_cnt.get(t, 0) + 1
            overlap = 0
            for t in hyp_toks:
                if ref_cnt.get(t, 0) > 0:
                    overlap += 1
                    ref_cnt[t] -= 1
            total_p += overlap / len(hyp_toks)
            total_r += overlap / len(ref_toks)
            n += 1

        p = total_p / n if n else 0.0
        r = total_r / n if n else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    def save(self, path: str) -> None:
        """Modelni saqlaydi."""
        state = {
            "d_model": self._d_model,
            "nhead": self._nhead,
            "src_t2i": self._src_t2i,
            "src_i2t": self._src_i2t,
            "tgt_t2i": self._tgt_t2i,
            "tgt_i2t": self._tgt_i2t,
            "torch_state": self._model.state_dict() if (HAS_TORCH and self._model) else None,
        }
        Path(path).write_bytes(pickle.dumps(state))

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        state = pickle.loads(Path(path).read_bytes())
        self._d_model = state["d_model"]
        self._nhead = state["nhead"]
        self._src_t2i = state["src_t2i"]
        self._src_i2t = state["src_i2t"]
        self._tgt_t2i = state["tgt_t2i"]
        self._tgt_i2t = state["tgt_i2t"]
        self._device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        if HAS_TORCH and state["torch_state"] is not None:
            self._model = _TinyTransformer(
                len(self._src_t2i), len(self._tgt_t2i),
                d_model=self._d_model, nhead=self._nhead,
            ).to(self._device)
            self._model.load_state_dict(state["torch_state"])
        else:
            self._model = None
