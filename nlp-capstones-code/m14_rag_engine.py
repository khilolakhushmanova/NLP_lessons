"""
m14_rag_engine.py — Kapstone modul: FAISS + Sentence-Transformers RAG tizimi.

Shartnoma:
    class RAGEngine:
        def index(texts, batch_size=32) -> None
        def answer(question, k=3) -> dict
        def save_index(path) -> None
        def load_index(path) -> None

Online rejim: paraphrase-multilingual-MiniLM-L12-v2 + FAISS IndexFlatIP.
Offline rejim: TF-IDF + cosine o'xshashlik.

d15_p14_rag_engine.ipynb dan olingan.
"""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    _EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _EMBED_DEVICE = "cpu"

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_USE_SEMANTIC = HAS_ST

_CHUNK_SIZE = 80
_CHUNK_OVERLAP = 15


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def _build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_text = "\n\n".join(
        f"[{i}] {item.get('title', 'hujjat')}\n{item['text']}"
        for i, item in enumerate(contexts, 1)
    )
    return (
        "Quyidagi kontekst asosida savolga qisqa javob bering.\n\n"
        f"Kontekst:\n{ctx_text}\n\n"
        f"Savol: {question}\n\nJavob:"
    )


class RAGEngine:
    """FAISS + sentence-transformers + LLM API asosida RAG qidirish tizimi.

    Corpus: uz_kb (yangiliklar + lex.uz, 10 000 chunk, FAISS indeksi).
    Consumed by: m15 (agent tool: retrieve_docs).
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._titles: list[str] = []
        self._embedder: Any = None
        self._chunk_embeddings: Any = None
        self._faiss_index: Any = None
        self._vectorizer: Any = None
        self._tfidf_matrix: Any = None

    def index(self, texts: list[str], batch_size: int = 32) -> None:
        """Hujjatlarni embedding qilib FAISS indeksiga qo'shadi.

        Args:
            texts:      Indekslanadigan matnlar ro'yxati.
            batch_size: Embedding batch hajmi.
        """
        if not texts:
            raise ValueError("texts bo'sh bo'lmasligi kerak.")

        self._chunks = []
        self._titles = []
        for idx, text in enumerate(texts):
            for chunk in _chunk_text(text):
                self._chunks.append(chunk)
                self._titles.append(f"hujjat_{idx + 1}")

        if _USE_SEMANTIC:
            self._embedder = SentenceTransformer(_MODEL_NAME, device=_EMBED_DEVICE)
            self._chunk_embeddings = self._embedder.encode(
                self._chunks,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            if HAS_FAISS:
                dim = self._chunk_embeddings.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_index.add(self._chunk_embeddings.astype("float32"))
        else:
            if not HAS_SKLEARN:
                raise RuntimeError("scikit-learn yoki sentence-transformers talab qilinadi.")
            self._vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            self._tfidf_matrix = self._vectorizer.fit_transform(self._chunks)

    def _retrieve(self, question: str, k: int = 3) -> list[dict]:
        """Ichki retrieve yordamchi — k ta eng o'xshash chunkni qaytaradi."""
        if self._faiss_index is not None:
            q_emb = self._embedder.encode([question], normalize_embeddings=True)
            scores, indices = self._faiss_index.search(q_emb.astype("float32"), k)
            return [
                {"score": float(s), "title": self._titles[i], "text": self._chunks[i]}
                for s, i in zip(scores[0], indices[0])
                if 0 <= i < len(self._chunks)
            ]

        if _USE_SEMANTIC and self._chunk_embeddings is not None:
            import numpy as np
            q_emb = self._embedder.encode([question], normalize_embeddings=True)[0]
            sims = self._chunk_embeddings @ q_emb
            top = sims.argsort()[-k:][::-1]
            return [
                {"score": float(sims[i]), "title": self._titles[i], "text": self._chunks[i]}
                for i in top
            ]

        q_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        top = sims.argsort()[-k:][::-1]
        return [
            {"score": float(sims[i]), "title": self._titles[i], "text": self._chunks[i]}
            for i in top
        ]

    def answer(self, question: str, k: int = 3) -> dict[str, Any]:
        """Savolga RAG pipeline orqali javob qaytaradi.

        Returns:
            {
              'answer':     str,
              'sources':    list[str],
              'confidence': float,
            }
        """
        if not self._chunks:
            raise RuntimeError("Avval index() ni chaqiring.")
        if not question:
            raise ValueError("Savol bo'sh bo'lmasligi kerak.")

        contexts = self._retrieve(question, k=k)
        answer_text = contexts[0]["text"][:420] if contexts else "Kontekst topilmadi."
        confidence = max(0.0, min(1.0, contexts[0]["score"])) if contexts else 0.0
        return {
            "answer": answer_text,
            "sources": [item["text"] for item in contexts],
            "confidence": round(confidence, 3),
        }

    def save_index(self, path: str) -> None:
        """FAISS indeksini .faiss faylga saqlaydi."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chunks": self._chunks,
            "titles": self._titles,
            "chunk_embeddings": self._chunk_embeddings,
            "tfidf_vectorizer": self._vectorizer,
            "tfidf_matrix": self._tfidf_matrix,
        }
        p_pkl = p.with_suffix(".pkl")
        p_pkl.write_bytes(pickle.dumps(state))

        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(p.with_suffix(".faiss")))

    def load_index(self, path: str) -> None:
        """Saqlangan indeksni yuklaydi."""
        p = Path(path)
        state = pickle.loads(p.with_suffix(".pkl").read_bytes())
        self._chunks = state["chunks"]
        self._titles = state["titles"]
        self._chunk_embeddings = state["chunk_embeddings"]
        self._vectorizer = state["tfidf_vectorizer"]
        self._tfidf_matrix = state["tfidf_matrix"]

        faiss_path = p.with_suffix(".faiss")
        if HAS_FAISS and faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
        else:
            self._faiss_index = None

        if _USE_SEMANTIC and self._chunk_embeddings is not None:
            self._embedder = SentenceTransformer(_MODEL_NAME, device=_EMBED_DEVICE)
