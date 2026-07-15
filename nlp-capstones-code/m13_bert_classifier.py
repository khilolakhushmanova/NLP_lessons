"""
capstone/modules/m13_bert_classifier.py

FineTunedClassifier wraps multilingual DistilBERT for Uzbek binary sentiment.
It uses Hugging Face Trainer for fine-tuning and the standard
save_pretrained()/from_pretrained() artifact format.
"""
from __future__ import annotations

import json
import time
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _macro_f1(y_true, y_pred) -> float:
    scores = []
    for label in sorted(set(y_true) | set(y_pred)):
        true_positive = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        false_positive = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        false_negative = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def _inference_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "FineTunedClassifier uchun torch va transformers kerak. "
            "`pip install torch transformers` ni bajaring."
        ) from error
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _training_dependencies():
    torch, model_class, tokenizer_class = _inference_dependencies()
    try:
        import accelerate  # noqa: F401
        from datasets import Dataset
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as error:
        raise ImportError(
            "Fine-tuning uchun datasets va accelerate ham kerak. "
            "`pip install datasets accelerate` ni bajaring."
        ) from error
    return (
        torch,
        model_class,
        tokenizer_class,
        Dataset,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )


class FineTunedClassifier:
    """Fine-tuned multilingual DistilBERT sentiment classifier."""

    DEFAULT_MODEL = "distilbert-base-multilingual-cased"
    LABELS = {0: "salbiy", 1: "ijobiy"}
    LABEL_TO_ID = {"salbiy": 0, "ijobiy": 1}

    def __init__(self, max_length: int = 128, device: str | None = None) -> None:
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length musbat butun son bo'lishi kerak.")
        self.max_length = max_length
        self.device = device
        self.model_name = self.DEFAULT_MODEL
        self.model = None
        self.tokenizer = None
        self.fitted = False

    @classmethod
    def _normalize_labels(cls, labels) -> list[int]:
        normalized = []
        for label in labels:
            if isinstance(label, Integral) and not isinstance(label, bool):
                label_id = int(label)
                if label_id not in cls.LABELS:
                    raise ValueError("Raqamli label faqat 0 yoki 1 bo'lishi mumkin.")
                normalized.append(label_id)
            elif isinstance(label, str) and label in cls.LABEL_TO_ID:
                normalized.append(cls.LABEL_TO_ID[label])
            else:
                raise ValueError("Label 'salbiy', 'ijobiy', 0 yoki 1 bo'lishi kerak.")
        return normalized

    @staticmethod
    def _validate_texts(texts) -> None:
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("texts bo'sh bo'lmagan ro'yxat bo'lishi kerak.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Har bir text bo'sh bo'lmagan string bo'lishi kerak.")

    @staticmethod
    def _validate_hyperparameters(epochs, batch_size, lr) -> None:
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs musbat butun son bo'lishi kerak.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size musbat butun son bo'lishi kerak.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr musbat son bo'lishi kerak.")

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        model_name: str = DEFAULT_MODEL,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> None:
        self._validate_texts(texts)
        if not isinstance(labels, (list, tuple)) or len(texts) != len(labels):
            raise ValueError("texts va labels uzunligi teng bo'lishi kerak.")
        numeric_labels = self._normalize_labels(labels)
        if len(set(numeric_labels)) < 2:
            raise ValueError("Tasniflash uchun kamida ikkita sinf kerak.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name bo'sh bo'lmagan string bo'lishi kerak.")
        self._validate_hyperparameters(epochs, batch_size, lr)

        dependencies = _training_dependencies()
        _, model_class, tokenizer_class, Dataset, Collator, Trainer, Arguments = dependencies
        self.model_name = model_name
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(
            model_name,
            num_labels=2,
            id2label=self.LABELS,
            label2id=self.LABEL_TO_ID,
        )

        dataset = Dataset.from_dict({"text": list(texts), "labels": numeric_labels})
        tokenized = dataset.map(
            lambda batch: self.tokenizer(
                batch["text"], truncation=True, max_length=self.max_length
            ),
            batched=True,
            remove_columns=["text"],
        )
        with TemporaryDirectory(prefix="m13_trainer_") as output_dir:
            arguments = Arguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                seed=42,
                data_seed=42,
            )
            trainer = Trainer(
                model=self.model,
                args=arguments,
                train_dataset=tokenized,
                data_collator=Collator(self.tokenizer),
            )
            trainer.train()
            self.model = trainer.model
        self.model.eval()
        self.fitted = True

    def _require_fitted(self) -> None:
        if not self.fitted or self.model is None or self.tokenizer is None:
            raise RuntimeError("Avval fit() yoki load() ni chaqiring.")

    def predict_proba(self, text: str) -> dict[str, float]:
        self._require_fitted()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text bo'sh bo'lmagan string bo'lishi kerak.")
        torch, _, _ = _inference_dependencies()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits[0], dim=-1).tolist()
        return {self.LABELS[index]: float(value) for index, value in enumerate(probabilities)}

    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)

    def evaluate(self, texts: list[str], labels: list[str]) -> dict[str, float]:
        self._validate_texts(texts)
        if not isinstance(labels, (list, tuple)) or len(texts) != len(labels):
            raise ValueError("texts va labels uzunligi teng bo'lishi kerak.")
        numeric_labels = self._normalize_labels(labels)
        started = time.perf_counter()
        rows = [self.predict_proba(text) for text in texts]
        inference_ms = (time.perf_counter() - started) * 1000 / len(texts)
        predictions = [self.LABEL_TO_ID[max(row, key=row.get)] for row in rows]
        validation_loss = -float(
            np.mean(
                [
                    np.log(max(row[self.LABELS[label]], 1e-12))
                    for row, label in zip(rows, numeric_labels)
                ]
            )
        )
        return {
            "f1": round(_macro_f1(numeric_labels, predictions), 4),
            "accuracy": round(float(np.mean(np.array(predictions) == numeric_labels)), 4),
            "val_loss": round(validation_loss, 4),
            "inference_time": round(inference_ms, 4),
        }

    def save(self, path: str) -> None:
        self._require_fitted()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        metadata = {
            "format": "huggingface-pretrained-v1",
            "base_model": self.model_name,
            "max_length": self.max_length,
            "labels": self.LABELS,
        }
        (output_dir / "capstone_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )

    def load(self, path: str) -> None:
        torch, model_class, tokenizer_class = _inference_dependencies()
        self.tokenizer = tokenizer_class.from_pretrained(path)
        self.model = model_class.from_pretrained(path)
        if self.model.config.num_labels != 2:
            raise ValueError("FineTunedClassifier artefaktida num_labels=2 bo'lishi kerak.")
        self.model.config.id2label = self.LABELS
        self.model.config.label2id = self.LABEL_TO_ID
        metadata_path = Path(path) / "capstone_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            self.model_name = metadata.get("base_model", str(path))
            self.max_length = int(metadata.get("max_length", self.max_length))
        else:
            self.model_name = str(path)
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(target_device)
        self.model.eval()
        self.fitted = True

    @property
    def parameter_count(self) -> int:
        self._require_fitted()
        return sum(parameter.numel() for parameter in self.model.parameters())
