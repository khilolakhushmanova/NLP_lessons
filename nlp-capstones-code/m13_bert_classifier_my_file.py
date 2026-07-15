class FineTunedClassifier:
    LABELS = {0: 'salbiy', 1: 'ijobiy'}

    def __init__(self, model_name='distilbert-base-multilingual-cased'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.baseline = None
        self.fitted = False

    def fit(self, texts, labels, epochs=3, batch_size=16):
        # SIZNING KODINGIZ:
        # Online rejimda AutoTokenizer + AutoModelForSequenceClassification + Trainer ishlating.
        # Offline rejimda train_baseline_classifier(texts, labels) dan foydalaning.
        # 1. Offline rejim
        if OFFLINE_FALLBACK or not HAS_TRANSFORMERS:
            self.baseline = train_baseline_classifier(texts, labels)
            self.fitted = True
            return self
    
        # 2. Kerakli kutubxonalarni import qilish
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
    
        # 3. Tokenizer yaratish
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
        # 4. Dataset yaratish
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
    
        # 5. Train/Test ga ajratish
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split["train"]
        test_ds = split["test"]
    
        # 6. Tokenize funksiyasi
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=128
            )
    
        train_ds = train_ds.map(tokenize, batched=True)
        test_ds = test_ds.map(tokenize, batched=True)
    
        # 7. Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)
    
        # 8. Model yaratish
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
    
        # 9. TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none"
        )
    
        # 10. Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
    
        # 11. O'qitish
        trainer.train()
    
        self.fitted = True
        return self


    def predict_proba(self, text):
        if self.baseline is not None:
            probs = self.baseline.predict_proba([text])[0]
            return {
                "salbiy": float(probs[0]),
                "ijobiy": float(probs[1])
            }

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Avval fit() metodini chaqiring.")
    
        self.model.eval()
    
        inputs = self.tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
    
        # Model qaysi qurilmada (CPU/GPU) bo'lsa, inputni ham o'sha yerga o'tkazish
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    
        return {
            "salbiy": float(probs[0]),
            "ijobiy": float(probs[1]),
        }
    
    def predict(self, text):
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)

    @staticmethod
    def bce_loss(logit, label=1):
        # SIZNING KODINGIZ:
        # bce_loss_value(logit, label) bilan bir xil qiymat qaytaring.
        probability = 1 / (1 + math.exp(-logit))
        return -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
        # return bce_loss_value(logit, label)

_check = FineTunedClassifier()
assert hasattr(_check, 'fit')
assert hasattr(_check, 'predict')
assert hasattr(_check, 'predict_proba')
assert abs(FineTunedClassifier.bce_loss(2.0) - 0.127) < 2e-3
print('Capstone klass scaffold tayyor.  [OK]')
