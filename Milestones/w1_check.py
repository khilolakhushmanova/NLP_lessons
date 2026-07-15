"""
course/milestones/w1_check.py
1-hafta milestone o'z-o'zini tekshiruvchi skript.

m01 (TextPreprocessor) va m02 (SentimentClassifier) ning capstone/contracts.py
imzolariga mosligini hamda m01 -> m02 pipeline ishlashini tekshiradi.

Ishga tushirish (repo ildizidan yoki istalgan joydan):
    python course/milestones/w1_check.py

Eslatma: gensim / datasketch SHART EMAS. Faqat numpy + scikit-learn kerak.
Qamrov: m01 (Faza A) + m02 (Faza B). m03/m04 bu yerda tekshirilmaydi (ular w2).
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# ─── yo'llar ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # course/milestones
REPO = HERE.parent.parent                        # repo ildizi
MODULES = REPO / "capstone" / "modules"
sys.path.insert(0, str(MODULES))

P1_DATA = REPO / "course" / "practices" / "d02_checkpoints" / "uz_news_mini.txt"
P2_DATA = REPO / "course" / "practices" / "d03_checkpoints" / "uz_sentiment_mini.txt"

_passed = 0


def check(nom: str, shart: bool) -> None:
    """Bitta tekshiruv: shart noto'g'ri bo'lsa, aniq xabar bilan to'xtaydi."""
    global _passed
    assert shart, f"FAIL: {nom}"
    print(f"  ✓ {nom}")
    _passed += 1


print("=" * 64)
print("1-HAFTA MILESTONE — O'Z-O'ZINI TEKSHIRUV (w1_check)")
print("=" * 64)

# ══════════════════════════════════════════════════════════════════════════
# FAZA A — m01 TextPreprocessor
# ══════════════════════════════════════════════════════════════════════════
print("\n[Faza A] m01 TextPreprocessor — shartnoma mosligi va korpus statistikasi")

from m01_text_preprocessor import TextPreprocessor

tp = TextPreprocessor()

# Shartnoma metodlari mavjudligi
check("m01.preprocess mavjud", callable(getattr(tp, "preprocess", None)))
check("m01.preprocess_batch mavjud", callable(getattr(tp, "preprocess_batch", None)))
check("m01.fit_stopwords mavjud", callable(getattr(tp, "fit_stopwords", None)))

# preprocess xulqi
toks = tp.preprocess("O'zbekistonda NLP texnologiyalari rivojlanmoqda")
check("preprocess() bo'sh bo'lmagan list qaytaradi", isinstance(toks, list) and len(toks) > 0)
check("tokenlar kichik harfda", all(t == t.lower() for t in toks))

# bo'sh matnda ValueError (shartnoma talabi)
_raised = False
try:
    tp.preprocess("")
except ValueError:
    _raised = True
check("bo'sh matnda ValueError ko'tariladi", _raised)

# preprocess_batch
batch = tp.preprocess_batch(["Salom dunyo", "NLP juda qiziq"])
check("preprocess_batch() ro'yxatlar ro'yxatini qaytaradi",
      isinstance(batch, list) and len(batch) == 2 and all(isinstance(d, list) for d in batch))

# fit_stopwords ishlaydi (xato bermaydi)
tp.fit_stopwords(["bu bir misol", "bu yana bir misol"], max_df=0.9)
check("fit_stopwords() xatosiz ishlaydi", True)

# Faza A topshirig'i namunasi: korpus statistikasi
assert P1_DATA.exists(), f"Korpus topilmadi: {P1_DATA}"
corpus = [l.strip() for l in P1_DATA.open(encoding="utf-8") if l.strip()]
processed = TextPreprocessor().preprocess_batch(corpus)   # toza nusxa (stopword fit qilinmagan)
vocab = set(w for doc in processed for w in doc)
n_clean = sum(len(d) for d in processed)
n_raw = sum(len(c.split()) for c in corpus)
freq = Counter(w for d in processed for w in d)

check("korpus to'liq qayta ishlandi", len(processed) == len(corpus) and n_clean > 0)
check("lug'at xom matndan kichik (stopword + stemming ta'siri)", len(vocab) < n_raw)

ratio = n_clean / n_raw if n_raw else 0.0
print(f"    Korpus: {len(corpus)} hujjat | lug'at hajmi: {len(vocab)} so'z")
print(f"    Tokenlar: xom {n_raw} -> tozalangan {n_clean} (nisbat {ratio:.2f})")
print(f"    Eng ko'p uchraydigan 5 so'z: {[w for w, _ in freq.most_common(5)]}")

# ══════════════════════════════════════════════════════════════════════════
# FAZA B — m01 -> m02 pipeline (SentimentClassifier)
# ══════════════════════════════════════════════════════════════════════════
print("\n[Faza B] m01 -> m02 pipeline — SentimentClassifier")

from m02_sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()
for metod in ["fit", "predict", "predict_proba", "save", "load"]:
    check(f"m02.{metod} mavjud", callable(getattr(clf, metod, None)))

# Ma'lumotni yuklash va binarizatsiya (rating -> ijobiy/salbiy)
assert P2_DATA.exists(), f"Sentiment data topilmadi: {P2_DATA}"


def binarize(rating: str):
    r = int(rating)
    if r >= 4:
        return "ijobiy"
    if r <= 2:
        return "salbiy"
    return None   # neytral (3) tashlanadi


texts, labels = [], []
for line in P2_DATA.open(encoding="utf-8"):
    if not line.strip():
        continue
    rating, text = line.rstrip("\n").split("\t", 1)
    lab = binarize(rating)
    if lab is not None:
        texts.append(text)
        labels.append(lab)

check("binarizatsiya faqat ijobiy/salbiy beradi", set(labels) == {"ijobiy", "salbiy"})

# m01 -> m02: m02 ichida TextPreprocessor (m01) preprocessing qiladi
clf.fit(texts, labels)
p_pos = clf.predict("Mahsulot juda sifatli, tez yetkazib berishdi, rahmat!")
p_neg = clf.predict("Sifatsiz narsa keldi, sindi, pulim behuda ketdi, afsus.")
check("ijobiy sharh 'ijobiy' deb tasniflandi", p_pos == "ijobiy")
check("salbiy sharh 'salbiy' deb tasniflandi", p_neg == "salbiy")

pr = clf.predict_proba("Mahsulot zo'r va sifatli")
check("predict_proba kalitlari {ijobiy, salbiy}", set(pr.keys()) == {"ijobiy", "salbiy"})
check("predict_proba ehtimolliklari yig'indisi 1", abs(sum(pr.values()) - 1.0) < 1e-6)

# save / load roundtrip
_tmp = os.path.join(tempfile.gettempdir(), "w1_m02.pkl")
clf.save(_tmp)
clf2 = SentimentClassifier()
clf2.load(_tmp)
_t = "Mahsulot juda sifatli, rahmat!"
check("save/load dan keyin bashorat mos keladi", clf2.predict(_t) == clf.predict(_t))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"NATIJA: {_passed} tekshiruvning hammasi O'TDI ✓")
print("1-hafta milestone tayyor: m01 (Faza A) + m01 -> m02 (Faza B).")
print("m03/m04 keyingi (w2) milestone da integratsiya qilinadi.")
print("=" * 64)
