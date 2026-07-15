"""
course/milestones/w2_check.py
2-hafta milestone o'z-o'zini tekshiruvchi skript.

Klassik pipeline integratsiyasi: m01–m05 modullarining capstone/contracts.py
imzolariga mosligini va uchta integratsiya zanjirini tekshiradi:
    1) m01 -> m04: imloni tuzatish (correct) -> LSH qidiruv (retrieve_lsh) -> top-k hujjat
    2) m01 + m02: preprocessing -> TF-IDF -> predict (sentiment, ijobiy/salbiy)
    3) m05 Autocomplete: keyingi so'z taklifi (complete)
Qo'shimcha: m03 (PretrainedEmbedder) funksional tekshiruvi (embed/most_similar/oov_rate).

Ishga tushirish (repo ildizidan yoki istalgan joydan):
    python course/milestones/w2_check.py

Eslatma: gensim / datasketch / nltk SHART EMAS. Faqat numpy + scikit-learn kerak.
Qamrov: m01, m02, m03, m04, m05. m05b (pedagogik) va m06 (Day 7) TEKSHIRILMAYDI.
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

NEWS_DATA = REPO / "course" / "practices" / "d02_checkpoints" / "uz_news_mini.txt"
SENT_DATA = REPO / "course" / "practices" / "d03_checkpoints" / "uz_sentiment_mini.txt"
VEC_DATA  = REPO / "course" / "practices" / "d04_checkpoints" / "uz_mini.vec"
LM_DATA   = REPO / "course" / "practices" / "d05_checkpoints" / "uz_news_corpus.txt"

_passed = 0


def check(nom: str, shart: bool) -> None:
    """Bitta tekshiruv: shart noto'g'ri bo'lsa, aniq xabar bilan to'xtaydi."""
    global _passed
    assert shart, f"FAIL: {nom}"
    print(f"  ✓ {nom}")
    _passed += 1


print("=" * 68)
print("2-HAFTA MILESTONE — O'Z-O'ZINI TEKSHIRUV (w2_check)")
print("Klassik pipeline integratsiyasi: m01–m05")
print("=" * 68)

# ══════════════════════════════════════════════════════════════════════════
# 0-QISM — shartnoma mosligi (capstone/contracts.py): m01–m05
# ══════════════════════════════════════════════════════════════════════════
print("\n[0-qism] Shartnoma mosligi — m01, m02, m03, m04, m05")

from m01_text_preprocessor import TextPreprocessor
from m02_sentiment_classifier import SentimentClassifier
from m03_pretrained_embedder import PretrainedEmbedder
from m04_spell_lsh_retriever import SpellLSHRetriever
from m05_autocomplete import Autocomplete

tp  = TextPreprocessor()
clf = SentimentClassifier()
emb = PretrainedEmbedder()
ret = SpellLSHRetriever()
acp = Autocomplete()

for metod in ["preprocess", "preprocess_batch", "fit_stopwords"]:
    check(f"m01.{metod} mavjud", callable(getattr(tp, metod, None)))
for metod in ["fit", "predict", "predict_proba", "save", "load"]:
    check(f"m02.{metod} mavjud", callable(getattr(clf, metod, None)))
for metod in ["load", "embed", "most_similar", "oov_rate"]:
    check(f"m03.{metod} mavjud", callable(getattr(emb, metod, None)))
for metod in ["correct", "edit_distance", "index_docs", "retrieve_lsh", "save", "load"]:
    check(f"m04.{metod} mavjud", callable(getattr(ret, metod, None)))
for metod in ["train", "complete", "perplexity"]:
    check(f"m05.{metod} mavjud", callable(getattr(acp, metod, None)))

# ══════════════════════════════════════════════════════════════════════════
# 1-INTEGRATSIYA — m01 -> m04: imlo tuzatish -> LSH qidiruv -> top-k hujjat
# ══════════════════════════════════════════════════════════════════════════
print("\n[1-integratsiya] m01 -> m04 — imlo tuzatish + LSH qidiruv (SpellLSHRetriever)")

assert LM_DATA.exists(), f"Korpus topilmadi: {LM_DATA}"
corpus = [l.strip() for l in LM_DATA.open(encoding="utf-8") if l.strip()]

# edit_distance — L4 [I]-slayd bilan solishtiring (qulflangan qiymatlar)
check("edit_distance(\"qo'l\",\"ko'l\")==1  # Ma'ruza L4 [I]-slayd", ret.edit_distance("qo'l", "ko'l") == 1)
check("edit_distance('dastur','dastir')==1  # Ma'ruza L4 [I]-slayd", ret.edit_distance("dastur", "dastir") == 1)

# m04 ichida m01 (TextPreprocessor) ishlatiladi -> m01->m04 uzviyligi tabiiy
ret.fit_dictionary(corpus)     # imlo lug'ati P(w) — correct() uchun zarur
ret.index_docs(corpus)         # LSH indeks — retrieve_lsh() uchun

# lug'atdagi eng chastotali stemni topamiz (m01 normalizatsiyasidan keyin)
freq = Counter(tok for doc in tp.preprocess_batch(corpus) for tok in doc)
top_word = next(w for w, _ in freq.most_common() if len(w) >= 4)

# toza lug'at so'zi o'zini qaytaradi (noisy channel: w in dict -> w)
check(f"correct('{top_word}')=='{top_word}' (lug'atda bor)", ret.correct(top_word) == top_word)

# bir belgi buzilgan so'z eng ehtimoliy to'g'ri variantga tuzaladi
typo = top_word[:-1] + ("x" if top_word[-1] != "x" else "z")   # tahrir masofasi 1
fixed = ret.correct(typo)
check(f"correct('{typo}') -> lug'atdagi so'z (masofa<=2)",
      isinstance(fixed, str) and fixed in freq and ret.edit_distance(fixed, typo) <= 2)

# imlosi xato so'rovni tuzatib, LSH bilan top-k hujjat qidiramiz
noisy_query = "yangi telifon"                                   # 'telifon' — imlo xatosi
corrected = " ".join(ret.correct(w) for w in noisy_query.split())
docs = ret.retrieve_lsh(corrected, k=3)
check("retrieve_lsh() bo'sh bo'lmagan str ro'yxat qaytaradi (top-k)",
      isinstance(docs, list) and 0 < len(docs) <= 3 and all(isinstance(d, str) for d in docs))
print(f"    Buzuq so'rov: '{noisy_query}' -> tuzatilgan: '{corrected}'")
print(f"    Topilgan hujjatlar (top-{len(docs)}): {docs[:2]}{' …' if len(docs) > 2 else ''}")

# save/load round-trip
_tmp_ret = os.path.join(tempfile.gettempdir(), "w2_m04.pkl")
ret.save(_tmp_ret)
ret2 = SpellLSHRetriever(); ret2.load(_tmp_ret)
check("m04 save/load dan keyin qidiruv mos keladi",
      ret2.retrieve_lsh(corrected, k=3) == docs)

# ══════════════════════════════════════════════════════════════════════════
# 2-INTEGRATSIYA — m01 + m02: preprocessing -> TF-IDF -> predict (sentiment)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2-integratsiya] m01 + m02 — sentiment pipeline (SentimentClassifier)")

assert SENT_DATA.exists(), f"Sentiment data topilmadi: {SENT_DATA}"


def binarize(rating: str):
    r = int(rating)
    if r >= 4:
        return "ijobiy"
    if r <= 2:
        return "salbiy"
    return None   # neytral (3) tashlanadi


texts, labels = [], []
for line in SENT_DATA.open(encoding="utf-8"):
    if not line.strip():
        continue
    rating, text = line.rstrip("\n").split("\t", 1)
    lab = binarize(rating)
    if lab is not None:
        texts.append(text)
        labels.append(lab)

check("binarizatsiya faqat ijobiy/salbiy beradi", set(labels) == {"ijobiy", "salbiy"})

# m02 ichida m01 (TextPreprocessor) preprocessing + TF-IDF qiladi
clf.fit(texts, labels)
p_pos = clf.predict("Mahsulot juda sifatli, tez yetkazib berishdi, rahmat!")
p_neg = clf.predict("Sifatsiz narsa keldi, sindi, pulim behuda ketdi, afsus.")
check("ijobiy sharh 'ijobiy' deb tasniflandi", p_pos == "ijobiy")
check("salbiy sharh 'salbiy' deb tasniflandi", p_neg == "salbiy")

pr = clf.predict_proba("Mahsulot zo'r va sifatli")
check("predict_proba kalitlari {ijobiy, salbiy}", set(pr.keys()) == {"ijobiy", "salbiy"})
check("predict_proba ehtimolliklari yig'indisi 1", abs(sum(pr.values()) - 1.0) < 1e-6)

# ══════════════════════════════════════════════════════════════════════════
# 3-INTEGRATSIYA — m05 Autocomplete: keyingi so'z taklifi
# ══════════════════════════════════════════════════════════════════════════
print("\n[3-integratsiya] m05 Autocomplete — keyingi so'z bashorati")

# korpusni tokenlangan jumlalarga aylantiramiz (m05.train: list[list[str]])
tokenized = [line.split() for line in corpus]
acp.train(tokenized, n=2)

sugg = acp.complete("yangi", k=3)
check("complete() bo'sh bo'lmagan str ro'yxat qaytaradi (k<=3)",
      isinstance(sugg, list) and 0 < len(sugg) <= 3 and all(isinstance(s, str) for s in sugg))
check("complete() takliflari lug'atdan", all(s in acp._vocab for s in sugg))

ppl = acp.perplexity("yangi telefon chiqdi")
check("perplexity() chekli musbat son qaytaradi", isinstance(ppl, float) and 0.0 < ppl < float("inf"))
print(f"    complete('yangi', 3) = {sugg}")
print(f"    perplexity('yangi telefon chiqdi') = {ppl:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# m03 — PretrainedEmbedder funksional tekshiruvi (qamrovda: modules_covered[3])
# ══════════════════════════════════════════════════════════════════════════
print("\n[m03] PretrainedEmbedder — embed / most_similar / oov_rate (offline .vec)")

assert VEC_DATA.exists(), f"Embedding data topilmadi: {VEC_DATA}"
emb.load(str(VEC_DATA))   # offline word2vec matn formati (.vec), gensimsiz

v = emb.embed("toshkent")
check("embed() to'g'ri o'lchamli vektor qaytaradi (50-dim)",
      hasattr(v, "shape") and v.shape == (50,))
check("OOV so'z uchun nol-vektor", float((emb.embed("zzqwerty") ** 2).sum()) == 0.0)

sim = emb.most_similar("toshkent", n=5)
check("most_similar() (so'z, o'xshashlik) juftlar ro'yxati",
      isinstance(sim, list) and 0 < len(sim) <= 5
      and all(isinstance(w, str) and -1.000001 <= s <= 1.000001 for w, s in sim))
check("most_similar() o'xshashlik bo'yicha kamayuvchi tartibda",
      all(sim[i][1] >= sim[i + 1][1] for i in range(len(sim) - 1)))

oov = emb.oov_rate([["toshkent", "uzbekiston", "zzqwerty", "qwxyz"]])
check("oov_rate() [0,1] oralig'ida (2/4 = 0.5)", abs(oov - 0.5) < 1e-9)
print(f"    most_similar('toshkent', 5): {[w for w, _ in sim]}")
print(f"    oov_rate (2 OOV / 4 token) = {oov:.2f}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print(f"NATIJA: {_passed} tekshiruvning hammasi O'TDI ✓")
print("2-hafta milestone tayyor: m01–m05 klassik pipeline integratsiyasi.")
print("m05b (pedagogik) va m06 (Day 7) bu milestone qamrovida emas.")
print("=" * 68)
