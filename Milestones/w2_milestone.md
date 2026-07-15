# 2-hafta milestone: Klassik pipeline integratsiyasi

**Sana:** 24-iyun 2026 (chorshanba) · **Topshirish muddati:** 29-iyun 2026 (dushanba, 09:30 gacha)
**Format:** asinxron, mustaqil · **Qamrov:** m01–m05 (TextPreprocessor, SentimentClassifier,
PretrainedEmbedder, SpellLSHRetriever, Autocomplete)

Chorshanba — **integratsiya kuni**. Birinchi haftada va shu hafta boshida siz beshta
kapstone modulni qurdingiz. Endi ularni yagona zanjirga ulaymiz va **o'z hujjat
yordamchingiz**ning klassik (neyron tarmoqdan oldingi) o'zagini ishga tushiramiz.

Bu milestone bitta **to'liq integratsiya** — m01–m05 hammasi tayyor, shuning uchun
ularni alohida-alohida emas, **birga** sinaymiz. Klassik pipeline yoyi quyidagicha:

```
       so'rov
         │
   m01 tozalash ──► m04 imlo tuzatish ──► m04 LSH qidiruv ──► top-k hujjat
         │
         ├──────► m02 TF-IDF + sentiment ──► ijobiy / salbiy
         │
         └──────► m05 autocomplete ──► keyingi so'z taklifi
                  (m03 embeddinglar — o'xshashlik/OOV nazorati)
```

---

## O'z korpusingiz

Kurs davomida siz **o'z hujjat to'plamingiz** ustida ishlaysiz. Bu milestone uchun
1-haftada tanlagan korpusingizdan (50–100 o'zbek matni) foydalaning. Korpusingiz hali
bo'lmasa, vaqtincha quyidagi tayyor fayllardan foydalaning:

- qidiruv/imlo uchun: `course/practices/d05_checkpoints/uz_news_corpus.txt`
- sentiment uchun: `course/practices/d03_checkpoints/uz_sentiment_mini.txt`
- embedding uchun: `course/practices/d04_checkpoints/uz_mini.vec`

---

## 1-topshiriq — m01 → m04: imloni tuzatib, hujjat qidiramiz

So'rovda imlo xatosi bo'lishi tabiiy. Avval uni tuzatamiz, keyin qidiramiz.

**Qadamlar:**

1. `SpellLSHRetriever` ni tayyorlang: `fit_dictionary(korpus)` (imlo lug'ati P(w))
   va `index_docs(korpus)` (LSH indeks).
2. Imlosi xato so'rov so'zini `correct()` bilan tuzating (Noisy Channel + Levenshtein).
3. Tuzatilgan so'rov bilan `retrieve_lsh(query, k)` orqali eng o'xshash **top-k**
   hujjatni oling.

**Kutilgan natija:** `correct("telifon")` lug'atdagi to'g'ri so'zga (`telefon`)
yaqinlashadi; `retrieve_lsh` so'rov so'zlarini o'z ichiga olgan hujjatlarni qaytaradi.
Eslatma: `m04` ichida `m01` (`TextPreprocessor`) ishlatiladi — integratsiya tabiiy.

---

## 2-topshiriq — m01 + m02: sentiment tahlili

**Qadamlar:**

1. Sentiment ma'lumotlarini yuklang; reytingni binarizatsiya qiling
   (`{4,5}→ijobiy`, `{1,2}→salbiy`, `3` tashlanadi).
2. `SentimentClassifier.fit(texts, labels)` — model ichida `m01` preprocessing va
   TF-IDF qiladi.
3. Yangi sharhlarni `predict(text)` bilan tasniflang; `predict_proba(text)`
   ehtimolliklarini ko'ring (yig'indisi 1).

**Kutilgan natija:** aniq ijobiy sharh → `ijobiy`, aniq salbiy sharh → `salbiy`.
Yorliqlar faqat **`ijobiy` / `salbiy`** (musbat/manfiy emas).

---

## 3-topshiriq — m05 Autocomplete: keyingi so'zni taklif qilamiz

**Qadamlar:**

1. Korpusni tokenlangan jumlalarga aylantiring va `Autocomplete.train(texts, n=2)`
   bilan bigram modelini o'qiting.
2. `complete("yangi", k=3)` — eng ehtimoliy keyingi 3 so'zni oling.
3. `perplexity("...")` bilan modelning matnga "hayron"lik darajasini o'lchang.

**Kutilgan natija:** `complete` lug'atdan mazmunli davom takliflarini qaytaradi;
`perplexity` chekli musbat son (past qiymat — yaxshiroq model).

---

## Qo'shimcha — m03: embedding sifatini nazorat qilamiz

`PretrainedEmbedder` ni `.vec` (yoki Kaggle'da `.kv`) faylidan yuklang va
`most_similar("toshkent")` hamda `oov_rate(tokenlar)` ni sinab ko'ring. Bu
keyingi hafta (neyron embeddinglar) uchun ko'prik: tayyor vektorlarning o'zbek
so'zlari uchun coverage darajasini his qilasiz.

---

## O'z-o'zini tekshirish

Modullaringiz shartnomaga (`capstone/contracts.py`) mos ekanini va uchala
integratsiya ishlayotganini quyidagi skript bilan tekshiring:

```bash
python course/milestones/w2_check.py
```

Skript m01–m05 ni va uchala pipeline'ni avtomatik sinaydi, har tekshiruv uchun
**✓** chiqaradi va oxirida "hammasi o'tdi" xabarini beradi. Biror tekshiruv
`FAIL` bersa — xabardagi ko'rsatmaga amal qilib, tegishli modulni tuzating.

---

## Yozma mulohaza (qisqa)

Topshiriq bilan birga **3–5 jumlalik** mulohaza yozing:

> Imlo tuzatish (Noisy Channel) qaysi so'zlarda yaxshi ishladi, qayerda
> adashdi — nega? Sentiment klassifikatori qaysi sharhlarni noto'g'ri tasnifladi?
> Autocomplete takliflari korpusingiz mavzusiga mosmi? Tayyor embeddinglarning
> OOV darajasi o'z korpusingizda qancha — bu nimani anglatadi?

---

## Topshirish

- **Muddat:** 29-iyun 2026, dushanba, 09:30 (3-hafta boshlanishigacha).
- **Nima topshiriladi:** (1) uchala integratsiya natijalari (tuzatilgan so'rov +
  topilgan hujjatlar, sentiment bashoratlari, autocomplete takliflari),
  (2) `w2_check.py` muvaffaqiyatli o'tgani (ekran tasviri yoki natija),
  (3) yozma mulohaza.
- **Qanday:** o'z kapstone repozitoriyingizga (`nlp-course-capstone`) joylang va
  havolani ulashing.

> **Eslatma:** bu milestone klassik pipeline (m01–m05) ga taalluqli. Neyron
> embeddinglar (m06, Word2Vec) keyingi kun (P6) da quriladi; rekurrent modellar
> (m07, m08) **3-hafta milestone (w3)** da integratsiya qilinadi.
