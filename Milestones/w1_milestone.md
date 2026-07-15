# 1-hafta milestone: TextPreprocessor integratsiyasi

**Sana:** 17-iyun 2026 (chorshanba) · **Topshirish muddati:** 22-iyun 2026 (dushanba, 09:30 gacha)
**Format:** asinxron, mustaqil · **Qamrov:** m01 `TextPreprocessor` (m02 — oldindan ko'rish)

Chorshanba — dam olish kuni emas, **integratsiya kuni**. Bu haftada siz birinchi
kapstone modulingiz — `TextPreprocessor` (m01) — ni qurdingiz. Endi uni **o'z
hujjatlaringizda** sinab ko'ramiz va keyingi modul (`SentimentClassifier`, m02) bilan
ulashga tayyorlanamiz.

Milestone ikki fazadan iborat, chunki m01 bugun (chorshanba) tayyor, m02 esa
payshanba (P2) da quriladi:

| Faza | Qachon | Nima qilamiz | Modul |
|------|--------|--------------|-------|
| **A** | Chorshanba (bugun) | m01 ni o'z korpusingizda ishlatib, statistika chiqaramiz | m01 (qamrovda) |
| **B** | Payshanba–dam olish | m01 → m02 pipeline ni sinaymiz | m02 (oldindan ko'rish) |

---

## Faza A — m01 ni o'z korpusingizda ishlatamiz (chorshanba)

Bu kurs davomida siz **o'z hujjat to'plamingiz** ustida ishlaysiz — bu sizning
shaxsiy kapstone loyihangiz. Bugun shu korpusni tanlaymiz.

**Qadamlar:**

1. **O'z korpusingizni tanlang** — 50–100 ta qisqa o'zbek matni. Masalan:
   yangiliklar maqolalari (kun.uz, gazeta.uz), o'z hujjatlaringiz, yoki
   Wikipedia paragraflari. Har bir matn alohida qatorda bo'lsin.
   *(Korpusingiz bo'lmasa, vaqtincha `course/practices/d02_checkpoints/uz_news_mini.txt`
   dan foydalaning.)*

2. **m01 `TextPreprocessor` ni qo'llang** — har bir matnni `preprocess()` orqali
   tozalang (tokenizatsiya → kichik harf → stop-so'z filtri → stemming).

3. **Quyidagi statistikani chiqaring:**
   - **lug'at hajmi** — tozalangandan keyingi unikal so'zlar soni;
   - **token/stopword nisbati** — xom tokenlar soniga nisbatan tozalangan tokenlar ulushi;
   - **eng ko'p uchraydigan 20 so'z** (`collections.Counter`).

**Kutilgan natija:** lug'at hajmi xom matndagi so'zlar sonidan **kichik** bo'lishi
kerak (stop-so'z filtri va stemming tufayli), va eng ko'p uchraydigan so'zlar
korpusingiz mavzusini aks ettirishi lozim.

---

## Faza B — m01 → m02 pipeline (payshanbadan keyin)

Payshanba kuni (P2 amaliyoti) siz `SentimentClassifier` (m02) ni qurasiz. U
**ichida m01 dan foydalanadi**: matn → m01 preprocessing → TF-IDF → bashorat.

**Qadamlar:**

1. m02 ni hissiyot ma'lumotlarida o'qiting (`fit(texts, labels)`); yorliqlar
   **`ijobiy`** yoki **`salbiy`**.
2. Bir nechta yangi sharhni tasniflang: `predict(text)` → `ijobiy` / `salbiy`.
3. `predict_proba(text)` ehtimolliklarini ko'ring (yig'indisi 1 ga teng).

**Kutilgan natija:** aniq ijobiy sharh → `ijobiy`, aniq salbiy sharh → `salbiy`.
Shu bilan **m01 → m02 pipeline** to'liq ishlayotganini ko'rasiz — bu sizning
hujjat yordamchingizning birinchi tugallangan bo'g'ini.

---

## O'z-o'zini tekshirish

Modullaringiz shartnomaga (`capstone/contracts.py`) mos ekanini va pipeline
ishlayotganini quyidagi skript bilan tekshiring:

```bash
python course/milestones/w1_check.py
```

Skript m01 va m02 ni hamda m01 → m02 pipeline ni avtomatik sinaydi va har bir
tekshiruv uchun **✓** belgisini chiqaradi. Oxirida "hammasi o'tdi" xabari
ko'rinishi kerak. Agar biror tekshiruv `FAIL` bersa — xabardagi ko'rsatmaga
amal qilib, tegishli modulni tuzating.

---

## Yozma mulohaza (qisqa)

Topshiriq bilan birga **3–5 jumlalik** mulohaza yozing:

> O'z korpusingizda stemming va stop-so'z filtri lug'at hajmini qanchaga
> kichraytirdi? Eng ko'p uchraydigan so'zlar korpusingiz mavzusiga mosmi?
> m01 ning qaysi qarori (masalan, apostrof yoki qo'shimchani qirqish) sizning
> matningizda kutilmagan natija berdi?

---

## Topshirish

- **Muddat:** 22-iyun 2026, dushanba, 09:30 (2-hafta boshlanishigacha).
- **Nima topshiriladi:** (1) Faza A statistikangiz (lug'at hajmi, nisbat, top-20),
  (2) `w1_check.py` muvaffaqiyatli o'tgani (ekran tasviri yoki natija),
  (3) yozma mulohaza.
- **Qanday:** o'z kapstone repozitoriyingizga (`nlp-course-capstone`) joylang va
  havolani ulashing.

> **Eslatma:** bu milestone faqat m01 (va m02 oldindan ko'rish) ga taalluqli.
> Embedding (m03) va imlo-tuzatish/qidiruv (m04) modullari **keyingi hafta
> milestone (w2)** da birlashtiriladi.
