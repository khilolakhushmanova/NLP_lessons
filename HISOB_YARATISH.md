
\# Hisob Yaratish Yo'riqnomasi

**Kurs boshlanishidan OLDIN** (15-iyun 2026, 09:30 gacha) quyidagi
uchta hisobni oching va tekshiring. Bu yo'riqnomani birinchi amaliyot
notebookini (`d01_orientatsiya.ipynb`) ishga tushirishdan oldin bajaring.

---

## 1. Kaggle Hisob

### Yangi hisob ochish
1. [kaggle.com](https://www.kaggle.com) ga o'ting
2. **Register** → email va parol bilan ro'yxatdan o'ting
3. Email tasdiqlash xatini oching va havolani bosing

### Telefon verifikatsiyasi (GPU uchun majburiy)
> GPU kvotasi faqat telefon tasdiqlangan hisobda ishlaydi.

1. Kaggledagi profilingizga o'ting: **Account → Settings**
2. **Phone Verification** bo'limini toping
3. **Add phone number** → raqamingizni kiriting → SMS kod
4. Tasdiqlash tugagandan keyin **"Verified"** belgisi paydo bo'ladi

### Kaggle API kaliti (dataset yuklash uchun)
1. **Account → Settings → API**
2. **Create New Token** → `kaggle.json` yuklab olinadi
3. `kaggle.json` faylini Kaggle Secrets ga qo'shish (ixtiyoriy, ammo qulay):
   - Notebook ichida: **Add-ons → Secrets → Add Secret**
   - Name: `KAGGLE_KEY`, Value: `kaggle.json` ichidagi `"key"` qiymati

### Tekshiruv
- [ ] Kaggle profilingizga kira olasiz
- [ ] **"Phone Verified"** belgisi ko'rinadi
- [ ] Yangi notebook yaratib, kod katakda `print("ishladi")` ishlaydi

---

## 2. GitHub Hisob

### Yangi hisob ochish
1. [github.com](https://github.com) ga o'ting
2. **Sign up** → email, username, parol
3. Email tasdiqlash

### Kapstone repo yaratish
1. Kirganingizdan keyin **"+"** → **New repository**
2. Repository name: `nlp-course-capstone`
3. **Public** yoki **Private** (ixtiyoriy)
4. **Initialize with README** belgisini qo'ying
5. **Create repository**

### SSH kalit (ixtiyoriy, ammo qulay)
Agar har safar parol kiritishni xohlamasangiz:
```bash
ssh-keygen -t ed25519 -C "sizning@email.com"
# Enter, Enter, Enter (parolsiz)
cat ~/.ssh/id_ed25519.pub
# Chiqgan matnni GitHub → Settings → SSH Keys → New SSH Key ga joylashtiring
```

### Tekshiruv
- [ ] GitHub profilingizga kira olasiz
- [ ] `nlp-course-capstone` repo yaratilgan
- [ ] Repo URL: `https://github.com/<username>/nlp-course-capstone`

---

## 3. Hugging Face Hisob

### Yangi hisob ochish
1. [huggingface.co](https://huggingface.co) ga o'ting
2. **Sign Up** → email va parol
3. Email tasdiqlash

### API token olish (modellar va datasetlar uchun)
1. Kirganingizdan keyin: **Profile → Settings → Access Tokens**
2. **New token** → Name: `nlp-course`, Role: **Read**
3. **Generate a token** → tokenni nusxalang (keyinroq ko'rish imkoni bo'lmaydi!)

### Tokenni Kaggle Secrets ga saqlash
1. Kaggle notebook → **Add-ons → Secrets**
2. **Add a new secret**:
   - Label: `HF_TOKEN`
   - Value: `hf_...` (olgan tokeningiz)
3. **Save**

### Tekshiruv
- [ ] Hugging Face profilingizga kira olasiz
- [ ] API token olindi va nusxalab saqladingiz
- [ ] Kaggle Secrets da `HF_TOKEN` mavjud

---

## Yakuniy tekshiruv jadvali

| Hisobchi | Status |
|---|---|
| Kaggle hisob (email tasdiqlangan) | ☐ |
| Kaggle telefon verifikatsiyasi | ☐ |
| GitHub hisob | ☐ |
| `nlp-course-capstone` repo | ☐ |
| Hugging Face hisob | ☐ |
| `HF_TOKEN` olingan va saqlanaan | ☐ |

---

## Muammo bo'lsa

- **Kaggle SMS kelmasa:** VPN orqali urinib ko'ring yoki boshqa raqam ishlating
- **GitHub email kelmasa:** spam papkasini tekshiring
- **HF token ishlamasa:** yangi token oling (eski tokenni o'chirish shart emas)

Kurs boshlanishida (09:30, 15-iyun) `d01_orientatsiya.ipynb` barcha tekshiruvlarni
avtomatik o'tkazadi va aniq xato xabarlar ko'rsatadi.
