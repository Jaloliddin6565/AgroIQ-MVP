<div align="center">

<img src="assets/agroiq_logo.png" alt="AgroIQ" width="420"/>

### Tuproq diagnostikasi va aniq o'g'itlash uchun aqlli platforma

**O'simlik o'zlashtira oladigan fosforni tezkor baholang va dalangiz uchun tushunarli
o'g'itlash tavsiyasini oling.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-1B5E20.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-99%20passed-27AE60.svg)](tests/)

</div>

---

> ⚠️ **Demo model.** Ushbu versiya prototipni namoyish qilish uchun mo'ljallangan va dala
> sharoitida ishlatishdan oldin real laboratoriya ma'lumotlari bilan kalibrlanishi shart.
> Batafsil: [MODEL_CARD.md](MODEL_CARD.md).

---

## 1. Loyiha haqida

**AgroIQ** — portativ kolorimetrik tuproq diagnostikasi va sun'iy intellektga asoslangan
o'g'itlash tavsiyalari platformasi. Birinchi MVP o'simlik o'zlashtira oladigan **fosforga
(Olsen-P)** qaratilgan.

Asosiy ish oqimi:

```
Kolorimetrik o'lchov
   → AI fosforni baholash (+ noaniqlik)
      → Tuproq holatini tahlil qilish
         → Ekinga moslashtirilgan o'g'itlash tavsiyasi
            → Tushuntiriladigan fermer hisoboti (PDF)
```

### Muammo

Fermerlarda dala darajasida o'simlik o'zlashtira oladigan fosforni **tez va arzon**
o'lchash imkoniyati deyarli yo'q. Laboratoriya tahlili qimmat, uzoq davom etadi va ko'pincha
mavsumiy qarorlar uchun kech keladi. Natijada o'g'it me'yori taxminan belgilanadi — bu
ikki tomonlama yo'qotishga olib keladi: keraksiz xarajat yoki hosildorlikning pasayishi.

### Yechim

Portativ kolorimetrik o'lchov qurilmasi + AI kalibrlash modeli + shaffof agronomik
tavsiya dvigateli. Har bir tavsiya **sababi, ishonch darajasi va cheklovlari** bilan
birga taqdim etiladi.

### Maqsadli foydalanuvchilar

Fermerlar · agroklasterlar · agronomlar · sayyor tuproq tahlili xizmatlari ·
agrolaboratoriyalar · o'g'it distribyutorlari · agrokonsalting tashkilotlari.

Interfeys tili — **o'zbek (lotin)**, ilmiy jargonsiz, fermerga tushunarli.

---

## 2. AI arxitekturasi

Loyihaning eng muhim ilmiy tamoyili — **AI va agronomik qoidalarni qat'iy ajratish**:

```
┌───────────────────────────────────────────────────────────────┐
│  1-BOSQICH — AI (mashinali o'qitish)                          │
│                                                               │
│  Kirish : R, G, B, reaksiya vaqti, harorat                    │
│           → 23 ta optik xususiyat (absorbsiya, HSV, indekslar)│
│  Chiqish: Olsen-P (mg/kg) + noaniqlik + ishonch darajasi      │
│                                                               │
│  ❌ pH, EC, namlik, ekin, hosildorlik BU YERDA ISHLATILMAYDI  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│  2-BOSQICH — Shaffof agronomik qoidalar (qora quti EMAS)      │
│                                                               │
│  Kirish : baholangan Olsen-P, fosfor sinfi, ekin, hosildorlik,│
│           pH, EC, namlik, o'g'it tarkibi, JSON konfiguratsiya │
│  Chiqish: o'g'it turi, me'yor oralig'i, muddat, usul,         │
│           ogohlantirishlar, tushuntirish                      │
│                                                               │
│  Har bir raqam 6 qadamli oshkora hisob-kitobda ko'rsatiladi   │
└───────────────────────────────────────────────────────────────┘
```

**Nima uchun bu muhim:** pH, EC va namlik fosforning *o'zlashtirilishiga* ta'sir qiladi,
lekin ular tuproqdagi fosfor *konsentratsiyasini* aniqlamaydi. Ularni prediktor sifatida
ishlatish ilmiy jihatdan noto'g'ri bo'lar edi.

### O'g'it me'yorining formulasi

```
kerakli_P2O5 (kg/ga) = maqsadli_hosil (t/ga)
                       × P2O5_olib_chiqish (kg/t)      ← ekin profilidan
                       × fosfor_holati_koeffitsienti    ← tuproq tahlilidan
                       × pH_tuzatish_koeffitsienti      ← tuproq muhitidan

o'g'it_mahsuloti (kg/ga) = kerakli_P2O5 / mahsulotdagi_P2O5_ulushi
```

Barcha koeffitsientlar `config/*.json` fayllarida — agronom kodga tegmasdan sozlashi mumkin.

---

## 3. Ekran ko'rinishlari

Skrinshotlar `assets/screenshots/` katalogida saqlanadi.

| Sahifa | Fayl | Nima ko'rsatiladi |
|---|---|---|
| Bosh sahifa | `01_home.png` | Qiymat taklifi, muammo/yechim, 4 qadamli ish oqimi |
| Yangi tahlil | `02_analysis.png` | Demo namuna va qo'lda kiritish shakli |
| Natijalar | `03_results.png` | Olsen-P, fosfor shkalasi, o'g'it me'yori, tushuntirish |
| Yuqori fosfor ssenariysi | `04_results_high.png` | "Qo'shimcha fosfor tavsiya etilmaydi" holati |
| Model va validatsiya | `05_model.png` | Metrikalar, model taqqoslash, kalibrlash oralig'i |
| Demo rejimi | `06_demo.png` | Uchta tayyor ssenariy |
| Loyiha haqida | `07_about.png` | Biznes modeli va yo'l xaritasi |

---

## 4. Repozitoriya tuzilishi

```
AgroIQ/
├── app.py                        # Streamlit ilovasi (6 ta bo'lim)
├── requirements.txt
├── README.md · DEPLOY.md · MODEL_CARD.md · LICENSE · .gitignore
├── conftest.py                   # pytest sozlamalari va fikstura'lar
│
├── .streamlit/config.toml        # Rang mavzusi
│
├── assets/
│   ├── agroiq_logo.png           # Yo'q bo'lsa — matnli logotipga o'tadi
│   └── screenshots/
│
├── config/                       # ⚙️ Agronom sozlaydigan qismlar
│   ├── crop_profiles.json         #   ekin me'yorlari, pH tuzatish, sharoit chegaralari
│   ├── phosphorus_thresholds.json #   fosfor sinflari (Juda past … Yuqori)
│   └── fertilizer_products.json   #   o'g'it tarkibi (P2O5 ulushi)
│
├── data/
│   ├── soil_samples.csv          # O'quv dataseti (demo: SYNTHETIC_DEMO markeri bilan)
│   └── demo_samples.csv          # Uchta namoyish ssenariysi
│
├── models/
│   └── phosphorus_model.joblib   # O'qitilgan pipeline + metrikalar + kalibrlash
│
├── src/
│   ├── data_validation.py        # pydantic validatsiya + konfiguratsiya yuklash
│   ├── feature_engineering.py    # 23 ta optik xususiyat
│   ├── model_training.py         # sintetik generator + model taqqoslash
│   ├── model_inference.py        # bashorat + noaniqlik + ishonch
│   ├── recommendation_engine.py  # shaffof agronomik qoidalar
│   ├── explanations.py           # "Nima uchun bu tavsiya berildi?"
│   ├── report_generator.py       # PDF hisobot (reportlab)
│   └── ui_components.py          # CSS, kartalar, Plotly grafiklar
│
├── scripts/
│   └── train_model.py            # Modelni o'qitish CLI
│
└── tests/
    ├── test_validation.py
    ├── test_inference.py
    └── test_recommendations.py
```

---

## 5. Lokal o'rnatish

### 5.1. Virtual muhit yaratish

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows uchun:

```bat
.venv\Scripts\activate
```

### 5.2. Kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

### 5.3. Modelni o'qitish

```bash
python scripts/train_model.py
```

Bu buyruq:
- `data/soil_samples.csv` ni tekshiradi (kamida 30 ta yaroqli qator kerak);
- real ma'lumot bo'lmasa — takrorlanuvchi sintetik dataset yaratadi (seed = 42) va uni
  `data_source = SYNTHETIC_DEMO` markeri bilan belgilaydi;
- uchta modelni taqqoslaydi va eng yaxshisini tanlaydi;
- `models/phosphorus_model.joblib` faylini saqlaydi;
- `data/demo_samples.csv` ni yangilaydi.

### 5.4. Ilovani ishga tushirish

```bash
streamlit run app.py
```

Brauzerda: <http://localhost:8501>

### 5.5. Testlarni ishga tushirish

```bash
pytest
```

---

## 6. Real ma'lumot bilan ishlash

Demo rejimdan chiqish uchun `data/soil_samples.csv` faylini real ma'lumot bilan
almashtiring:

| Ustun | Turi | Izoh |
|---|---|---|
| `sample_id` | matn | Namuna identifikatori |
| `field_id` | matn | **Muhim** — guruhlangan bo'linish uchun |
| `red`, `green`, `blue` | son | 0–255 |
| `reaction_time_sec` | son | Reaksiya vaqti, sekund |
| `sample_temperature_c` | son | Namuna harorati, °C |
| `lab_olsen_p_mg_kg` | son | **Maqsad** — laboratoriya Olsen-P, mg/kg |
| `region`, `soil_type`, `measurement_date` | ixtiyoriy | Metama'lumot |

`data_source` ustunini olib tashlang (yoki `SYNTHETIC_DEMO` dan boshqa qiymat qo'ying),
so'ng `python scripts/train_model.py` ni qayta ishga tushiring. Demo bannerlar avtomatik
o'chadi.

### Agronomik sozlash

Kodni o'zgartirmasdan quyidagilarni sozlash mumkin:

- `config/phosphorus_thresholds.json` — fosfor sinflari chegaralari;
- `config/crop_profiles.json` — ekin me'yorlari, pH tuzatish, tuproq sharoiti chegaralari;
- `config/fertilizer_products.json` — o'g'it tarkibi va indikativ narxlar.

---

## 7. Demo ma'lumot haqida ogohlantirish

> Joriy model **sintetik (sun'iy yaratilgan)** ma'lumotlar asosida o'qitilgan. Bu ma'lumot
> real dala tajribasi natijasi **emas**. U Beer-Lambert qonuni va reaksiya kinetikasi
> asosida fizik modellashtirilgan bo'lsa-da, real tuproq namunalarida validatsiya
> qilinmagan.
>
> Ilova ichida demo holati bir necha joyda ochiq ko'rsatiladi: yon panelda, natijalar
> sahifasida va "Model va validatsiya" sahifasida.
>
> Loyiha hech qachon sintetik ma'lumotni real eksperimental ma'lumot sifatida taqdim etmaydi.

---

## 8. Biznes modeli

| Yo'nalish | Tavsif |
|---|---|
| 📦 Qurilma savdosi | Portativ AgroIQ o'qish qurilmasi |
| 🧪 Test kartrijlari | Takroriy sotiladigan reagent kartrijlari (barqaror daromad) |
| 🚐 Xizmat sifatida tahlil | Sayyor tuproq tahlili (har bir test uchun to'lov) |
| ☁️ Platforma obunasi | AI tahlil platformasiga yillik obuna |
| 🏢 B2B xizmatlar | Agroklasterlar uchun integratsiya va ma'lumot xizmatlari |

---

## 9. Yo'l xaritasi

| Bosqich | Modul | Holat |
|---|---|---|
| MVP | Fosfor (Olsen-P) + o'g'itlash tavsiyasi | ✅ Ushbu repozitoriya |
| 1-bosqich | Real kalibrlash dataseti (150–300 juftlik) | 🔜 Keyingi qadam |
| 2-bosqich | Kaliy (K₂O) moduli | 📋 Rejalashtirilgan |
| 2-bosqich | Nitrat azot (NO₃-N) moduli | 📋 Rejalashtirilgan |
| 3-bosqich | Sug'orish suvi tahlili | 📋 Rejalashtirilgan |
| 3-bosqich | Dron va sun'iy yo'ldosh integratsiyasi | 📋 Rejalashtirilgan |
| 4-bosqich | O'zgaruvchan me'yorli o'g'itlash xaritalari | 📋 Rejalashtirilgan |
| 4-bosqich | Oflayn mobil ilova | 📋 Rejalashtirilgan |

---

## 10. Jamoa

| Rol | Ism | Mas'uliyat |
|---|---|---|
| Loyiha rahbari | *[Ism Familiya]* | Mahsulot strategiyasi, biznes rivojlantirish |
| AI / ML muhandisi | *[Ism Familiya]* | Model, kalibrlash, ma'lumotlar tahlili |
| Agronom-maslahatchi | *[Ism Familiya]* | Agronomik qoidalar, dala validatsiyasi |
| Apparat muhandisi | *[Ism Familiya]* | Optik o'qish qurilmasi |

Aloqa: *[email]* · *[telefon]*

---

## 11. Deployment

Streamlit Community Cloud'ga joylashtirish bo'yicha to'liq qo'llanma:
[**DEPLOY.md**](DEPLOY.md)

Qisqacha: repozitoriyani GitHub'ga yuklang → <https://share.streamlit.io> → asosiy fayl
sifatida `app.py` ni ko'rsating. Hech qanday pullik API, tashqi LLM yoki maxfiy kalit
talab qilinmaydi.

---

## 12. Muhim ogohlantirish

> **AgroIQ MVP natijalari dastlabki tavsiya hisoblanadi. Qurilma va algoritm real hududiy
> tuproq namunalari hamda dala tajribalari bilan to'liq validatsiya qilingunga qadar
> yakuniy o'g'itlash qarori malakali agronom bilan kelishilishi kerak.**

AgroIQ sertifikatlangan laboratoriya uskunasi emas va akkreditatsiyalangan tuproq
tahlilining o'rnini bosmaydi.

---

## Litsenziya

[MIT](LICENSE) — agronomik ogohlantirish bilan.
