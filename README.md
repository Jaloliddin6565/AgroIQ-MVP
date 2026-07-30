<div align="center">

<img src="assets/agroiq_logo.png" alt="AgroIQ" width="420"/>

### Tuproq diagnostikasi va aniq o'g'itlash uchun integratsiyalashgan AI platformasi

**Portativ fosfor analizatori, universal tuproq sensori va sun'iy intellekt yordamida
dalangizning oziqa holatini baholang hamda ekinga mos o'g'itlash tavsiyasini oling.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-1B5E20.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-240%20passed-27AE60.svg)](tests/)
[![Version](https://img.shields.io/badge/version-v0.2.0-0E7C86.svg)](#)

</div>

---

> ⚠️ **Demo model.** Fosfor modeli sintetik ma'lumotlarda o'qitilgan. Universal sensorning
> azot va kaliy qiymatlari **kalibrlanmagan skrining indikatorlari** hisoblanadi.
> Batafsil: [MODEL_CARD.md](MODEL_CARD.md).

---

## 1. Loyiha haqida

**AgroIQ** — ikkita mustaqil o'lchov manbaini birlashtiruvchi tuproq intellekti platformasi:

| Manba | Nima o'lchaydi | Roli |
|---|---|---|
| **AgroIQ portativ fosfor analizatori** | Kolorimetrik (molibden-ko'k) reaksiya → Olsen-P ekvivalenti | Fosfor uchun **ASOSIY** manba |
| **Universal tuproq sensori** | pH, EC, harorat, namlik + N/P/K indikatorlari | Kontekst + **skrining** |

### Ish oqimi

```
AgroIQ kolorimetrik fosfor o'lchovi
        +
Universal sensor o'lchovlari
        +
Ekin va dala ma'lumotlari
        ↓
Ma'lumot validatsiyasi va ishonchlilikni baholash
        ↓
AI yordamidagi tuproq holati tahlili
        ↓
Ekinga xos o'g'itlash tavsiyasi
        ↓
Fermerga tushunarli tushuntirish va PDF hisobot
```

### Muammo

Fermerlar o'g'itlash bo'yicha qarorlarni ko'pincha umumiy tavsiyalar asosida, dala
darajasidagi tezkor ma'lumotsiz qabul qiladi. Laboratoriya tahlili qimmat, sekin yoki
uzoq bo'lishi mumkin. Umumiy tuproq sensorlari esa o'simlik o'zlashtira oladigan
fosforni ishonchli o'lchay olmaydi.

### Yechim

AgroIQ maxsus kolorimetrik fosfor analizatorini, ko'p parametrli tuproq sensori
ma'lumotlarini va tushuntiriladigan tavsiya dvigatelini birlashtiradi.

---

## 2. ⚠️ Asosiy ilmiy qoida

Platforma sensor chiqishlarini **hech qachon** laboratoriya o'lchovlariga teng deb
qaramaydi. To'rt daraja aniq ajratilgan:

| # | Qiymat | Maqomi | Ishlatilishi |
|---|---|---|---|
| 1 | **Kolorimetrik Olsen-P** | AI bahosi (mg/kg) | Miqdoriy fosfor tavsiyasi — **asosiy manba** |
| 2 | **Sensor P indikatori** | Empirik indikator | Faqat **moslik tekshiruvi** |
| 3 | **pH, EC, namlik, harorat** | To'g'ridan-to'g'ri o'lchov | Tuproq sharoiti talqini |
| 4 | **N va K indikatorlari** | Skrining | Faqat **sifatiy** baho |

**Qat'iy taqiqlar:**

- ❌ Sensor P qiymati Olsen-P o'rniga **hech qachon** qo'yilmaydi.
- ❌ Ikki fosfor qiymati **hech qachon o'rtachalanmaydi**.
- ❌ N va K uchun kalibrlashsiz **miqdoriy me'yor berilmaydi**.

Ikki fosfor qiymati sezilarli farq qilsa, `P_INDICATORS_DISAGREE` bayrog'i qo'yiladi va
foydalanuvchiga ko'rsatiladi:

> «Fosfor bo'yicha qurilmalar o'rtasida tafovut aniqlandi. AgroIQ kolorimetrik natijasi
> tavsiyada asosiy qiymat sifatida ishlatildi. Laboratoriya tasdig'i tavsiya etiladi.»

Kelajakda N/K uchun miqdoriy tavsiya `config/nutrient_calibration.json` faylidagi
`quantitative_enabled` bayrog'i orqali — **faqat validatsiyalangan kalibrlash
parametrlari kiritilgandan keyin** — yoqilishi mumkin.

---

## 3. AI arxitekturasi

```
┌──────────────────────────────────────────────────────────────────┐
│  1-YO'NALISH — FOSFOR (miqdoriy)                                 │
│  R, G, B, reaksiya vaqti, harorat → 23 optik xususiyat           │
│  → Olsen-P (mg/kg) + noaniqlik + ishonch darajasi                │
│  ❌ pH, EC, namlik BU YERDA ISHLATILMAYDI                        │
└──────────────────────────────────────────────────────────────────┘
                              +
┌──────────────────────────────────────────────────────────────────┐
│  2-YO'NALISH — TUPROQ SHAROITI (kontekst)                        │
│  Universal sensor → pH/EC/namlik/harorat sinflari                │
│                   → N va K sifatiy skrining bahosi               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  TUSHUNTIRILADIGAN BIRLASHTIRISH (src/data_fusion.py)            │
│  1. Vaqt belgilari va oraliqlarni tekshirish                     │
│  2. Yetishmayotgan/ziddiyatli kirishlarni aniqlash               │
│  3. Olsen-P baholash                                             │
│  4. pH/EC/namlik/harorat sinflash                                │
│  5. N va K ni konservativ talqin qilish                          │
│  6. Sensor P ↔ Olsen-P moslik tekshiruvi (o'rtachalashsiz)       │
│  7. Sifat bayroqlari                                             │
│  8. Umumiy ishonchlilik                                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  SHAFFOF AGRONOMIK QOIDALAR (qora quti EMAS)                     │
│  Oshkora hisob-kitob qadamlari + JSON konfiguratsiya             │
└──────────────────────────────────────────────────────────────────┘
```

### O'g'it me'yorining formulasi

```
kerakli_P2O5 = maqsadli_hosil × P2O5_olib_chiqish
               × fosfor_holati_koeffitsienti
               × pH_tuzatish_koeffitsienti
               × o'sish_bosqichi_koeffitsienti      (v0.2.0)
               × (1 − oldingi_o'g'itlash_krediti)   (v0.2.0)

o'g'it_mahsuloti = kerakli_P2O5 / mahsulotdagi_P2O5_ulushi
```

Har bir koeffitsient `config/*.json` da — agronom kodga tegmasdan sozlaydi.

---

## 4. Universal Soil Sensor Integration

### Nima uchun gateway kerak?

Streamlit Community Cloud'da ishlayotgan ilova foydalanuvchining kompyuteridagi
**RS485 portiga to'g'ridan-to'g'ri ula olmaydi**. Shu sababli:

```
Universal sensor → RS485/Modbus RTU → AgroIQ Edge Gateway → REST API → AgroIQ platformasi
```

Ilova faqat HTTP orqali gaplashadi. **Apparat ixtiyoriy** — demo, qo'lda va fayl
rejimlari har doim ishlaydi.

### Qo'llab-quvvatlanadigan rejimlar

| Rejim | Apparat kerakmi | Tavsif |
|---|---|---|
| **Demo ssenariy** | ❌ | To'rtta tayyorlangan ssenariy |
| **Qo'lda kiritish** | ❌ | Barcha qiymatlar shakl orqali |
| **Mock API** (`mock://`) | ❌ | Soxta gateway, tarmoqsiz |
| **Gateway API** | ✅ | Haqiqiy sensor lokal gateway orqali |
| **JSON / CSV yuklash** | ❌ | Eksport qilingan o'lchovlar |

### Lokal gateway'ni ishga tushirish

```bash
pip install -r edge_gateway/requirements-edge.txt

# Apparatsiz (namoyish uchun):
python -m edge_gateway.gateway --mode mock --port 8000

# Haqiqiy Modbus sensori bilan:
python -m edge_gateway.gateway --mode modbus --serial-port /dev/ttyUSB0 --baudrate 4800

# Serversiz bitta o'lchovni ko'rish:
python -m edge_gateway.gateway --print-once
```

So'ngra AgroIQ ilovasida **Qurilmalar → API** bo'limiga o'ting va manzilni kiriting:
`http://<gateway-ip>:8000/api/v1/readings/latest`

Batafsil: [`edge_gateway/README.md`](edge_gateway/README.md)

### API javob formati

```json
{
  "device_id": "SOIL-001",
  "timestamp": "2026-07-30T10:00:00+05:00",
  "source": "modbus_gateway",
  "nitrogen_indicator": 42.0,
  "phosphorus_indicator": 18.0,
  "potassium_indicator": 165.0,
  "ph": 7.8,
  "ec_ds_m": 1.9,
  "soil_temperature_c": 29.4,
  "soil_moisture_percent": 21.7,
  "quality_flags": []
}
```

> 🔒 **Xavfsizlik:** API token faqat sessiya xotirasida saqlanadi. U manba kodiga,
> konfiguratsiyaga yoki GitHub'ga **hech qachon** yozilmaydi. Gateway MVP da
> **faqat o'qish** rejimida ishlaydi — sensorga yozish qo'llab-quvvatlanmaydi.

---

## 5. Ekran ko'rinishlari

Skrinshotlar `assets/screenshots/` katalogida.

| Sahifa | Fayl |
|---|---|
| Bosh sahifa | `01_home.png` |
| Demo rejimi (4 ssenariy) | `02_demo.png` |
| Natijalar — past fosfor | `03_results_low.png` |
| **Qurilmalar tafovuti** | `04_results_disagreement.png` |
| Natijalar — yuqori fosfor | `05_results_high.png` |
| Qurilmalar sahifasi | `06_devices.png` |
| Gateway API rejimi | `07_devices_api.png` · `08_devices_api_data.png` |
| Yangi tahlil (4 bo'lim) | `09_analysis.png` |
| Fayl yuklash rejimi | `11_analysis_upload.png` |
| Model va validatsiya | `12_model.png` |
| Loyiha haqida | `13_about.png` |
| Mobil ko'rinish | `14_mobile.png` |

---

## 6. Repozitoriya tuzilishi

```
AgroIQ/
├── app.py                          # Streamlit ilovasi (7 bo'lim, yuqori navigatsiya)
├── requirements.txt                # Apparat bog'liqliklarisiz
├── README.md · DEPLOY.md · MODEL_CARD.md · LICENSE
│
├── config/                         # ⚙️ Agronom sozlaydigan qismlar
│   ├── crop_profiles.json           #   ekin me'yorlari, pH tuzatish
│   ├── phosphorus_thresholds.json   #   fosfor sinflari
│   ├── fertilizer_products.json     #   o'g'it tarkibi
│   ├── sensor_thresholds.json       #   🆕 sensor talqin chegaralari
│   ├── device_profiles.json         #   🆕 qurilma profillari
│   ├── nutrient_calibration.json    #   🆕 N/K kalibrlash bayroqlari
│   └── recommendation_rules.json    #   🆕 bosqich, kredit, indeks
│
├── src/
│   ├── sensor_schemas.py           # 🆕 sensor/analizator sxemalari, bayroqlar
│   ├── device_integration.py       # 🆕 demo/API/fayl rejimlari, mock gateway
│   ├── data_fusion.py              # 🆕 tushuntiriladigan birlashtirish
│   ├── soil_interpretation.py      # 🆕 pH/EC/namlik/N/K talqini
│   ├── demo_scenarios.py           # 🆕 4 ta ssenariy
│   ├── model_training.py           # kolorimetrik model (o'zgarmagan)
│   ├── model_inference.py          # bashorat + noaniqlik (o'zgarmagan)
│   ├── recommendation_engine.py    # kengaytirilgan agronomik qoidalar
│   ├── report_generator.py         # PDF (ikkala qurilma ma'lumoti bilan)
│   ├── explanations.py · feature_engineering.py · data_validation.py
│   └── ui_components.py            # CSS, yuqori navigatsiya, grafiklar
│
├── edge_gateway/                   # 🆕 LOKAL gateway (bulutga joylashtirilmaydi)
│   ├── gateway.py · modbus_reader.py · mock_sensor.py · schemas.py
│   ├── config.example.json · requirements-edge.txt · README.md
│
├── data/ · models/ · scripts/ · assets/
└── tests/                          # 240 ta test
    ├── test_validation.py · test_inference.py · test_recommendations.py
    ├── test_sensors.py             # 🆕 sensor, API, fayl, kalibrlash bayroqlari
    ├── test_fusion.py              # 🆕 birlashtirish, tafovut, ssenariylar
    └── test_integration.py         # 🆕 navigatsiya, PDF, bulut mosligi
```

---

## 7. Lokal o'rnatish

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows uchun:

```bat
.venv\Scripts\activate
```

Keyin:

```bash
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

Testlar:

```bash
pytest
```

Brauzerda: <http://localhost:8501>

---

## 8. Real ma'lumot bilan ishlash

### Fosfor modelini qayta kalibrlash

`data/soil_samples.csv` faylini real ma'lumot bilan almashtiring:

| Ustun | Izoh |
|---|---|
| `sample_id`, `field_id` | `field_id` guruhlangan bo'linish uchun **muhim** |
| `red`, `green`, `blue` | 0–255 |
| `reaction_time_sec`, `sample_temperature_c` | O'lchov sharoiti |
| `lab_olsen_p_mg_kg` | **Maqsad** — laboratoriya Olsen-P |

`data_source` ustunini olib tashlang, so'ng `python scripts/train_model.py` ni qayta
ishga tushiring.

### N va K uchun miqdoriy tavsiyani yoqish

`config/nutrient_calibration.json` da har bir oziq modda uchun talablar ro'yxati bor.
Ular bajarilgandan keyin:

```json
"nitrogen": {
  "quantitative_enabled": true,
  "calibration": { "validated": true, "slope": 0.85, "intercept": 2.1, "r2": 0.78 }
}
```

Shundan keyingina platforma azot bo'yicha miqdoriy me'yor bera boshlaydi.

---

## 9. Biznes modeli

| Yo'nalish | Tavsif |
|---|---|
| AgroIQ o'qish qurilmasi savdosi | Portativ kolorimetrik analizator |
| Universal sensor to'plami | Sensor sotuvi yoki integratsiya |
| Fosfor kartrijlari | Takroriy daromad manbai |
| Xizmat sifatida tahlil | Sayyor tuproq tahlili |
| AI platforma obunasi | Yillik obuna |
| B2B klaster paneli | Ko'p dalali boshqaruv paneli |
| Kalibrlash va texnik xizmat | Davriy kalibrlash |
| Agronomik API xizmatlari | Uchinchi tomon integratsiyasi |

---

## 10. Yo'l xaritasi

| Bosqich | Modul | Holat |
|---|---|---|
| v0.1.0 | Fosfor + o'g'itlash tavsiyasi | ✅ |
| **v0.2.0** | **Universal sensor, gateway, birlashtirish qatlami** | ✅ **Joriy** |
| v0.3.0 | Real kalibrlash dataseti (150–300 juftlik) | 🔜 |
| v0.4.0 | Validatsiyalangan N va K kartrijlari | 📋 |
| — | Sug'orish suvi diagnostikasi | 📋 |
| — | Mobil ilova (oflayn) | 📋 |
| — | Dron va sun'iy yo'ldosh integratsiyasi | 📋 |
| — | O'zgaruvchan me'yorli o'g'itlash xaritalari | 📋 |
| — | Markaziy Osiyo kalibrlash datasetlari | 📋 |

---

## 11. Jamoa

| Rol | Ism | Mas'uliyat |
|---|---|---|
| Loyiha rahbari | *[Ism Familiya]* | Mahsulot strategiyasi, biznes |
| AI / ML muhandisi | *[Ism Familiya]* | Model, kalibrlash, ma'lumotlar |
| Agronom-maslahatchi | *[Ism Familiya]* | Agronomik qoidalar, dala validatsiyasi |
| Apparat muhandisi | *[Ism Familiya]* | Analizator optikasi, sensor integratsiyasi |

Aloqa: *[email]* · *[telefon]*

---

## 12. Deployment

To'liq qo'llanma: [**DEPLOY.md**](DEPLOY.md)

Qisqacha: GitHub → <https://share.streamlit.io> → asosiy fayl `app.py`, Python 3.11.
Pullik API, tashqi LLM yoki maxfiy kalit talab qilinmaydi.

> **Joylashtirilgan MVP havolasi:** *(deploy qilingandan keyin shu yerga qo'shiladi)*

---

## 13. Ochiq cheklovlar

Bu cheklovlar ilovada ham yashirilmaydi:

1. **Fosfor modeli sintetik ma'lumotda o'qitilgan** — real qurilma-laboratoriya
   juftliklari bilan kalibrlash talab qilinadi.
2. **N va K sertifikatlangan o'lchov emas** — ular skrining indikatori sifatida
   qaraladi va miqdoriy me'yor uchun ishlatilmaydi.
3. **Dala validatsiyasi yakunlanmagan** — o'g'itlash me'yorlari hosildorlik bo'yicha
   tajribalar bilan tasdiqlanmagan.
4. **Sensor registr xaritasi qurilmaga bog'liq** — Modbus manzillari ishlab
   chiqaruvchiga qarab farq qiladi.
5. **Laboratoriya o'rnini bosmaydi** — rasmiy agrokimyoviy xulosa uchun
   sertifikatlangan tahlil zarur.

> **AgroIQ MVP natijalari dastlabki tavsiya hisoblanadi. Qurilma va algoritm real
> hududiy tuproq namunalari hamda dala tajribalari bilan to'liq validatsiya qilingunga
> qadar yakuniy o'g'itlash qarori malakali agronom bilan kelishilishi kerak.**

---

## Litsenziya

[MIT](LICENSE) — agronomik ogohlantirish bilan.
