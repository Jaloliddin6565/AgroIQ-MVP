# Model Card — AgroIQ fosforni baholash modeli

**Model nomi:** AgroIQ Olsen-P colorimetric estimator
**Versiya:** 0.2.0 (MVP / pilot — ko'p sensorli)
**Sana:** 2026
**Holat:** ⚠️ **Demo model** — sintetik ma'lumotlarda o'qitilgan, dala sharoitida ishlatishdan
oldin real laboratoriya ma'lumotlari bilan kalibrlanishi shart.

---

## 1. Modelning maqsadi

Model kolorimetrik (molibden-ko'k usuli) o'lchov natijasidan **o'simlik o'zlashtira oladigan
fosfor (Olsen-P, mg/kg)** miqdorini baholaydi.

**Nima uchun ishlatiladi:**
- optik kalibrlash (rang → konsentratsiya bog'lanishini o'rganish);
- fosforni tezkor baholash;
- baholash noaniqligini aniqlash;
- natijani agronomik talqin qilish uchun asos berish.

**Nima uchun ISHLATILMAYDI:**
- o'g'it me'yorini "o'ylab topish" uchun — me'yor alohida, shaffof agronomik qoidalar
  bilan hisoblanadi (`src/recommendation_engine.py`);
- laboratoriya tahlilining o'rnini bosish uchun;
- rasmiy agrokimyoviy xulosa berish uchun.

---

## 2. O'quv ma'lumotlari

### Hozirgi holat: sintetik demo dataset

Real laboratoriya dataseti (`data/soil_samples.csv`, kamida 30 ta yaroqli qator) mavjud
bo'lmagani uchun model **takrorlanuvchi sintetik dataset** (seed = 42) asosida o'qitilgan.

| Ko'rsatkich | Qiymat |
|---|---|
| Namunalar soni | ~265 ta |
| Dalalar soni | 48 ta |
| Olsen-P oralig'i | ~2–55 mg/kg |
| Manba | `src/model_training.py::generate_demo_dataset` |
| Marker | `data_source = SYNTHETIC_DEMO` |

### Sintetik ma'lumot qanday hosil qilinadi

Generator o'zboshimchalik bilan son yaratmaydi — u fizik modelga asoslanadi:

1. **Reaksiya kinetikasi:** rang rivojlanishi `1 − exp(−k(T)·t)`, bu yerda `k(T)` harorat
   bilan Q10 = 1.9 koeffitsienti bo'yicha o'zgaradi.
2. **Beer-Lambert qonuni:** `A = A₀ + ε·C_effective`, har bir kanal uchun alohida yutilish
   koeffitsienti (`ε_red ≫ ε_green > ε_blue`, chunki ko'k rangli eritma qizil yorug'likni
   ko'proq yutadi).
3. **Kanal qiymati:** `I = 255 · 10^(−A)`.
4. **Realistik shovqin manbalari:**
   - tuproq matritsasi interferensiyasi (dala darajasida ±6%, namuna darajasida ±5%);
   - sensor/yoritilish siljishi (dala bo'yicha σ = 4 birlik);
   - o'lchov shovqini (σ = 3 birlik);
   - laboratoriya Olsen-P takrorlanuvchanligi (±8%).

**Muhim:** bu ma'lumot real dala tajribasi natijasi EMAS. U faqat dasturiy prototipni
namoyish qilish uchun xizmat qiladi va hech qachon real eksperimental ma'lumot sifatida
taqdim etilmaydi. Ilova ichida har bir tegishli joyda demo banner ko'rsatiladi.

### Real datasetga o'tish

`data/soil_samples.csv` fayliga quyidagi ustunlar bilan real ma'lumot qo'yilsa va unda
kamida 30 ta yaroqli qator bo'lsa, `python scripts/train_model.py` avtomatik ravishda real
ma'lumotdan foydalanadi va demo bannerlar o'chadi:

```
sample_id, field_id, red, green, blue, reaction_time_sec,
sample_temperature_c, lab_olsen_p_mg_kg
```

Ixtiyoriy: `region`, `soil_type`, `measurement_date`.

---

## 3. Kirish xususiyatlari (features)

Model **faqat optik va o'lchov sharoiti** ma'lumotlaridan foydalanadi. Jami 23 ta xususiyat
(`src/feature_engineering.py`):

| Guruh | Xususiyatlar |
|---|---|
| Xom kanallar | `red`, `green`, `blue` |
| Normallashgan RGB | `red_norm`, `green_norm`, `blue_norm` |
| Nisbatlar | `ratio_r_g`, `ratio_r_b`, `ratio_g_b` |
| HSV | `hue`, `saturation`, `value` |
| Yorqinlik | `brightness` |
| Absorbsiya indekslari | `absorbance_red`, `absorbance_green`, `absorbance_blue` |
| Rang indekslari | `blue_red_index`, `green_red_index`, `chroma` |
| O'lchov sharoiti | `reaction_time_sec`, `log_reaction_time`, `sample_temperature_c`, `time_temp_index` |

### ⚠️ Qat'iy ilmiy chegaralanish

Quyidagi o'zgaruvchilar **fosfor konsentratsiyasini bashorat qilishda ISHLATILMAYDI**:

- tuproq pH;
- elektr o'tkazuvchanlik (EC);
- tuproq namligi;
- ekin turi;
- maqsadli hosildorlik.

Ular kimyoviy fosfor o'lchovining o'rnini bosa olmaydi. Bu o'zgaruvchilar **faqat tavsiya
bosqichida** — fosforning o'simlik tomonidan o'zlashtirilishini va o'g'it qo'llash
sharoitini baholash uchun ishlatiladi.

---

## 4. Maqsad o'zgaruvchisi (target)

`lab_olsen_p_mg_kg` — laboratoriyada Olsen usuli (0.5 M NaHCO₃, pH 8.5) bilan aniqlangan
o'simlik o'zlashtira oladigan fosfor, mg/kg.

---

## 5. Modelni tanlash metodologiyasi

Har bir o'qitishda **uchta nomzod** bir xil test to'plamida taqqoslanadi:

| Nomzod | Tuzilishi |
|---|---|
| Chiziqli regressiya | `StandardScaler` → `LinearRegression` |
| Polinomial regressiya (2-daraja) | `StandardScaler` → `PolynomialFeatures(2)` → `Ridge(α=10)` |
| Random Forest | `RandomForestRegressor(n=350, min_samples_leaf=2, max_features=0.6)` |

**Bo'linish:** `GroupShuffleSplit` (test 25%), **`field_id` bo'yicha guruhlangan** — bitta
daladan olingan namunalar o'zaro bog'liq bo'lgani uchun ular train va test o'rtasida
bo'linmaydi. `field_id` mavjud bo'lmasa, oddiy tasodifiy bo'linish ishlatiladi.
Qo'shimcha ravishda `GroupKFold` (5 fold) cross-validation hisoblanadi.

**Tanlash qoidasi:** eng yuqori R² ga ega model tanlanadi. Agar Random Forest natijasi eng
yaxshi modeldan 0.02 R² ichida bo'lsa, daraxtlar orqali noaniqlikni baholash imkoniyati
uchun unga ustunlik beriladi.

**Barcha tasodifiylik `random_state = 42` bilan qat'iy belgilangan** — natijalar
takrorlanuvchi.

---

## 6. Metrikalar (joriy demo model)

Test to'plami — o'qitishda umuman ko'rilmagan dalalar:

| Model | R² | RMSE (mg/kg) | MAE (mg/kg) |
|---|---|---|---|
| Chiziqli regressiya | 0.782 | 2.09 | 1.58 |
| **Polinomial regressiya (2-daraja)** ✅ | **0.799** | **2.01** | **1.45** |
| Random Forest | 0.690 | 2.49 | 1.84 |

**GroupKFold (5 fold) cross-validation:** R² = 0.930, RMSE = 1.70 mg/kg.

**Izoh:** bu ma'lumot to'plamida rang → konsentratsiya bog'lanishi silliq va monoton
bo'lgani uchun parametrik modellar Random Forest'dan ustun chiqdi. Random Forest chiziqli
bo'lmagan asosiy nomzod sifatida har doim sinovdan o'tkaziladi, ammo natija sun'iy ravishda
uning foydasiga o'zgartirilmaydi. Real dala ma'lumotlarida bu tartib o'zgarishi mumkin.

Aniq joriy qiymatlar ilovaning **"Model va validatsiya"** sahifasida jonli ko'rsatiladi.

---

## 7. Noaniqlikni baholash

Har bir bashorat uchun standart noaniqlik ikki manbadan birlashtiriladi:

1. **Model kelishmovchiligi (σ_model).** Random Forest tanlanganda — alohida daraxtlar
   bashoratlarining standart chetlanishi. Boshqa modellarda bu qism 0 ga teng.
2. **Qoldiq tarqoqlik (σ_qoldiq).** Geteroskedastik baho: test to'plamidagi `|qoldiq|`
   qiymatlari bashoratga nisbatan chiziqli moslanadi va normal taqsimot uchun
   `σ = √(π/2)·E|qoldiq|` sifatida qayta hisoblanadi:

   ```
   σ_qoldiq(ŷ) ≈ intercept + slope · ŷ
   ```

   Bu yondashuv fizik jihatdan asoslangan: kolorimetrik o'lchovda xatolikning katta qismi
   ko'paytiruvchi (nisbiy) xarakterga ega, shuning uchun yuqori konsentratsiyalarda absolyut
   xato ham kattaroq bo'ladi.

**Yakuniy noaniqlik:** `σ = √(σ_model² + σ_qoldiq²)`

Ilovada ko'rsatiladi: `±σ` (standart noaniqlik) va taxminiy 95% oraliq (`±1.96σ`).

### Ishonch darajasi qoidalari (konservativ)

| Daraja | Shart |
|---|---|
| **Yuqori** | nisbiy noaniqlik < 15% **va** namuna kalibrlash oralig'ida |
| **O'rtacha** | nisbiy noaniqlik < 30% |
| **Past** | qolgan barcha holatlar |

Qo'shimcha qoidalar:
- kalibrlash oralig'idan chiqqan **har qanday** namuna avtomatik "past" deb belgilanadi va
  foydalanuvchiga laboratoriya tasdig'i tavsiya qilinadi;
- bashorat o'quv to'plamining maqsad oralig'idan chiqsa, daraja bir pog'ona pasaytiriladi.

---

## 8. Kalibrlash oralig'i

Model o'quv to'plamidagi `red`, `green`, `blue`, `reaction_time_sec`,
`sample_temperature_c` qiymatlarining min/max oralig'ida validatsiya qilingan (aniq
qiymatlar "Model va validatsiya" sahifasida ko'rsatilgan). Kirish qiymati oraliqdan
5% dan ko'proq chetga chiqsa, ilova quyidagini ko'rsatadi:

> "Ushbu namuna modelning validatsiya qilingan kalibrlash oralig'idan tashqarida.
> Laboratoriya tasdig'i tavsiya etiladi."

---

## 9. Cheklovlar

1. **Laboratoriya o'rnini bosmaydi.** Bu tezkor dala bahosi. Rasmiy agrokimyoviy xulosa
   uchun sertifikatlangan laboratoriya tahlili talab qilinadi.
2. **Sintetik ma'lumot.** Joriy model real tuproq namunalarida kalibrlanmagan. Ko'rsatilgan
   metrikalar generator taxminlariga bog'liq va real aniqlikni kafolatlamaydi.
3. **Hududiy sezgirlik.** Tuproq turi, karbonatlilik, temir/kremniy birikmalari, reagent
   partiyasi va qurilma optikasi kalibrlashga sezilarli ta'sir qiladi. Har bir hudud va
   har bir qurilma partiyasi uchun qayta kalibrlash zarur.
4. **Faqat fosfor.** Azot, kaliy, mikroelementlar va organik modda baholanmaydi.
5. **Ekstrapolyatsiya.** Kalibrlash oralig'idan tashqarida model ishonchsiz. Yuqori
   konsentratsiyalarda (>50 mg/kg) suyultirish talab qilinadi.
6. **Dala validatsiyasi yo'q.** O'g'itlash me'yorlari hosildorlik bo'yicha dala tajribalari
   bilan tasdiqlanmagan.

---

## 10. Taqiqlangan foydalanish

Ushbu modeldan quyidagi maqsadlarda foydalanish **mumkin emas**:

- rasmiy laboratoriya sertifikati yoki agrokimyoviy pasport o'rnida;
- sug'urta, kredit yoki subsidiya bo'yicha yuridik ahamiyatga ega qaror uchun;
- yer sifati bo'yicha rasmiy davlat hisoboti uchun;
- agronom nazoratisiz avtomatik o'g'itlash tizimini boshqarish uchun;
- "dala tajribasida tasdiqlangan" deb reklama qilish uchun (bunday ma'lumot hozircha yo'q).

---

## 11. Talab qilinadigan validatsiya (keyingi qadamlar)

Modelni dala sharoitida ishlatishdan oldin quyidagilar bajarilishi kerak:

1. **Kalibrlash to'plami.** Kamida 150–300 ta real juftlik (kolorimetrik o'lchov +
   akkreditatsiyalangan laboratoriya Olsen-P natijasi), kamida 3 ta hududdan va turli
   tuproq tiplaridan.
2. **Mustaqil test.** Kalibrlashda umuman qatnashmagan dalalarda tekshirish
   (`GroupShuffleSplit` bilan bir xil mantiq).
3. **Takrorlanuvchanlik sinovi.** Bir xil namunani bir necha marta o'lchash orqali
   qurilmaning ichki xatoligini aniqlash.
4. **Interferensiya tekshiruvi.** Karbonatli, sho'rlangan va temirga boy tuproqlarda
   alohida sinov.
5. **Agronomik dala tajribasi.** Tavsiya etilgan me'yorlarni nazorat variantlari bilan
   taqqoslab, hosildorlikka ta'sirini o'lchash (kamida 2 mavsum).
6. **Chegaralarni tasdiqlash.** `config/phosphorus_thresholds.json` dagi sinf chegaralari
   mahalliy tuproqshunos mutaxassis tomonidan tasdiqlanishi kerak.

---

## 12. Javobgarlik va ogohlantirish

> AgroIQ MVP natijalari dastlabki tavsiya hisoblanadi. Qurilma va algoritm real hududiy
> tuproq namunalari hamda dala tajribalari bilan to'liq validatsiya qilingunga qadar
> yakuniy o'g'itlash qarori malakali agronom bilan kelishilishi kerak.

---

## 13. v0.2.0 — Ko'p sensorli arxitektura va oziq moddalar maqomi

v0.2.0 da platformaga universal ko'p parametrli tuproq sensori qo'shildi. **Fosfor
modeli o'zgartirilmadi va qayta o'qitilmadi** — u avvalgidek faqat optik
xususiyatlardan foydalanadi.

### Har bir qiymatning analitik maqomi

| Qiymat | Manba | Maqomi | Miqdoriy tavsiyada ishlatiladimi |
|---|---|---|---|
| **Olsen-P (mg/kg)** | AgroIQ kolorimetrik analizatori | AI bahosi | ✅ **Ha — asosiy manba** |
| **Sensor P indikatori** | Universal sensor | Empirik indikator | ❌ Yo'q — faqat moslik tekshiruvi |
| **pH, EC, namlik, harorat** | Universal sensor | To'g'ridan-to'g'ri o'lchov | ✅ Sharoit tuzatishi sifatida |
| **Azot indikatori (N)** | Universal sensor | Kalibrlanmagan skrining | ❌ Yo'q — faqat sifatiy |
| **Kaliy indikatori (K)** | Universal sensor | Kalibrlanmagan skrining | ❌ Yo'q — faqat sifatiy |

### Nima uchun sensor P Olsen-P ni almashtira olmaydi

Arzon NPK sensorlari odatda tuproq eritmasining elektr xossalarini o'lchaydi va
ishlab chiqaruvchining empirik formulasi orqali "fosfor" qiymatini chiqaradi. Bu
qiymat:

- standartlashtirilgan kimyoviy ekstraksiyaga (Olsen, Bray, Machigin) asoslanmaydi;
- tuproq namligi, harorati va sho'rlanishiga kuchli bog'liq;
- karbonatli tuproqlarda ayniqsa ishonchsiz;
- o'simlik o'zlashtira oladigan fosfor bilan izchil korrelyatsiyaga ega emas.

Shu sababli platforma bu ikki qiymatni **hech qachon o'rtachalamaydi**. Ular farq
qilsa, `P_INDICATORS_DISAGREE` bayrog'i qo'yiladi va kolorimetrik natija asosiy
bo'lib qoladi.

### Moslik tekshiruvi mezoni

Tafovut faqat **ikkala shart** bajarilganda belgilanadi (bu past
konsentratsiyalarda soxta ogohlantirishlarning oldini oladi):

```
nisbiy_farq > 0.60   VA   absolyut_farq > 6.0 mg/kg
```

Chegaralar: `config/sensor_thresholds.json` → `phosphorus_agreement`.

### N va K uchun kalibrlash talablari

Miqdoriy azot/kaliy tavsiyalari `config/nutrient_calibration.json` da
`quantitative_enabled: false` bilan **o'chirilgan**. Ularni yoqish uchun:

1. Kamida **150 juft** sensor–laboratoriya namunasi;
2. Kamida **3 hudud** va turli tuproq tiplaridan;
3. Mustaqil test to'plamida **R² ≥ 0.70**;
4. Akkreditatsiyalangan laboratoriya usuli (nitrat azot / almashinuvchi K₂O);
5. Mahalliy agrokimyo mutaxassisi tasdig'i.

Ushbu shartlar bajarilmaguncha platforma faqat sifatiy baho beradi:
*ehtimol past · qoniqarli · ehtimol yuqori* — har birida laboratoriya tasdig'i tavsiyasi bilan.

### Ishonchlilikni hisoblash (v0.2.0)

Umumiy tavsiya ishonchliligi model ishonchidan boshlanadi va sifat bayroqlari
bo'yicha konservativ pasaytiriladi. **Muhim loyihaviy qaror:** agronomik sharoit
bayroqlari (masalan `PH_HIGH`) ishonchlilikni pasaytirmaydi, chunki ular hisob-kitobda
tuzatish koeffitsienti orqali **allaqachon hisobga olingan**. Ishonchlilik faqat
o'lchov sifati bilan bog'liq muammolar uchun pasaytiriladi:

| Bayroq | Jazo | Sabab |
|---|---|---|
| `COLOR_OUTSIDE_CALIBRATION_RANGE` | −2 | Model validatsiya qilinmagan hududda |
| `SENSOR_DATA_STALE` | −1 | Tuproq holati o'zgargan bo'lishi mumkin |
| `P_INDICATORS_DISAGREE` | −1 | Qurilmalar bir-birini tasdiqlamaydi |
| `EC_HIGH` | −1 | Sho'rlanish oziq o'zlashtirishga tavsiya modeli qamramagan ta'sir qiladi |
| `SENSOR_DATA_MISSING` | −1 | Kontekst to'liq emas |
| `PH_HIGH`, `MOISTURE_LOW`, `DEVICE_ID_MISSING` | 0 | Hisobga olingan yoki o'lchov aniqligiga ta'sir qilmaydi |

### Dastlabki AgroIQ diagnostika indeksi

Natijalar sahifasidagi 0–100 ko'rsatkich **sertifikatlangan tuproq unumdorligi bahosi
EMAS**. U faqat kiritilgan o'lchovlarning maqbul oraliqdan chetlanishini umumlashtiradi.
Mavjud bo'lmagan parametrlar indeksga qo'shilmaydi va og'irliklar qayta
normallashtiriladi. Ilovada har doim ushbu ogohlantirish bilan birga ko'rsatiladi.

### v0.2.0 da nima o'zgarmadi

- Fosfor modeli pipeline'i, xususiyatlari va o'qitish jarayoni;
- Model tanlash mantiqi va metrikalari;
- Noaniqlikni baholash usuli;
- Fosfor sinflari chegaralari.
