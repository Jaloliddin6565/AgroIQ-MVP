# DEPLOY — AgroIQ ni joylashtirish qo'llanmasi

Ushbu hujjat AgroIQ MVP ni **Streamlit Community Cloud**'ga bepul joylashtirish tartibini
tavsiflaydi. Muqobil variantlar (lokal tarmoq, Docker) ham keltirilgan.

---

## 0. Oldindan tekshirish ro'yxati

Joylashtirishdan oldin quyidagilarni tekshiring:

- [ ] `pytest` — barcha testlar muvaffaqiyatli o'tadi;
- [ ] `streamlit run app.py` — ilova lokal xatosiz ochiladi;
- [ ] `models/phosphorus_model.joblib` mavjud va repozitoriyaga commit qilingan;
- [ ] `config/*.json` fayllari to'g'ri JSON formatida;
- [ ] `.streamlit/secrets.toml` **repozitoriyada YO'Q** (`.gitignore` da);
- [ ] kodda hech qanday API kalit, parol yoki maxfiy ma'lumot yo'q;
- [ ] `requirements.txt` barcha kutubxonalarni o'z ichiga oladi.

> **Muhim:** `models/phosphorus_model.joblib` fayli repozitoriyaga **commit qilingan
> bo'lishi kerak**. Streamlit Community Cloud faqat `pip install -r requirements.txt` ni
> bajaradi va o'qitish skriptini avtomatik ishga tushirmaydi. Model fayli ~1–3 MB —
> bu Git uchun muammo emas.

---

## 1. Streamlit Community Cloud (tavsiya etiladi)

### 1.1. Repozitoriyani GitHub'ga yuklash

```bash
git init
git add .
git commit -m "AgroIQ MVP"
git branch -M main
git remote add origin https://github.com/<foydalanuvchi>/<repozitoriya>.git
git push -u origin main
```

Repozitoriya **public** yoki **private** bo'lishi mumkin (Community Cloud ikkalasini ham
qo'llab-quvvatlaydi).

### 1.2. Ilovani yaratish

1. <https://share.streamlit.io> saytiga GitHub akkaunti bilan kiring.
2. **"New app"** tugmasini bosing.
3. Quyidagilarni to'ldiring:

   | Maydon | Qiymat |
   |---|---|
   | Repository | `<foydalanuvchi>/<repozitoriya>` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | ixtiyoriy (masalan `agroiq`) |

4. **"Advanced settings"** → **Python version**: `3.11`.
5. **"Deploy"** tugmasini bosing.

Birinchi build 2–5 daqiqa davom etadi (scikit-learn va reportlab kompilyatsiyasi).

### 1.3. Natija

Ilova quyidagi manzilda ochiladi:

```
https://<app-nomi>.streamlit.app
```

Autentifikatsiya talab qilinmaydi — havolani ochgan har kim ilovadan foydalana oladi.
Bu ko'rik-tanlov namoyishi uchun qulay.

---

## 2. Muhim texnik eslatmalar

### 2.1. Python versiyasi

Loyiha **Python 3.11** uchun mo'ljallangan. Community Cloud'da versiyani "Advanced
settings" bo'limida tanlang. Agar tanlanmasa, platforma standart versiyani ishlatadi va
kutubxona mosligi buzilishi mumkin.

### 2.2. Model fayli

Ilova ishga tushganda `models/phosphorus_model.joblib` ni qidiradi. Fayl topilmasa,
foydalanuvchiga o'zbek tilida tushunarli xabar ko'rsatiladi va ilova qulab tushmaydi:

> "Model fayli topilmadi (models/phosphorus_model.joblib). Iltimos, avval
> `python scripts/train_model.py` buyrug'ini bajaring."

Shuning uchun modelni **lokal o'qitib, commit qiling**:

```bash
python scripts/train_model.py
git add models/phosphorus_model.joblib data/soil_samples.csv data/demo_samples.csv
git commit -m "Model artefaktini yangilash"
git push
```

### 2.3. Xotira cheklovi

Community Cloud bepul rejasida ~1 GB RAM mavjud. AgroIQ ehtiyoji ~150–250 MB —
muammo yo'q. Model `@st.cache_resource`, konfiguratsiyalar esa `@st.cache_data` orqali
keshlanadi, shuning uchun har bir foydalanuvchi uchun qayta yuklanmaydi.

### 2.4. Maxfiy ma'lumotlar

Loyiha **hech qanday maxfiy kalit talab qilmaydi**:

- pullik API yo'q;
- tashqi LLM yo'q;
- ma'lumotlar bazasi yo'q;
- autentifikatsiya yo'q.

`.streamlit/secrets.toml` `.gitignore` da — uni hech qachon commit qilmang.

### 2.5. Mavzu (theme)

Rang sxemasi `.streamlit/config.toml` faylida belgilangan va repozitoriya bilan birga
joylashtiriladi. Qo'shimcha sozlash shart emas.

---

## 3. Yangilanishlarni chiqarish

Community Cloud tanlangan branch'ni kuzatadi. `git push` qilganingizda ilova avtomatik
qayta build qilinadi.

Model yoki konfiguratsiya o'zgargan bo'lsa, keshni tozalash uchun ilova menyusidan
**"Reboot app"** ni tanlang.

---

## 4. Muqobil: lokal tarmoqda ishga tushirish

Namoyish internetsiz o'tkazilsa (masalan, ko'rik-tanlov zalida):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Ilova bir xil Wi-Fi tarmog'idagi qurilmalar uchun quyidagi manzilda ochiladi:

```
http://<kompyuter-IP>:8501
```

> **Tavsiya:** namoyishdan oldin ilovani lokal ishga tushirib, uchta demo ssenariyni
> bir marta bajaring — bu model va konfiguratsiyalarni keshga yuklaydi va namoyish
> paytida javob tezroq bo'ladi.

---

## 5. Muqobil: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

```bash
docker build -t agroiq .
docker run -p 8501:8501 agroiq
```

---

## 6. Ko'p uchraydigan muammolar

| Muammo | Sabab | Yechim |
|---|---|---|
| "Model fayli topilmadi" | `.joblib` commit qilinmagan | Modelni o'qitib, commit qiling (2.2-bo'lim) |
| `ModuleNotFoundError` | `requirements.txt` to'liq emas | Yetishmayotgan kutubxonani qo'shing va push qiling |
| Build juda uzoq | scikit-learn kompilyatsiyasi | Birinchi build uchun 5 daqiqa normal |
| "Konfiguratsiya xatosi" | `config/*.json` da JSON sintaksis xatosi | JSON ni validator orqali tekshiring |
| Grafiklar chizilmaydi | Plotly versiyasi eski | `requirements.txt` da `plotly>=5.20` ekanini tekshiring |
| Ilova "uxlab qoladi" | Community Cloud bepul reja | Havolani ochish ilovani qayta uyg'otadi (~30 sek) |
| Eski natijalar ko'rinadi | Streamlit keshi | Menyudan "Reboot app" |

---

## 7. Namoyishdan oldingi yakuniy tekshiruv

```bash
# 1. Toza muhitda o'rnatish
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Modelni o'qitish
python scripts/train_model.py

# 3. Testlar
pytest

# 4. Ilovani ishga tushirish
streamlit run app.py
```

Brauzerda tekshiring:

- [ ] Bosh sahifa ochiladi, "Tahlilni boshlash" tugmasi ishlaydi;
- [ ] Demo rejimida uchala ssenariy ham turli natija beradi;
- [ ] "Keyingi demo namuna" tugmasi ssenariylarni almashtiradi;
- [ ] Qo'lda kiritish shakli ishlaydi va noto'g'ri qiymatda o'zbekcha xato beradi;
- [ ] Natijalar sahifasida fosfor shkalasi, tavsiya va tushuntirish ko'rinadi;
- [ ] "PDF hisobotni yuklab olish" fayl yuklaydi;
- [ ] "Model va validatsiya" sahifasida R², RMSE, MAE ko'rinadi;
- [ ] Mobil ekran o'lchamida ham interfeys o'qilishi mumkin.
