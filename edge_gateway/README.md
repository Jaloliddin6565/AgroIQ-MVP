# AgroIQ Edge Gateway

Universal tuproq sensori uchun **lokal, faqat o'qish rejimidagi** REST xizmati.

---

## Nima uchun gateway kerak?

Streamlit Community Cloud'da ishlayotgan ilova foydalanuvchining kompyuteridagi
**RS485 portiga to'g'ridan-to'g'ri ula olmaydi** — bulut serveri boshqa mashinada
turadi. Shu sababli apparat bilan aloqa lokal gateway orqali amalga oshiriladi:

```
Universal tuproq sensori
      │  RS485 / Modbus RTU
      ▼
AgroIQ Edge Gateway  (Raspberry Pi / noutbuk — dala yonida)
      │  REST API (JSON, faqat o'qish)
      ▼
AgroIQ Streamlit platformasi
```

Gateway'ning bog'liqliklari (`requirements-edge.txt`) asosiy ilovadan **ajratilgan** —
bulut deploy'i hech qachon apparat kutubxonalariga bog'liq bo'lmaydi.

---

## Tez ishga tushirish (apparatsiz, mock rejimi)

```bash
pip install -r edge_gateway/requirements-edge.txt
python -m edge_gateway.gateway --mode mock --port 8000
```

Tekshirish:

```bash
curl http://localhost:8000/api/v1/readings/latest
```

Serverni ishga tushirmasdan bitta o'lchovni ko'rish (hech qanday qo'shimcha
kutubxona talab qilmaydi):

```bash
python -m edge_gateway.gateway --print-once
```

---

## Haqiqiy sensor bilan (Modbus RTU)

```bash
python -m edge_gateway.gateway \
  --mode modbus \
  --serial-port /dev/ttyUSB0 \
  --baudrate 4800 \
  --slave-id 1 \
  --device-id SOIL-001
```

Windows uchun `--serial-port COM3`.

> ⚠️ **Registr manzillari ishlab chiqaruvchiga qarab farq qiladi.**
> `modbus_reader.py` dagi `RegisterMap` keng tarqalgan 7-in-1 NPK sensorlari uchun
> namuna qiymatlarga ega. Qurilmangiz hujjatiga ko'ra ularni tekshiring va
> zarur bo'lsa o'zgartiring. Noto'g'ri registr xaritasi noto'g'ri qiymat beradi.

---

## Endpoint'lar

| Metod | Yo'l | Tavsif |
|---|---|---|
| GET | `/health` | Xizmat holati, rejim, Modbus mavjudligi |
| GET | `/api/v1/readings/latest` | Oxirgi validatsiyalangan o'lchov |

### Javob namunasi

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

Sensordan o'qib bo'lmasa, gateway **503** qaytaradi va ishdan chiqmaydi.

---

## AgroIQ platformasiga ulash

1. Gateway'ni ishga tushiring (yuqoridagidek).
2. AgroIQ ilovasida **Qurilmalar** bo'limiga o'ting.
3. Ulanish rejimini **API** ga o'zgartiring.
4. Manzilni kiriting, masalan `http://192.168.1.50:8000/api/v1/readings/latest`.
5. **Ulanishni tekshirish** tugmasini bosing.
6. **Oxirgi ma'lumotni olish** tugmasi bilan o'lchovni yuklang.

Apparatsiz namoyish uchun ilovada `mock://` manzilidan foydalaning — hech qanday
tarmoq so'rovi yuborilmaydi.

---

## Xavfsizlik

- **Faqat o'qish.** MVP da sensorga yozish, registrni o'zgartirish yoki kalibrlash
  buyruqlari **umuman qo'llab-quvvatlanmaydi**.
- **Token saqlanmaydi.** API tokeni AgroIQ ilovasida faqat sessiya xotirasida
  turadi — u manba kodiga, konfiguratsiyaga yoki GitHub'ga yozilmaydi.
- **Lokal tarmoq.** Gateway'ni ochiq internetga chiqarmang. Kerak bo'lsa VPN yoki
  teskari proksi (reverse proxy) orqali himoyalang.
- `config.json` `.gitignore` ga kiritilgan — maxfiy ma'lumotni u yerda saqlamang,
  muhit o'zgaruvchisidan foydalaning.

---

## Fayllar

| Fayl | Vazifasi |
|---|---|
| `gateway.py` | FastAPI xizmati va CLI |
| `modbus_reader.py` | RS485/Modbus RTU o'quvchi (ixtiyoriy) |
| `mock_sensor.py` | Apparatsiz soxta sensor (sutkalik siklga ega) |
| `schemas.py` | Validatsiya sxemalari (platforma bilan mos) |
| `config.example.json` | Namuna konfiguratsiya |
| `requirements-edge.txt` | Ajratilgan bog'liqliklar |

`pymodbus` o'rnatilmagan bo'lsa, `modbus_reader.py` import xatosi bermaydi —
u shunchaki `MODBUS_AVAILABLE = False` deb belgilanadi va mock rejimi ishlaydi.
