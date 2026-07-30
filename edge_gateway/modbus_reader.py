"""RS485 / Modbus RTU o'quvchi (IXTIYORIY).

`pymodbus` va `pyserial` o'rnatilmagan bo'lsa, bu modul XATO BERMAYDI —
u shunchaki "mavjud emas" deb belgilanadi va gateway mock rejimida ishlaydi.
Shu tufayli asosiy ilova va CI hech qachon apparat bog'liqligiga tayanmaydi.

MVP xavfsizlik qoidasi: FAQAT O'QISH. Sensorga yozish (registrga yozish,
manzilni o'zgartirish, kalibrlash buyruqlari) qo'llab-quvvatlanmaydi.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from edge_gateway.schemas import TASHKENT_TZ, SensorPayload

try:  # pragma: no cover - apparat kutubxonasi CI da mavjud emas
    from pymodbus.client import ModbusSerialClient

    MODBUS_AVAILABLE = True
except Exception:  # noqa: BLE001 - har qanday import muammosi
    ModbusSerialClient = None  # type: ignore[assignment]
    MODBUS_AVAILABLE = False


class ModbusUnavailableError(RuntimeError):
    """Modbus kutubxonasi yoki qurilma mavjud emas."""


@dataclass
class RegisterMap:
    """Sensor registr xaritasi.

    DIQQAT: registr manzillari va ko'paytuvchilar ishlab chiqaruvchiga qarab
    FARQ QILADI. Ushbu qiymatlar keng tarqalgan 7-in-1 NPK sensorlari uchun
    namuna sifatida berilgan va qurilma hujjatiga ko'ra tekshirilishi shart.
    """

    soil_moisture: int = 0x0000
    soil_temperature: int = 0x0001
    ec: int = 0x0002
    ph: int = 0x0003
    nitrogen: int = 0x0004
    phosphorus: int = 0x0005
    potassium: int = 0x0006

    moisture_scale: float = 0.1
    temperature_scale: float = 0.1
    ec_scale: float = 0.001  # µS/cm -> dS/m
    ph_scale: float = 0.1
    npk_scale: float = 1.0


def read_modbus(
    port: str = "/dev/ttyUSB0",
    baudrate: int = 4800,
    slave_id: int = 1,
    timeout: float = 2.0,
    device_id: str = "SOIL-001",
    registers: RegisterMap | None = None,
) -> SensorPayload:
    """Sensordan bitta o'lchovni o'qiydi (faqat o'qish rejimi).

    Raises:
        ModbusUnavailableError: kutubxona yo'q yoki qurilmaga ulanib bo'lmadi.
    """

    if not MODBUS_AVAILABLE:
        raise ModbusUnavailableError(
            "pymodbus o'rnatilmagan. `pip install -r edge_gateway/requirements-edge.txt` "
            "buyrug'ini bajaring yoki mock rejimidan foydalaning."
        )

    registers = registers or RegisterMap()
    client = ModbusSerialClient(
        port=port, baudrate=baudrate, parity="N", stopbits=1, bytesize=8, timeout=timeout
    )
    if not client.connect():
        raise ModbusUnavailableError(f"'{port}' portiga ulanib bo'lmadi.")

    try:
        response = client.read_holding_registers(
            address=registers.soil_moisture, count=7, slave=slave_id
        )
        if response.isError():
            raise ModbusUnavailableError("Sensor xato javob qaytardi.")
        values: list[int] = list(response.registers)
    finally:
        client.close()

    if len(values) < 7:
        raise ModbusUnavailableError("Sensor to'liq bo'lmagan ma'lumot qaytardi.")

    return SensorPayload(
        device_id=device_id,
        timestamp=datetime.now(TASHKENT_TZ),
        source="modbus_gateway",
        soil_moisture_percent=round(values[0] * registers.moisture_scale, 2),
        soil_temperature_c=round(_signed(values[1]) * registers.temperature_scale, 2),
        ec_ds_m=round(values[2] * registers.ec_scale, 3),
        ph=round(values[3] * registers.ph_scale, 2),
        nitrogen_indicator=round(values[4] * registers.npk_scale, 2),
        phosphorus_indicator=round(values[5] * registers.npk_scale, 2),
        potassium_indicator=round(values[6] * registers.npk_scale, 2),
        quality_flags=[],
    )


def _signed(value: int) -> int:
    """16-bitli registrni ishorali songa aylantiradi (manfiy harorat uchun)."""

    return value - 0x10000 if value > 0x7FFF else value


def describe() -> dict[str, Any]:
    """Modul holati haqida ma'lumot (diagnostika uchun)."""

    return {
        "modbus_available": MODBUS_AVAILABLE,
        "write_supported": False,
        "note": "MVP faqat o'qish rejimida ishlaydi — sensorga yozish qo'llab-quvvatlanmaydi.",
    }
