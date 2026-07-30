"""AgroIQ Edge Gateway — lokal, faqat o'qish uchun REST xizmati.

Sensorga yaqin joyda (masalan Raspberry Pi yoki dala noutbukida) ishlaydi va
RS485/Modbus ma'lumotini AgroIQ platformasi o'qiy oladigan JSON ko'rinishiga
aylantiradi.

Ishga tushirish (mock rejimi — apparat shart emas):

    pip install -r edge_gateway/requirements-edge.txt
    python -m edge_gateway.gateway --mode mock --port 8000

Modbus rejimi (haqiqiy sensor bilan):

    python -m edge_gateway.gateway --mode modbus --serial-port /dev/ttyUSB0

Endpoint'lar:
    GET /health                     — xizmat holati
    GET /api/v1/readings/latest     — oxirgi o'lchov

XAVFSIZLIK: barcha endpoint'lar FAQAT O'QISH uchun. Sensorga yozish yoki
kalibrlash buyruqlari MVP da umuman qo'llab-quvvatlanmaydi.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from edge_gateway.mock_sensor import read_mock
from edge_gateway.schemas import HealthResponse, SensorPayload

DEFAULT_PORT = 8000


class GatewayState:
    """Gateway ish rejimi va sozlamalari."""

    def __init__(
        self,
        mode: str = "mock",
        device_id: str = "SOIL-001",
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 4800,
        slave_id: int = 1,
    ) -> None:
        self.mode = mode
        self.device_id = device_id
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.slave_id = slave_id

    def read(self) -> SensorPayload:
        """Joriy rejimga mos ravishda o'lchov oladi.

        Modbus rejimida xatolik yuz bersa, gateway ishdan chiqmaydi — xato
        yuqoriga uzatiladi va HTTP 503 qaytariladi.
        """

        if self.mode == "modbus":
            from edge_gateway.modbus_reader import read_modbus

            return read_modbus(
                port=self.serial_port,
                baudrate=self.baudrate,
                slave_id=self.slave_id,
                device_id=self.device_id,
            )
        return read_mock(device_id=self.device_id)

    def health(self) -> HealthResponse:
        try:
            from edge_gateway.modbus_reader import MODBUS_AVAILABLE
        except Exception:  # noqa: BLE001
            MODBUS_AVAILABLE = False  # noqa: N806
        return HealthResponse(
            status="ok",
            mode=self.mode,
            device_id=self.device_id,
            modbus_available=bool(MODBUS_AVAILABLE),
        )


def build_app(state: GatewayState) -> Any:
    """FastAPI ilovasini quradi (FastAPI o'rnatilgan bo'lsa)."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - ixtiyoriy bog'liqlik
        raise SystemExit(
            "FastAPI o'rnatilmagan. Bajaring: "
            "pip install -r edge_gateway/requirements-edge.txt"
        ) from exc

    app = FastAPI(
        title="AgroIQ Edge Gateway",
        version="0.2.0",
        description="Universal tuproq sensori uchun lokal, faqat o'qish rejimidagi REST xizmati.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return json.loads(state.health().model_dump_json())

    @app.get("/api/v1/readings/latest")
    def latest() -> dict[str, Any]:
        try:
            payload = state.read()
        except Exception as exc:  # noqa: BLE001 - apparat xatolari
            raise HTTPException(
                status_code=503,
                detail=f"Sensordan o'qib bo'lmadi: {exc}",
            ) from exc
        return json.loads(payload.model_dump_json())

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgroIQ Edge Gateway")
    parser.add_argument("--mode", choices=["mock", "modbus"], default="mock")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device-id", default="SOIL-001")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=4800)
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument(
        "--print-once",
        action="store_true",
        help="Serverni ishga tushirmasdan bitta o'lchovni chop etadi (sinov uchun).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = GatewayState(
        mode=args.mode,
        device_id=args.device_id,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        slave_id=args.slave_id,
    )

    if args.print_once:
        try:
            payload = state.read()
        except Exception as exc:  # noqa: BLE001
            print(f"[XATO] {exc}", file=sys.stderr)
            return 1
        print(payload.model_dump_json(indent=2))
        return 0

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - ixtiyoriy bog'liqlik
        print(
            "uvicorn o'rnatilmagan. Bajaring: "
            "pip install -r edge_gateway/requirements-edge.txt",
            file=sys.stderr,
        )
        return 1

    print(f"AgroIQ Edge Gateway — rejim: {args.mode}, qurilma: {args.device_id}")
    print(f"  http://{args.host}:{args.port}/api/v1/readings/latest")
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
