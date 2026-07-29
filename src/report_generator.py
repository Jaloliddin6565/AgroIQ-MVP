"""PDF hisobot generatori (reportlab).

Hisobot xotirada (BytesIO) yaratiladi va to'g'ridan-to'g'ri yuklab olinadi —
serverda vaqtinchalik fayl saqlanmaydi.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import DISCLAIMER_UZ, PILOT_RECOMMENDATION_NOTICE_UZ, __app_name__
from src.model_inference import PhosphorusEstimate
from src.recommendation_engine import Recommendation

PRIMARY = colors.HexColor("#1B5E20")
ACCENT = colors.HexColor("#0E7C86")
LIGHT = colors.HexColor("#F1F6F2")
BORDER = colors.HexColor("#D5E0D7")
MUTED = colors.HexColor("#5A6B5D")


class ReportError(RuntimeError):
    """PDF yaratishda xatolik yuz berganda ko'tariladi."""


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AgroTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=PRIMARY,
            spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "AgroSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "AgroH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=PRIMARY,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "AgroBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
        ),
        "cell": ParagraphStyle(
            "AgroCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
        ),
        "cell_bold": ParagraphStyle(
            "AgroCellBold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
        ),
        "note": ParagraphStyle(
            "AgroNote",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11.5,
            textColor=MUTED,
        ),
        "badge": ParagraphStyle(
            "AgroBadge",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def _kv_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle], width: float) -> Table:
    data = [
        [Paragraph(key, styles["cell_bold"]), Paragraph(str(value), styles["cell"])]
        for key, value in rows
    ]
    table = Table(data, colWidths=[width * 0.42, width * 0.58], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, BORDER),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ]
        )
    )
    return table


def _footer(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback imzosi
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 12 * mm, f"{__app_name__} — pilot diagnostika hisoboti")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"{doc.page}-bet")
    canvas.setStrokeColor(BORDER)
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.restoreState()


def make_analysis_id(now: datetime | None = None) -> str:
    """Takrorlanmaydigan, o'qishga qulay tahlil identifikatori."""

    now = now or datetime.now()
    return f"AGQ-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"


def build_pdf_report(
    estimate: PhosphorusEstimate,
    recommendation: Recommendation,
    measurements: dict[str, Any],
    explanations: list[str],
    model_info: dict[str, Any] | None = None,
    analysis_id: str | None = None,
    sample_label: str = "Namuna",
) -> bytes:
    """To'liq PDF hisobotni xotirada yaratadi va baytlar sifatida qaytaradi."""

    try:
        buffer = io.BytesIO()
        styles = _styles()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=20 * mm,
            title=f"{__app_name__} hisobot",
            author=__app_name__,
        )
        content_width = doc.width
        now = datetime.now()
        analysis_id = analysis_id or make_analysis_id(now)

        story: list[Any] = []

        # --- Sarlavha ---
        story.append(Paragraph(f"{__app_name__} — tuproq diagnostika hisoboti", styles["title"]))
        story.append(
            Paragraph(
                f"Tahlil ID: <b>{analysis_id}</b> &nbsp;|&nbsp; Sana: {now.strftime('%d.%m.%Y %H:%M')} "
                f"&nbsp;|&nbsp; Namuna: {sample_label}",
                styles["subtitle"],
            )
        )
        story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceAfter=8))

        # --- Asosiy natija bloki ---
        headline = Table(
            [
                [
                    Paragraph(
                        f"<font size=9>Baholangan Olsen-P</font><br/>"
                        f"<font size=20><b>{estimate.olsen_p_mg_kg:.1f}</b></font> "
                        f"<font size=9>mg/kg</font><br/>"
                        f"<font size=8>±{estimate.uncertainty_mg_kg:.1f} mg/kg</font>",
                        styles["cell"],
                    ),
                    Paragraph(
                        f"<font size=9>Fosfor holati</font><br/>"
                        f"<font size=16><b>{estimate.status_label_uz}</b></font><br/>"
                        f"<font size=8>Ishonch: {estimate.confidence_label_uz}</font>",
                        styles["cell"],
                    ),
                    Paragraph(
                        "<font size=9>Tavsiya</font><br/>"
                        "<font size=13><b>Fosfor kerak emas</b></font><br/>"
                        "<font size=8>Zaxira yetarli — bu mavsumda qo'shimcha fosfor "
                        "tavsiya etilmaydi</font>"
                        if recommendation.no_phosphorus_needed
                        else f"<font size=9>Tavsiya etilgan o'g'it</font><br/>"
                        f"<font size=13><b>{recommendation.product_name_uz}</b></font><br/>"
                        f"<font size=8>{recommendation.product_kg_ha_low:.0f}–"
                        f"{recommendation.product_kg_ha_high:.0f} kg/ga</font>",
                        styles["cell"],
                    ),
                ]
            ],
            colWidths=[content_width / 3.0] * 3,
        )
        headline.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(headline)
        story.append(Spacer(1, 6))

        # --- Kiritilgan o'lchovlar ---
        story.append(Paragraph("1. Kiritilgan o'lchovlar", styles["h2"]))
        story.append(
            _kv_table(
                [
                    (
                        "Kolorimetriya (R / G / B)",
                        f"{measurements.get('red', 0):.0f} / {measurements.get('green', 0):.0f} / "
                        f"{measurements.get('blue', 0):.0f}",
                    ),
                    ("Reaksiya vaqti", f"{measurements.get('reaction_time_sec', 0):.0f} sek"),
                    ("Namuna harorati", f"{measurements.get('sample_temperature_c', 0):.1f} °C"),
                    ("Tuproq pH", f"{measurements.get('ph', 0):.2f}"),
                    ("Elektr o'tkazuvchanlik (EC)", f"{measurements.get('ec_ds_m', 0):.2f} dS/m"),
                    ("Namlik", f"{measurements.get('moisture_pct', 0):.1f} %"),
                    ("Ekin", recommendation.crop_name_uz),
                    ("Dala maydoni", f"{recommendation.field_area_ha:g} ga"),
                    ("Maqsadli hosildorlik", f"{recommendation.target_yield_t_ha:g} t/ga"),
                ],
                styles,
                content_width,
            )
        )

        # --- AI natijasi ---
        story.append(Paragraph("2. AI tahlili natijasi", styles["h2"]))
        ai_rows = [
            ("Baholangan Olsen-P", f"{estimate.olsen_p_mg_kg:.2f} mg/kg"),
            ("Noaniqlik (±1σ)", f"±{estimate.uncertainty_mg_kg:.2f} mg/kg"),
            ("Taxminiy 95% oraliq", f"{estimate.ci95_low:.1f} … {estimate.ci95_high:.1f} mg/kg"),
            ("Fosfor holati", estimate.status_label_uz),
            ("Model ishonchliligi", estimate.confidence_label_uz),
            (
                "Kalibrlash oralig'i",
                "Namuna oraliq ichida"
                if estimate.within_calibration
                else "Namuna oraliqdan tashqarida — laboratoriya tasdig'i tavsiya etiladi",
            ),
        ]
        if model_info:
            metrics = model_info.get("metrics", {})
            ai_rows.append(
                (
                    "Model",
                    f"{model_info.get('model_name', '-')} (R² = {metrics.get('r2', 0):.3f}, "
                    f"RMSE = {metrics.get('rmse', 0):.2f} mg/kg)",
                )
            )
            ai_rows.append(
                (
                    "O'quv ma'lumoti",
                    "Demo (sintetik) ma'lumot"
                    if model_info.get("dataset_kind") == "demo"
                    else "Real laboratoriya ma'lumoti",
                )
            )
        story.append(_kv_table(ai_rows, styles, content_width))

        # --- Tavsiya ---
        story.append(Paragraph("3. O'g'itlash tavsiyasi", styles["h2"]))
        if recommendation.no_phosphorus_needed:
            rec_rows = [
                (
                    "Tavsiya",
                    "Ushbu mavsumda fosforli o'g'it berish tavsiya etilmaydi — tuproqdagi "
                    "fosfor zaxirasi yuqori baholandi.",
                ),
                ("Kerakli oziq modda", "0 kg P2O5/ga"),
                ("Keyingi qadam", recommendation.application_method_uz),
                ("Tavsiya ishonchliligi", recommendation.confidence_label_uz),
            ]
        else:
            rec_rows = [
                ("Tavsiya etilgan mahsulot", recommendation.product_name_uz),
                (
                    "Kerakli oziq modda",
                    f"{recommendation.required_p2o5_low:.0f} … "
                    f"{recommendation.required_p2o5_high:.0f} kg P2O5/ga",
                ),
                (
                    "Mahsulot me'yori",
                    f"{recommendation.product_kg_ha_low:.0f} … "
                    f"{recommendation.product_kg_ha_high:.0f} kg/ga",
                ),
                (
                    f"Dala uchun jami ({recommendation.field_area_ha:g} ga)",
                    f"{recommendation.total_product_kg_low:.0f} … "
                    f"{recommendation.total_product_kg_high:.0f} kg "
                    f"(≈ {recommendation.total_bags_low:.0f}–{recommendation.total_bags_high:.0f} "
                    "qop x 50 kg)",
                ),
                ("Qo'llash muddati", recommendation.application_stage_uz),
                ("Qo'llash usuli", recommendation.application_method_uz),
                ("Tavsiya ishonchliligi", recommendation.confidence_label_uz),
            ]
        if recommendation.estimated_cost_uzs_high:
            rec_rows.append(
                (
                    "Taxminiy o'g'it xarajati",
                    f"{recommendation.estimated_cost_uzs_low:,.0f} … "
                    f"{recommendation.estimated_cost_uzs_high:,.0f} so'm".replace(",", " "),
                )
            )
        story.append(_kv_table(rec_rows, styles, content_width))

        # --- Ogohlantirishlar ---
        if recommendation.warnings:
            story.append(Paragraph("4. Agronomik ogohlantirishlar", styles["h2"]))
            warning_rows = [
                (warning.title_uz, warning.message_uz) for warning in recommendation.warnings
            ]
            story.append(_kv_table(warning_rows, styles, content_width))

        # --- Tushuntirish ---
        story.append(Paragraph("5. Nima uchun bu tavsiya berildi?", styles["h2"]))
        for index, reason in enumerate(explanations, start=1):
            story.append(Paragraph(f"{index}. {reason}", styles["body"]))
            story.append(Spacer(1, 2))

        # --- Hisoblash qadamlari ---
        story.append(Paragraph("6. Hisoblash qadamlari (agronom uchun)", styles["h2"]))
        calc_data = [
            [
                Paragraph("Qadam", styles["cell_bold"]),
                Paragraph("Formula", styles["cell_bold"]),
                Paragraph("Natija", styles["cell_bold"]),
            ]
        ]
        for step in recommendation.calculation_steps:
            calc_data.append(
                [
                    Paragraph(str(step["step"]), styles["cell"]),
                    Paragraph(str(step["formula"]), styles["cell"]),
                    Paragraph(str(step["result"]), styles["cell"]),
                ]
            )
        calc_table = Table(
            calc_data,
            colWidths=[content_width * 0.30, content_width * 0.42, content_width * 0.28],
        )
        calc_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(calc_table)

        # --- Ogohlantirish / disclaimer ---
        story.append(Spacer(1, 10))
        disclaimer = Table(
            [
                [
                    Paragraph(
                        f"<b>Muhim eslatma.</b> {DISCLAIMER_UZ}<br/><br/>{PILOT_RECOMMENDATION_NOTICE_UZ}",
                        styles["note"],
                    )
                ]
            ],
            colWidths=[content_width],
        )
        disclaimer.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E6C86E")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(KeepTogether(disclaimer))

        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - foydalanuvchiga traceback ko'rsatilmaydi
        raise ReportError(
            "PDF hisobotni yaratishda xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring."
        ) from exc
