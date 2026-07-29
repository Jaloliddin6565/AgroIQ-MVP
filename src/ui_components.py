"""Streamlit UI komponentlari: CSS, kartalar, grafiklar va logotip.

Barcha vizual elementlar shu yerda jamlangan — `app.py` faqat oqimni boshqaradi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_validation import ASSETS_DIR
from src.feature_engineering import FEATURE_LABELS_UZ

PRIMARY = "#1B5E20"
PRIMARY_SOFT = "#2E7D32"
ACCENT = "#0E7C86"
INK = "#16241A"
MUTED = "#5A6B5D"
SURFACE = "#FFFFFF"
CANVAS = "#F6F9F5"
BORDER = "#DCE7DD"

LEVEL_STYLES: dict[str, dict[str, str]] = {
    "attention": {"bg": "#FDECEA", "border": "#E4A199", "icon": "▲", "color": "#9B2C1F"},
    "caution": {"bg": "#FFF8E1", "border": "#EBD08A", "icon": "◆", "color": "#8A6100"},
    "info": {"bg": "#E8F4F6", "border": "#9FCDD4", "icon": "●", "color": "#0B5C64"},
}

CONFIDENCE_COLORS = {"high": "#27AE60", "medium": "#E9A21B", "low": "#C0392B"}


# ---------------------------------------------------------------------------
# Global uslub
# ---------------------------------------------------------------------------


def inject_css() -> None:
    """Boshqariladigan maxsus CSS — Streamlit'ning standart ko'rinishidan uzoqlashish uchun."""

    st.markdown(
        f"""
        <style>
        :root {{
            --agro-primary: {PRIMARY};
            --agro-accent: {ACCENT};
            --agro-border: {BORDER};
            --agro-muted: {MUTED};
        }}
        .stApp {{
            background: {CANVAS};
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}
        h1, h2, h3, h4 {{
            color: {INK};
            letter-spacing: -0.01em;
        }}
        h1 {{ font-weight: 750; }}
        p, li, label, .stMarkdown {{ color: {INK}; }}

        /* --- Kartalar --- */
        .agro-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 2px rgba(22, 36, 26, 0.04);
            height: 100%;
        }}
        .agro-card h4 {{ margin: 0 0 .35rem 0; font-size: 1.02rem; }}
        .agro-card p {{ margin: 0; color: {MUTED}; font-size: .92rem; line-height: 1.5; }}

        /* --- Metrik kartalar --- */
        .agro-metric {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-left: 5px solid var(--metric-accent, {PRIMARY});
            border-radius: 14px;
            padding: .9rem 1.05rem;
            height: 100%;
        }}
        .agro-metric .label {{
            font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
            color: {MUTED}; font-weight: 600; margin-bottom: .25rem;
        }}
        .agro-metric .value {{
            font-size: 1.72rem; font-weight: 700; color: {INK}; line-height: 1.15;
        }}
        .agro-metric .value .unit {{ font-size: .95rem; font-weight: 600; color: {MUTED}; }}
        .agro-metric .sub {{ font-size: .82rem; color: {MUTED}; margin-top: .2rem; }}

        /* --- Hero --- */
        .agro-hero {{
            background: linear-gradient(135deg, {PRIMARY} 0%, #14634F 55%, {ACCENT} 100%);
            color: #FFFFFF;
            border-radius: 22px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 1.4rem;
        }}
        .agro-hero h1 {{ color: #FFFFFF; font-size: 2.1rem; margin: 0 0 .6rem 0; }}
        .agro-hero p {{ color: rgba(255,255,255,.92); font-size: 1.04rem; max-width: 46rem; margin: 0; }}
        .agro-hero .eyebrow {{
            display: inline-block; background: rgba(255,255,255,.16); color: #fff;
            padding: .28rem .7rem; border-radius: 999px; font-size: .76rem;
            letter-spacing: .08em; text-transform: uppercase; margin-bottom: .9rem; font-weight: 600;
        }}

        /* --- Qadamlar --- */
        .agro-step {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 1.1rem; height: 100%;
        }}
        .agro-step .num {{
            width: 30px; height: 30px; border-radius: 9px; background: {PRIMARY};
            color: #fff; display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: .9rem; margin-bottom: .6rem;
        }}
        .agro-step h5 {{ margin: 0 0 .25rem 0; font-size: .98rem; color: {INK}; }}
        .agro-step p {{ margin: 0; font-size: .87rem; color: {MUTED}; line-height: 1.45; }}

        /* --- Ogohlantirish bloklari --- */
        .agro-alert {{
            border-radius: 12px; padding: .8rem 1rem; margin-bottom: .55rem;
            border: 1px solid; font-size: .92rem; line-height: 1.5;
        }}
        .agro-alert b {{ display: block; margin-bottom: .15rem; }}

        /* --- Nishonlar (badge) --- */
        .agro-badge {{
            display: inline-block; padding: .3rem .75rem; border-radius: 999px;
            font-size: .82rem; font-weight: 700; color: #fff;
        }}
        .agro-pill {{
            display: inline-block; padding: .22rem .6rem; border-radius: 999px;
            background: #EAF3EB; color: {PRIMARY}; font-size: .78rem; font-weight: 600;
            margin: 0 .3rem .3rem 0; border: 1px solid {BORDER};
        }}

        /* --- Banner --- */
        .agro-banner {{
            background: #FFF8E1; border: 1px solid #EBD08A; color: #6B4B00;
            border-radius: 12px; padding: .75rem 1rem; font-size: .88rem;
            line-height: 1.5; margin-bottom: 1rem;
        }}
        .agro-disclaimer {{
            background: {SURFACE}; border: 1px dashed {BORDER}; color: {MUTED};
            border-radius: 12px; padding: .85rem 1rem; font-size: .84rem; line-height: 1.55;
        }}

        /* --- Streamlit elementlarini moslashtirish --- */
        .stButton > button {{
            border-radius: 10px; font-weight: 650; border: 1px solid {BORDER};
            padding: .55rem 1.1rem;
        }}
        .stFormSubmitButton > button {{
            border-radius: 10px; font-weight: 650; padding: .55rem 1.1rem;
        }}
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {{
            background: {PRIMARY}; border-color: {PRIMARY}; color: #FFFFFF;
        }}
        .stButton > button[kind="primary"] p,
        .stFormSubmitButton > button[kind="primary"] p {{ color: #FFFFFF; }}
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {{
            background: {PRIMARY_SOFT}; border-color: {PRIMARY_SOFT}; color: #FFFFFF;
        }}
        .stDownloadButton > button {{
            border-radius: 10px; font-weight: 650; background: {ACCENT};
            color: #fff; border: 1px solid {ACCENT};
        }}
        .stDownloadButton > button p {{ color: #FFFFFF; }}
        div[data-testid="stMetricValue"] {{ color: {INK}; }}
        section[data-testid="stSidebar"] {{
            background: {SURFACE}; border-right: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .35rem; }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0; padding: .45rem 1rem; font-weight: 600;
        }}
        div[data-testid="stExpander"] details {{
            border: 1px solid {BORDER}; border-radius: 12px; background: {SURFACE};
        }}
        @media (max-width: 640px) {{
            .agro-hero {{ padding: 1.6rem 1.2rem; }}
            .agro-hero h1 {{ font-size: 1.5rem; }}
            .block-container {{ padding-left: .8rem; padding-right: .8rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Logotip
# ---------------------------------------------------------------------------


def render_logo(compact: bool = False) -> None:
    """Logotipni ko'rsatadi; fayl bo'lmasa — matnli variant (xatosiz)."""

    logo_path: Path = ASSETS_DIR / "agroiq_logo.png"
    if logo_path.exists():
        try:
            st.image(str(logo_path), width=120 if compact else 180)
            return
        except Exception:  # noqa: BLE001 - rasm buzilgan bo'lsa ham dastur to'xtamasligi kerak
            pass

    size = "1.25rem" if compact else "1.7rem"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.55rem;margin-bottom:.6rem;">
            <div style="width:{'32px' if compact else '42px'};height:{'32px' if compact else '42px'};
                        border-radius:11px;background:linear-gradient(135deg,{PRIMARY},{ACCENT});
                        display:flex;align-items:center;justify-content:center;color:#fff;
                        font-weight:800;font-size:{'0.95rem' if compact else '1.2rem'};">AI</div>
            <div style="line-height:1.1;">
                <div style="font-size:{size};font-weight:800;color:{PRIMARY};letter-spacing:-.02em;">
                    Agro<span style="color:{ACCENT};">IQ</span>
                </div>
                <div style="font-size:.68rem;color:{MUTED};letter-spacing:.1em;text-transform:uppercase;">
                    Soil Intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Kichik komponentlar
# ---------------------------------------------------------------------------


def metric_card(label: str, value: str, unit: str = "", sub: str = "", accent: str = PRIMARY) -> None:
    unit_html = f' <span class="unit">{unit}</span>' if unit else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="agro-metric" style="--metric-accent:{accent};">
            <div class="label">{label}</div>
            <div class="value">{value}{unit_html}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, icon: str = "") -> None:
    icon_html = f'<span style="margin-right:.4rem;">{icon}</span>' if icon else ""
    st.markdown(
        f"""<div class="agro-card"><h4>{icon_html}{title}</h4><p>{body}</p></div>""",
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="agro-step">
            <div class="num">{number}</div>
            <h5>{title}</h5>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_card(level: str, title: str, message: str) -> None:
    style = LEVEL_STYLES.get(level, LEVEL_STYLES["info"])
    st.markdown(
        f"""
        <div class="agro-alert" style="background:{style['bg']};border-color:{style['border']};
             color:{style['color']};">
            <b>{style['icon']} {title}</b>{message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_badge(level: str, label: str) -> str:
    color = CONFIDENCE_COLORS.get(level, MUTED)
    return f'<span class="agro-badge" style="background:{color};">{label}</span>'


def banner(text: str) -> None:
    st.markdown(f'<div class="agro-banner">{text}</div>', unsafe_allow_html=True)


def disclaimer_box(text: str) -> None:
    st.markdown(f'<div class="agro-disclaimer">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Grafiklar
# ---------------------------------------------------------------------------


def phosphorus_gauge(
    value: float,
    thresholds: dict[str, Any],
    uncertainty: float = 0.0,
) -> go.Figure:
    """Fosfor holatini ko'rsatuvchi gorizontal segmentli indikator."""

    classes = thresholds["classes"]
    gauge_max = float(thresholds.get("gauge_max", 60.0))
    display_max = max(gauge_max, value + max(uncertainty, 0) * 2 + 5)

    figure = go.Figure()
    for item in classes:
        low = float(item["min"])
        high = min(float(item["max"]), display_max)
        if high <= low:
            continue
        figure.add_trace(
            go.Bar(
                x=[high - low],
                y=["Olsen-P"],
                base=low,
                orientation="h",
                marker={"color": item.get("color", PRIMARY), "line": {"width": 0}},
                name=str(item["label_uz"]),
                hovertemplate=f"{item['label_uz']}: {low:.0f}–{float(item['max']):.0f} mg/kg<extra></extra>",
                showlegend=True,
            )
        )

    # Baholangan qiymat ko'rsatkichi (shkala ustidagi vertikal chiziq).
    figure.add_shape(
        type="line",
        x0=value,
        x1=value,
        y0=-0.5,
        y1=0.5,
        line={"color": INK, "width": 4},
        layer="above",
    )

    # Noaniqlik oralig'i — shkala ustida alohida "error bar" ko'rinishida,
    # shunda sinf ranglari to'silmaydi.
    if uncertainty > 0:
        low = max(value - uncertainty, 0.0)
        high = min(value + uncertainty, display_max)
        figure.add_shape(
            type="line",
            x0=low, x1=high, y0=0.66, y1=0.66,
            line={"color": INK, "width": 2.5},
            layer="above",
        )
        for cap in (low, high):
            figure.add_shape(
                type="line",
                x0=cap, x1=cap, y0=0.56, y1=0.76,
                line={"color": INK, "width": 2.5},
                layer="above",
            )
        figure.add_annotation(
            x=high, y=0.66, xshift=8, text=f"±{uncertainty:.1f}", showarrow=False,
            xanchor="left", font={"size": 11, "color": MUTED},
        )

    figure.add_annotation(
        x=value,
        y=1.02,
        text=f"<b>{value:.1f} mg/kg</b>",
        showarrow=False,
        font={"size": 15, "color": INK},
    )

    figure.update_layout(
        barmode="stack",
        height=190,
        margin={"l": 10, "r": 10, "t": 30, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "range": [0, display_max],
            "title": "Olsen-P, mg/kg",
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": MUTED},
        },
        yaxis={
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "range": [-0.6, 1.25],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.55,
            "x": 0,
            "traceorder": "normal",
            "font": {"size": 11, "color": MUTED},
        },
        font={"family": "Inter, Segoe UI, sans-serif"},
    )
    return figure


def feature_importance_chart(importances: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Xususiyat muhimligi diagrammasi."""

    top = importances.head(top_n).iloc[::-1]
    labels = [FEATURE_LABELS_UZ.get(name, name) for name in top["feature"]]
    figure = go.Figure(
        go.Bar(
            x=top["importance"],
            y=labels,
            orientation="h",
            marker={"color": ACCENT},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=28 * len(top) + 90,
        margin={"l": 10, "r": 20, "t": 20, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Nisbiy hissa", "showgrid": True, "gridcolor": BORDER},
        yaxis={"showgrid": False},
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def model_comparison_chart(candidates: list[dict[str, Any]], selected: str) -> go.Figure:
    """Model nomzodlarining R² bo'yicha taqqoslanishi."""

    names = [item["name"] for item in candidates]
    scores = [item["r2"] for item in candidates]
    colors = [PRIMARY if name == selected else "#B9CCBB" for name in names]
    figure = go.Figure(
        go.Bar(
            x=names,
            y=scores,
            marker={"color": colors},
            text=[f"{score:.3f}" for score in scores],
            textposition="outside",
            hovertemplate="%{x}<br>R² = %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": "R² (test to'plamida)", "range": [0, 1.08], "gridcolor": BORDER},
        xaxis={"tickfont": {"size": 11}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def prediction_scatter(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Haqiqiy va bashorat qilingan qiymatlar taqqoslanishi."""

    limit = float(max(actual.max(), predicted.max())) * 1.05
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            line={"color": MUTED, "dash": "dash", "width": 1.5},
            name="Ideal chiziq",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker={"color": ACCENT, "size": 8, "opacity": 0.7, "line": {"width": 0}},
            name="Namunalar",
            hovertemplate="Laboratoriya: %{x:.1f}<br>Model: %{y:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Laboratoriya Olsen-P, mg/kg", "gridcolor": BORDER, "range": [0, limit]},
        yaxis={"title": "Model bashorati, mg/kg", "gridcolor": BORDER, "range": [0, limit]},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def color_swatch(red: float, green: float, blue: float) -> None:
    """O'lchangan rangni vizual ko'rsatish."""

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.8rem;">
            <div style="width:64px;height:64px;border-radius:12px;border:1px solid {BORDER};
                        background:rgb({int(red)},{int(green)},{int(blue)});"></div>
            <div style="font-size:.86rem;color:{MUTED};line-height:1.5;">
                O'lchangan rang<br/>
                <b style="color:{INK};">R {int(red)} · G {int(green)} · B {int(blue)}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
