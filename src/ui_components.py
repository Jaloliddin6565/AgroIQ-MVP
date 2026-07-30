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
            overflow-wrap: normal; word-break: normal; hyphens: none;
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

        /* --- Yuqori gorizontal navigatsiya (v0.2.0) --- */
        .agro-navwrap {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: .35rem .5rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 1px 2px rgba(22,36,26,.05);
        }}
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] {{
            display: flex;
            flex-wrap: wrap;
            gap: .25rem;
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: .4rem .5rem;
            box-shadow: 0 1px 2px rgba(22,36,26,.05);
        }}
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label {{
            margin: 0 !important;
            padding: .5rem .95rem;
            border-radius: 10px;
            border: 1px solid transparent;
            cursor: pointer;
            transition: background .12s ease, color .12s ease;
        }}
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label:hover {{
            background: #EEF4EF;
        }}
        /* Radio doirachasini yashirish — faqat navigatsiya blokida.
           Streamlit tuzilishi: label > div > div > [doiracha, matn] */
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label[data-testid="stRadioOption"] > div > div > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label p {{
            font-weight: 600;
            font-size: .93rem;
            color: {MUTED};
            margin: 0;
            white-space: nowrap;
        }}
        /* Tanlangan bo'lim (Streamlit `data-selected` atributidan foydalanamiz) */
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label[data-selected="true"] {{
            background: {PRIMARY};
            border-color: {PRIMARY};
        }}
        div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
        div[role="radiogroup"] > label[data-selected="true"] p {{
            color: #FFFFFF;
        }}
        @media (max-width: 780px) {{
            div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
            div[role="radiogroup"] > label {{ padding: .45rem .7rem; }}
            div[data-testid="stElementContainer"]:has(.agro-nav-anchor) + div
            div[role="radiogroup"] > label p {{ font-size: .85rem; }}
        }}

        /* --- Qurilma kartalari --- */
        .agro-device {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 1.15rem 1.3rem; height: 100%;
        }}
        .agro-device .dev-head {{
            display: flex; align-items: flex-start; justify-content: space-between;
            gap: .8rem; margin-bottom: .7rem;
        }}
        .agro-device .dev-name {{ font-size: 1.05rem; font-weight: 700; color: {INK}; }}
        .agro-device .dev-role {{ font-size: .86rem; color: {MUTED}; margin-top: .2rem; line-height: 1.45; }}
        .agro-device .dev-row {{
            display: flex; justify-content: space-between; gap: 1rem;
            padding: .38rem 0; border-bottom: 1px solid #EDF2EE; font-size: .87rem;
        }}
        .agro-device .dev-row:last-child {{ border-bottom: none; }}
        .agro-device .dev-key {{ color: {MUTED}; }}
        .agro-device .dev-val {{ font-weight: 650; color: {INK}; text-align: right; }}

        .agro-status {{
            display: inline-flex; align-items: center; gap: .35rem;
            padding: .22rem .6rem; border-radius: 999px;
            font-size: .76rem; font-weight: 700; white-space: nowrap;
        }}
        .agro-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

        /* --- Manba nishonlari (provenance) --- */
        .agro-src {{
            display: inline-block; padding: .2rem .58rem; border-radius: 7px;
            background: #EAF3EB; color: {PRIMARY}; border: 1px solid {BORDER};
            font-size: .74rem; font-weight: 650; margin: 0 .3rem .3rem 0;
        }}

        /* --- Arxitektura diagrammasi --- */
        .agro-arch {{
            display: flex; flex-wrap: wrap; align-items: center; gap: .4rem;
            background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: .9rem 1rem;
        }}
        .agro-arch .node {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 9px;
            padding: .45rem .7rem; font-size: .82rem; font-weight: 600; color: {INK};
        }}
        .agro-arch .node.accent {{ background: {PRIMARY}; color: #fff; border-color: {PRIMARY}; }}
        .agro-arch .arrow {{ color: {MUTED}; font-weight: 700; }}

        /* --- Bo'lim sarlavhalari --- */
        .agro-section {{
            display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .7rem 0;
        }}
        .agro-section .bar {{ width: 4px; height: 22px; background: {PRIMARY}; border-radius: 3px; }}
        .agro-section .txt {{ font-size: 1.06rem; font-weight: 700; color: {INK}; }}
        .agro-section .hint {{ font-size: .82rem; color: {MUTED}; font-weight: 500; }}

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


# ---------------------------------------------------------------------------
# v0.2.0 komponentlari
# ---------------------------------------------------------------------------


def top_navigation(pages: list[str], key: str = "nav") -> str:
    """Yuqori gorizontal navigatsiya paneli.

    Native `st.radio(horizontal=True)` ustiga CSS qo'llanadi — uchinchi tomon
    kutubxonasi ishlatilmaydi, shuning uchun Streamlit yangilanishlarida barqaror.
    Tanlangan bo'lim aniq ajratib ko'rsatiladi va tor ekranda qatorlar bo'linadi.
    """

    st.markdown('<span class="agro-nav-anchor"></span>', unsafe_allow_html=True)
    return st.radio(
        "Bo'limlar",
        pages,
        key=key,
        horizontal=True,
        label_visibility="collapsed",
    )


def section_header(title: str, hint: str = "") -> None:
    """Bo'lim sarlavhasi (chap tomonda rangli chiziq bilan)."""

    hint_html = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="agro-section"><span class="bar"></span>'
        f'<span class="txt">{title}</span>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def status_pill(state: str, label: str) -> str:
    """Ulanish holati nishoni (HTML matn sifatida qaytaradi)."""

    palette = {
        "online": ("#E6F4EA", "#1B6E3C", "#27AE60"),
        "demo": ("#E8F4F6", "#0B5C64", "#0E7C86"),
        "manual": ("#FFF8E1", "#7A5800", "#E9A21B"),
        "offline": ("#FDECEA", "#9B2C1F", "#C0392B"),
    }
    background, text, dot = palette.get(state, palette["offline"])
    return (
        f'<span class="agro-status" style="background:{background};color:{text};">'
        f'<span class="agro-dot" style="background:{dot};"></span>{label}</span>'
    )


def device_card(
    name: str,
    role: str,
    state: str,
    state_label: str,
    rows: list[tuple[str, str]],
) -> None:
    """Qurilma holati kartasi."""

    row_html = "".join(
        f'<div class="dev-row"><span class="dev-key">{key}</span>'
        f'<span class="dev-val">{value}</span></div>'
        for key, value in rows
    )
    st.markdown(
        f"""
        <div class="agro-device">
            <div class="dev-head">
                <div>
                    <div class="dev-name">{name}</div>
                    <div class="dev-role">{role}</div>
                </div>
                {status_pill(state, state_label)}
            </div>
            {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def architecture_row(nodes: list[str], accent_last: bool = True) -> None:
    """Bir qatorli arxitektura diagrammasi (matn tugunlari va strelkalar)."""

    parts: list[str] = []
    for index, node in enumerate(nodes):
        is_last = index == len(nodes) - 1
        css = "node accent" if (accent_last and is_last) else "node"
        parts.append(f'<span class="{css}">{node}</span>')
        if not is_last:
            parts.append('<span class="arrow">→</span>')
    st.markdown(f'<div class="agro-arch">{"".join(parts)}</div>', unsafe_allow_html=True)


def source_badges(sources: list[str], labels: dict[str, str]) -> None:
    """Ma'lumot manbalari nishonlari."""

    if not sources:
        return
    badges = "".join(
        f'<span class="agro-src">{labels.get(source, source)}</span>' for source in sources
    )
    st.markdown(badges, unsafe_allow_html=True)


def soil_condition_chart(interpretations: list[dict[str, Any]]) -> go.Figure:
    """Tuproq sharoiti — gorizontal holat chiziqlari.

    Har bir parametr o'zining maqbul oralig'iga nisbatan ko'rsatiladi. Bu
    ilmiy jihatdan asossiz yagona "tuproq salomatligi bali"dan ko'ra halolroq.
    """

    labels = [item["label_uz"] for item in interpretations][::-1]
    values = [item["normalized"] for item in interpretations][::-1]
    colors = [item["color"] for item in interpretations][::-1]
    texts = [item["display"] for item in interpretations][::-1]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[100] * len(labels),
            y=labels,
            orientation="h",
            marker={"color": "#EDF2EE"},
            hoverinfo="skip",
            showlegend=False,
            width=0.55,
        )
    )
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=texts,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
            showlegend=False,
            width=0.55,
        )
    )
    figure.update_layout(
        barmode="overlay",
        height=52 * len(labels) + 70,
        margin={"l": 10, "r": 90, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 100], "showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showgrid": False, "tickfont": {"size": 12, "color": INK}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def diagnostic_index_gauge(index: dict[str, Any]) -> go.Figure:
    """Dastlabki diagnostika indeksi ko'rsatkichi (sertifikatlangan baho EMAS)."""

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(index["score"]),
            number={"suffix": "", "font": {"size": 34, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": index.get("color", PRIMARY), "thickness": 0.7},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#FBE4E0"},
                    {"range": [40, 60], "color": "#FDF0DC"},
                    {"range": [60, 80], "color": "#EDF6E3"},
                    {"range": [80, 100], "color": "#E1F2E6"},
                ],
            },
        )
    )
    figure.update_layout(
        height=210,
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def phosphorus_comparison_chart(
    colorimetric: float, sensor_value: float, agrees: bool
) -> go.Figure:
    """Ikki fosfor manbasini YONMA-YON ko'rsatadi (birlashtirmasdan)."""

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["AgroIQ kolorimetrik<br>(Olsen-P, asosiy)", "Universal sensor<br>(P indikatori)"],
            y=[colorimetric, sensor_value],
            marker={"color": [PRIMARY, "#9AA69C" if not agrees else ACCENT]},
            text=[f"{colorimetric:.1f}", f"{sensor_value:.1f}"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.1f} mg/kg<extra></extra>",
            width=0.5,
        )
    )
    figure.update_layout(
        height=290,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": "mg/kg", "gridcolor": BORDER, "rangemode": "tozero"},
        xaxis={"tickfont": {"size": 11}},
        showlegend=False,
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK},
    )
    return figure


def nutrient_card(assessment: Any) -> None:
    """Azot yoki kaliy uchun sifatiy baho kartasi."""

    value = assessment.value
    display = "—" if value is None else f"{value:.0f}"
    badge = (
        '<span style="font-size:.7rem;background:#FFF8E1;border:1px solid #EBD08A;'
        'color:#7A5800;padding:.12rem .45rem;border-radius:6px;font-weight:700;">'
        "Skrining</span>"
    )
    st.markdown(
        f"""
        <div class="agro-metric" style="--metric-accent:{assessment.color};">
            <div class="label">{assessment.label_uz} {badge}</div>
            <div class="value">{display} <span class="unit">{assessment.unit.split('(')[0].strip()}</span></div>
            <div class="sub"><b>{assessment.status_label_uz}</b> · laboratoriya tasdig'i tavsiya etiladi</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
