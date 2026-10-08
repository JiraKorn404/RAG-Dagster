"""Shared look for both pages: palette tokens, CSS, and small HTML pieces.

Colours follow the dataviz palette, validated for a dark surface: three model colours that stay
distinguishable for colour-blind readers, and a one-hue blue ramp for magnitude."""

import html

import streamlit as st

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"

# One colour per model, in order of model size. The first three are validated for all-pairs use.
MODEL_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
# One-hue ramp for magnitude (heatmaps): low values recede toward the surface, high values glow.
RAMP = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

_CSS = f"""
<style>
.stApp {{
  background: radial-gradient(1100px 480px at 12% -8%, rgba(57,135,229,.20), transparent 60%),
              radial-gradient(900px 420px at 95% 0%, rgba(25,158,112,.12), transparent 55%), {PAGE};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid rgba(255,255,255,.08); }}

.hero-title {{
  font-size: 2.5rem; font-weight: 750; letter-spacing: -0.02em; margin: 0; line-height: 1.1;
  background: linear-gradient(90deg, #6da7ec 0%, #3987e5 40%, #199e70 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-fill-color: transparent;
}}
.hero-sub {{ color: {INK_2}; margin: .7rem 0 1.4rem 0; font-size: 1.02rem; }}

.card {{
  background: {SURFACE}; border: 1px solid rgba(255,255,255,.10); border-radius: 16px;
  padding: 16px 20px; height: 100%;
}}
.card .label {{ color: {INK_2}; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; }}
.card .value {{ color: {INK}; font-size: 2.1rem; font-weight: 700; line-height: 1.25; margin-top: 4px; }}
.card .sub {{ color: {MUTED}; font-size: .86rem; margin-top: 2px; }}
.card.accent {{ border-color: rgba(57,135,229,.55); box-shadow: 0 0 0 1px rgba(57,135,229,.15), 0 8px 30px rgba(57,135,229,.12); }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {SURFACE}; border: 1px solid rgba(255,255,255,.10) !important; border-radius: 16px;
}}
button[data-baseweb="tab"] {{ font-weight: 600; }}

.section-title {{ color: {INK}; font-weight: 650; font-size: 1.05rem; margin: .2rem 0 .1rem 0; }}
.section-note {{ color: {MUTED}; font-size: .85rem; margin-bottom: .4rem; }}

.model-head {{ display: flex; align-items: center; gap: 10px; margin: 1.4rem 0 .6rem 0; }}
.model-head .dot {{ width: 12px; height: 12px; border-radius: 4px; }}
.model-head .name {{ color: {INK}; font-weight: 700; font-size: 1.15rem; }}
.model-head .meta {{ color: {MUTED}; font-size: .86rem; }}

.exp-head {{ color: {INK}; font-weight: 650; margin-bottom: 2px; }}
.exp-meta {{ color: {MUTED}; font-size: .78rem; margin-bottom: 10px; }}

.hit {{
  background: {PAGE}; border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
  padding: 10px 12px; margin-bottom: 10px;
}}
.hit-top {{ display: flex; align-items: center; gap: 8px; }}
.rank {{
  background: {GRID}; color: {INK}; border-radius: 6px; font-weight: 700; font-size: .78rem;
  min-width: 24px; text-align: center; padding: 1px 6px;
}}
.sim {{ color: {INK}; font-weight: 650; font-size: .92rem; font-variant-numeric: tabular-nums; }}
.bar {{ flex: 1; height: 6px; background: {GRID}; border-radius: 3px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; border-radius: 3px; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 6px 0; }}
.badge {{ font-size: .7rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid; }}
.badge.text {{ color: #6da7ec; border-color: rgba(109,167,236,.45); background: rgba(57,135,229,.10); }}
.badge.table {{ color: #e8a24b; border-color: rgba(232,162,75,.45); background: rgba(217,89,38,.12); }}
.badge.picture {{ color: #e58bb0; border-color: rgba(213,81,129,.45); background: rgba(213,81,129,.12); }}
.hit img.picture {{ display: block; max-width: 100%; border-radius: 8px; margin: 6px 0 8px 0; }}
.badge.page {{ color: {INK_2}; border-color: rgba(255,255,255,.18); }}
.badge.agree {{ color: #4cc9a0; border-color: rgba(76,201,160,.45); background: rgba(25,158,112,.12); }}
.badge.solo {{ color: {MUTED}; border-color: rgba(255,255,255,.12); }}
.heading {{ color: {INK_2}; font-size: .78rem; margin-bottom: 4px; }}
.snippet {{ color: {INK_2}; font-size: .82rem; line-height: 1.4; }}
.hit details summary {{ color: {MUTED}; font-size: .76rem; cursor: pointer; margin-top: 6px; }}
.hit details pre {{
  white-space: pre-wrap; color: {INK_2}; font-size: .78rem; background: {SURFACE};
  border-radius: 8px; padding: 8px; margin-top: 6px;
}}
.hit code {{ font-size: .74rem; color: #9ec5f4; }}

.doc {{
  background: {PAGE}; border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 14px 16px;
  white-space: pre-wrap; word-break: break-word; line-height: 1.7; font-size: .86rem; color: {INK};
  max-height: 72vh; overflow-y: auto;
}}
.doc .gap {{ color: {MUTED}; }}
.doc .c0 {{ background: rgba(57,135,229,.18); }}
.doc .c1 {{ background: rgba(25,158,112,.18); }}
.doc .ov {{ background: rgba(201,133,0,.42); }}
.doc .ap {{ background: rgba(255,255,255,.06); outline: 1px dashed rgba(255,255,255,.30); }}
.doc .cut {{
  display: inline-block; font-size: .68rem; font-weight: 700; color: {INK}; background: {GRID};
  border-radius: 6px; padding: 0 7px; margin: 0 5px 0 2px; white-space: nowrap; vertical-align: 1px;
  border-left: 3px solid #3987e5;
}}
.doc .cut-1 {{ border-left-color: #199e70; }}
.doc .cut-ov {{ border-left-color: #c98500; }}
.doc .cut-ap {{ border-left-color: #c3c2b7; }}
.doc .cut-none {{ border-left-color: {MUTED}; color: {MUTED}; }}
.doc-keys {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 .6rem 0; font-size: .78rem; color: {INK_2}; }}
.doc-key span {{ border-radius: 3px; }}
.doc-key .c0 {{ background: rgba(57,135,229,.18); }}
.doc-key .c1 {{ background: rgba(25,158,112,.18); }}
.doc-key .ov {{ background: rgba(201,133,0,.42); }}
.doc-key .ap {{ background: rgba(255,255,255,.06); outline: 1px dashed rgba(255,255,255,.30); }}
.doc-key .gap {{ background: {GRID}; }}
.hit pre.chunktext {{
  white-space: pre-wrap; word-break: break-word; margin: 6px 0 0 0; padding: 0; background: transparent;
  color: {INK_2}; font-size: .82rem; line-height: 1.45; font-family: inherit;
}}
.hit .dim {{ color: {MUTED}; }}
</style>
"""


def palette() -> dict:
    """The colours the Altair charts use (ui/charts.py reads this rather than the constants above, so
    the chart colours can be changed in one place)."""
    return {
        "surface": SURFACE,
        "grid": GRID,
        "baseline": BASELINE,
        "ink_2": INK_2,
        "muted": MUTED,
        "models": MODEL_COLORS,
        "ramp": RAMP,
        "text_on_high": "#0b0b0b",  # value labels on the bright end of the heatmap ramp
        "text_on_low": "#ffffff",
    }


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero-title">{html.escape(title)}</div><p class="hero-sub">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    cls = "card accent" if accent else "card"
    size = "2.1rem" if len(value) <= 8 else "1.55rem"  # long names (experiment ids) stay on one line
    return (
        f'<div class="{cls}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value" style="font-size:{size}">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>'
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>'
        + (f'<div class="section-note">{html.escape(note)}</div>' if note else ""),
        unsafe_allow_html=True,
    )
