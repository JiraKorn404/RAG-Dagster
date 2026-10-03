"""Altair charts in the app's dark style. Each takes a frame with name, model, strategy, value."""

import altair as alt
import pandas as pd
import streamlit as st
import style
from data import model_order, strategy_order


def _finish(chart: alt.Chart, height: int) -> None:
    p = style.palette()
    chart = (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=p["grid"],
            domainColor=p["baseline"],
            tickColor=p["baseline"],
            labelColor=p["ink_2"],
            titleColor=p["muted"],
            labelFontSize=12,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=p["ink_2"], titleColor=p["muted"], orient="top", symbolType="square"
        )
    )
    st.altair_chart(chart, width="stretch", theme=None)


def heatmap(df: pd.DataFrame, fmt: str = ".3f", height: int = 300) -> None:
    """Strategy by model, coloured by value on a one-hue ramp, with the value printed in each cell."""
    p = style.palette()
    models, strategies = model_order(df["model"]), strategy_order(df["strategy"])
    lo, hi = df["value"].min(), df["value"].max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    mid = lo + (hi - lo) * 0.55
    base = alt.Chart(df).encode(
        x=alt.X("model:N", sort=models, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=14)),
        y=alt.Y("strategy:N", sort=strategies, title=None, axis=alt.Axis(labelFontSize=13)),
    )
    cells = base.mark_rect(cornerRadius=8, stroke=p["surface"], strokeWidth=3).encode(
        color=alt.Color(
            "value:Q", scale=alt.Scale(domain=[lo, hi], range=p["ramp"]), legend=None
        ),
        tooltip=[
            alt.Tooltip("name:N", title="Experiment"),
            alt.Tooltip("value:Q", title="Value", format=fmt),
        ],
    )
    labels = base.mark_text(fontWeight=700, fontSize=15).encode(
        text=alt.Text("value:Q", format=fmt),
        color=alt.condition(alt.datum.value > mid, alt.value(p["text_on_high"]), alt.value(p["text_on_low"])),
    )
    _finish(cells + labels, height)


def grouped_bars(df: pd.DataFrame, unit: str, fmt: str = ".3f", height: int = 300) -> None:
    """Strategies on the x axis, one bar per model, in the model's colour."""
    p = style.palette()
    models = model_order(df["model"])
    colors = p["models"][: len(models)]
    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke=p["surface"], strokeWidth=2
        )
        .encode(
            x=alt.X("strategy:N", sort=strategy_order(df["strategy"]), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=models),
            y=alt.Y("value:Q", title=unit),
            color=alt.Color(
                "model:N", scale=alt.Scale(domain=models, range=colors), legend=alt.Legend(title=None)
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Experiment"),
                alt.Tooltip("value:Q", title=unit, format=fmt),
            ],
        )
    )
    _finish(chart, height)
