"""The report as a PDF, drawn with reportlab from the same data the page shows. Landscape A4, standard
fonts only (Latin-1): other characters in a name are replaced with '?'."""

import io

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

from rag_lab.benchmark.report import Chart, Report

MODEL_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
INK, MUTED, RULE = colors.HexColor("#1b1b1a"), colors.HexColor("#6b6a64"), colors.HexColor("#d4d2c8")
BEST = colors.HexColor("#d7f0e5")
HEAD = colors.HexColor("#eceae1")


def _latin1(text) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _escape(text) -> str:
    return _latin1(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], alignment=0, fontSize=22, leading=26, textColor=INK, spaceAfter=4),
        "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=10, leading=13, textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=INK, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9, leading=12, textColor=INK),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=7, leading=9, textColor=INK),
        "note": ParagraphStyle("note", parent=base["Normal"], fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=3),
    }


def _bar_chart(chart: Chart, models: list[str], strategies: list[str], width: float, height: float) -> Drawing:
    """Strategies along the x axis, one bar per model."""
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 10, _latin1(chart.title), fontSize=9, fillColor=INK, fontName="Helvetica-Bold"))
    by = {(v["model"], v["strategy"]): v["value"] for v in chart.values}
    bars = VerticalBarChart()
    bars.x, bars.y = 38, 52
    bars.width, bars.height = width - 50, height - 52 - 38
    bars.data = [[by.get((m, s), 0) or 0 for s in strategies] for m in models]
    bars.categoryAxis.categoryNames = [_latin1(s) for s in strategies]
    bars.categoryAxis.labels.fontSize = 7
    bars.categoryAxis.labels.fontName = "Helvetica"
    bars.categoryAxis.labels.dy = -2
    bars.valueAxis.labels.fontSize = 7
    bars.valueAxis.labels.fontName = "Helvetica"
    bars.barLabelFormat = "%.0f" if chart.fmt == ".0f" else "%.3f" if chart.fmt == ".3f" else "%.1f"
    bars.barLabels.nudge = 6
    bars.barLabels.fontSize = 6
    bars.barLabels.fontName = "Helvetica"
    bars.valueAxis.valueMin = 0
    bars.valueAxis.labelTextFormat = "%.0f" if chart.fmt == ".0f" else "%.2f" if chart.fmt == ".3f" else "%.1f"
    bars.barSpacing = 1
    bars.groupSpacing = 8
    bars.strokeColor = None
    bars.valueAxis.strokeColor = RULE
    bars.categoryAxis.strokeColor = RULE
    for i in range(len(models)):
        bars.bars[i].fillColor = colors.HexColor(MODEL_COLORS[i % len(MODEL_COLORS)])
        bars.bars[i].strokeColor = None
    drawing.add(bars)
    legend = Legend()
    legend.x, legend.y = 38, 4
    legend.alignment = "right"
    legend.fontSize = 7
    legend.fontName = "Helvetica"
    legend.deltax = 52
    legend.dxTextSpace = 4
    legend.columnMaximum = 1
    legend.boxAnchor = "sw"
    legend.colorNamePairs = [
        (colors.HexColor(MODEL_COLORS[i % len(MODEL_COLORS)]), _latin1(m)) for i, m in enumerate(models)
    ]
    drawing.add(legend)
    drawing.add(String(width - 4, 4, _latin1(chart.unit), fontSize=6.5, fillColor=MUTED, textAnchor="end"))
    return drawing


def _table(data: list[list], col_widths=None, font=7.5, head_rows=1) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=head_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), font),
                ("FONTNAME", (0, 0), (-1, head_rows - 1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, head_rows - 1), HEAD),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def report_pdf(report: Report) -> bytes:
    st = _styles()
    buffer = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=_latin1(f"Benchmark report - {report.document['name']}"),
    )
    width = page[0] - 32 * mm
    out = [
        Paragraph("Benchmark report", st["title"]),
        Paragraph(
            _escape(f"{report.document['name']}  |  report {report.report_id}  |  {report.created}  |  {report.status}"),
            st["sub"],
        ),
    ]

    d = report.document
    facts = [
        ["Document", _escape(d["name"]), "Pages", str(d["pages"] or "-")],
        ["Document id", _escape(d["id"]), "Tables", str(d["tables"] if d["tables"] is not None else "-")],
        ["Experiments", f"{len(report.rows)} finished, {len(report.failed)} failed", "Parse time", f"{d['parse_seconds'] or 0:.0f} s"],
    ]
    out.append(_table([[Paragraph(c, st["small"]) for c in row] for row in facts], [30 * mm, 90 * mm, 25 * mm, 40 * mm], head_rows=0))

    out.append(Paragraph("Settings", st["h2"]))
    out.append(_table([[Paragraph(f"<b>{_escape(k)}</b>", st["small"]), Paragraph(_escape(v), st["small"])] for k, v in report.settings], [42 * mm, width - 42 * mm], head_rows=0))

    out.append(Paragraph("Summary", st["h2"]))
    if report.rows:
        data = [report.columns] + report.rows
        table = _table(data, font=8)
        style_cmds = [("ALIGN", (2, 0), (-1, -1), "RIGHT")]
        for row, col in report.best:
            style_cmds += [
                ("BACKGROUND", (col, row + 1), (col, row + 1), BEST),
                ("FONTNAME", (col, row + 1), (col, row + 1), "Helvetica-Bold"),
            ]
        table.setStyle(TableStyle(style_cmds))
        out.append(table)
        out.append(Paragraph("Green: the best value in the column (speed and quality columns).", st["note"]))
    else:
        out.append(Paragraph("No experiment finished.", st["body"]))

    models = sorted({r[0] for r in report.rows}, key=lambda m: [row[0] for row in report.rows].index(m))
    strategies = list(dict.fromkeys(r[1] for r in report.rows))
    if report.rows:
        half = (width - 6 * mm) / 2
        cells = [_bar_chart(c, models, strategies, half, 190) for c in report.charts if c.values]
        rows = [
            Table([cells[i : i + 2] + [""] * (2 - len(cells[i : i + 2]))], colWidths=[half + 3 * mm, half + 3 * mm], hAlign="LEFT")
            for i in range(0, len(cells), 2)
        ]
        if rows:  # the heading stays with the first row of charts
            out.append(KeepTogether([Paragraph("Charts", st["h2"]), rows[0]]))
            out += rows[1:]

    if report.queries:
        q = report.queries
        out.append(Paragraph("Test queries: rank of the first relevant chunk", st["h2"]))
        head = ["Query"] + [_latin1(e["label"]).replace(" ", "\n") for e in q["experiments"]]
        body = [
            [Paragraph(_escape(item["query"]), st["small"])] + [("-" if r is None else str(r)) for r in ranks]
            for item, ranks in zip(q["queries"], q["ranks"])
        ]
        label_width = 70 * mm
        col = (width - label_width) / max(len(q["experiments"]), 1)
        table = _table([head] + body, [label_width] + [col] * len(q["experiments"]), font=6.5)
        table.setStyle(TableStyle([("ALIGN", (1, 0), (-1, -1), "CENTER")]))
        out.append(table)
        out.append(Paragraph("1 is best; a dash means no chunk in the top results contained the snippet.", st["note"]))

    if report.failed:
        out.append(Paragraph("Failed experiments", st["h2"]))
        out.append(_table([[Paragraph(_escape(n), st["small"]), Paragraph(_escape(e)[:300], st["small"])] for n, e in report.failed], [70 * mm, width - 70 * mm], head_rows=0))

    out.append(Paragraph("Notes", st["h2"]))
    out += [Paragraph(_escape(n), st["note"]) for n in report.notes]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8 * mm, _latin1(f"RAG Lab benchmark report {report.report_id}"))
        canvas.drawRightString(page[0] - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(out, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
