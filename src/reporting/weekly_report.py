from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.db.schema import SessionLocal, init_db
from src.db.queries import get_top_companies, list_signals

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports"))

_SCORE_BUCKETS = [
    ("80–100 (Pursue)", 80, 100, colors.HexColor("#16a34a")),
    ("60–79  (Watch)", 60, 79, colors.HexColor("#d97706")),
    ("40–59  (Pass)", 40, 59, colors.HexColor("#dc2626")),
    ("0–39   (Pass)", 0, 39, colors.HexColor("#991b1b")),
]


def _action(score: float | None) -> str:
    if score is None:
        return "unscored"
    if score >= 80:
        return "pursue"
    if score >= 60:
        return "watch"
    return "pass"


def _score_distribution_drawing(companies: list[Any], width: float = 400, height: float = 120) -> Drawing:
    """Return a simple horizontal bar chart of score buckets."""
    scores = [c.score for c in companies if c.score is not None]
    buckets = []
    for label, lo, hi, colour in _SCORE_BUCKETS:
        count = sum(1 for s in scores if lo <= s <= hi)
        buckets.append((label, count, colour))

    max_count = max((b[1] for b in buckets), default=1) or 1
    bar_height = 18
    gap = 8
    left_margin = 110
    bar_area = width - left_margin - 20

    d = Drawing(width, height)

    for i, (label, count, colour) in enumerate(buckets):
        y = height - (i + 1) * (bar_height + gap)
        bar_w = (count / max_count) * bar_area if max_count else 0
        # Label
        d.add(String(left_margin - 4, y + 4, label, fontSize=8, fillColor=colors.grey, textAnchor="end"))
        # Bar background
        d.add(Rect(left_margin, y, bar_area, bar_height, fillColor=colors.HexColor("#f3f4f6"), strokeColor=None))
        # Bar fill
        if bar_w > 0:
            d.add(Rect(left_margin, y, bar_w, bar_height, fillColor=colour, strokeColor=None))
        # Count label
        d.add(String(left_margin + bar_w + 4, y + 4, str(count), fontSize=8, fillColor=colors.black))

    return d


def generate_weekly_report() -> str:
    """Generate a PDF report. Returns the path to the saved file."""
    init_db()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pdf_path = REPORTS_DIR / f"PERIS_report_{date_str}.pdf"

    with SessionLocal() as session:
        top_companies = get_top_companies(session, limit=10, min_score=0)
        recent_signals = list_signals(session, limit=30)
        all_companies = get_top_companies(session, limit=500, min_score=0)

    _build_pdf(pdf_path, date_str, top_companies, all_companies, recent_signals)
    return str(pdf_path)


def _build_pdf(
    pdf_path: Path,
    date_str: str,
    top_companies: list[Any],
    all_companies: list[Any],
    recent_signals: list[Any],
) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    cover_title = ParagraphStyle(
        "cover_title",
        parent=styles["Title"],
        fontSize=28,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e3a5f"),
    )
    cover_sub = ParagraphStyle(
        "cover_sub",
        parent=styles["Normal"],
        fontSize=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=6,
    )
    section_head = ParagraphStyle(
        "section_head",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=14,
        spaceAfter=6,
    )

    story: list[Any] = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("PERIS", cover_title))
    story.append(Paragraph("Private Equity Research Intelligence System", cover_sub))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Weekly Intelligence Report", cover_sub))
    story.append(Paragraph(date_str, cover_sub))
    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#1e3a5f"), hAlign="CENTER"))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            f"{len(all_companies)} companies tracked  ·  {len(recent_signals)} recent signals",
            cover_sub,
        )
    )
    story.append(PageBreak())

    # ── Executive Summary — Top 10 Prospects ─────────────────────────────────
    story.append(Paragraph("Executive Summary — Top 10 Prospects", section_head))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1 * inch))

    if top_companies:
        table_data = [["#", "Company", "Sector", "Score", "Action"]]
        for i, c in enumerate(top_companies[:10], 1):
            action = _action(c.score)
            table_data.append([
                str(i),
                (c.name or "")[:35],
                (c.sector or "—")[:20],
                f"{c.score:.0f}" if c.score is not None else "—",
                action.upper(),
            ])
        col_widths = [0.3 * inch, 2.8 * inch, 1.5 * inch, 0.6 * inch, 0.8 * inch]
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (1, 0), (2, -1), "LEFT"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No scored companies yet.", styles["Normal"]))

    story.append(Spacer(1, 0.25 * inch))

    # ── Score Distribution ────────────────────────────────────────────────────
    story.append(Paragraph("Score Distribution", section_head))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_score_distribution_drawing(all_companies, width=420, height=130))
    story.append(Spacer(1, 0.25 * inch))

    # ── Signals Summary ───────────────────────────────────────────────────────
    story.append(Paragraph("Recent Signals Summary", section_head))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1 * inch))

    # Breakdown by signal type
    type_counts: dict[str, int] = {}
    for s in recent_signals:
        t_key = s.signal_type or "unknown"
        type_counts[t_key] = type_counts.get(t_key, 0) + 1

    if type_counts:
        type_table = [["Signal Type", "Count"]] + [
            [k, str(v)] for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        ]
        tt = Table(type_table, colWidths=[3 * inch, 1 * inch])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tt)
        story.append(Spacer(1, 0.15 * inch))

    # Most recent signal summaries
    story.append(Paragraph("Most Recent Signals", styles["Heading3"]))
    for sig in recent_signals[:10]:
        stype = (sig.signal_type or "").upper()
        summary = (sig.summary or "")[:200]
        story.append(
            Paragraph(f"<b>[{stype}]</b> {summary}", styles["Normal"])
        )
        story.append(Spacer(1, 0.04 * inch))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(
        Paragraph(
            f"PERIS Weekly Report · Generated {date_str} · Confidential",
            ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, alignment=TA_CENTER),
        )
    )

    doc.build(story)
