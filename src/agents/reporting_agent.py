from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from .base import BaseAgent

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports"))

HIGHLIGHTS_SYSTEM = """You are a PE managing director writing a weekly portfolio summary.
Write 3-5 concise bullet points covering: pipeline progress, portfolio health,
key risks, and recommended actions. Be direct and quantitative where possible."""


class ReportingAgent(BaseAgent):
    """Weekly PDF report generator pulling from all four DB tables."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm=llm)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._chain = self.build_chain()

    def build_chain(self) -> Any:
        return self.llm

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        Expected input: {
          companies: list, pipeline_stages: list,
          portfolio_kpis: list, signals: list
        }
        Returns: {pdf_path: str, highlights: list[str], generated_at: str}
        """
        result = self.generate_report(input)
        pdf_path = self._render_pdf(result)
        return {
            "pdf_path": str(pdf_path),
            "highlights": result.get("highlights", []),
            "generated_at": result.get("generated_at"),
        }

    def generate_report(self, report_data: dict[str, Any]) -> dict[str, Any]:
        """Build the structured report payload."""
        highlights = self.summarize_highlights(report_data)
        companies = report_data.get("companies", [])
        pipeline = report_data.get("pipeline_stages", [])
        kpis = report_data.get("portfolio_kpis", [])
        signals = report_data.get("signals", [])

        # Pipeline funnel stats
        stage_counts: dict[str, int] = {}
        for ps in pipeline:
            stage = ps.get("stage", "unknown") if isinstance(ps, dict) else str(ps)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # Top scored companies
        top_companies = sorted(
            [c for c in companies if isinstance(c, dict) and c.get("score") is not None],
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:10]

        return {
            "highlights": highlights.get("highlights", []),
            "total_companies": len(companies),
            "pipeline_funnel": stage_counts,
            "top_companies": top_companies,
            "recent_signals": signals[:20],
            "kpi_count": len(kpis),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def summarize_highlights(self, report_data: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to generate key highlights for the report."""
        summary = {
            "company_count": len(report_data.get("companies", [])),
            "pipeline_stages": len(report_data.get("pipeline_stages", [])),
            "recent_signals": [
                s.get("summary", "")[:100] if isinstance(s, dict) else str(s)
                for s in report_data.get("signals", [])[:10]
            ],
        }
        user_prompt = (
            f"Weekly portfolio data summary:\n{json.dumps(summary, indent=2)}\n\n"
            "Write the highlights bullets now."
        )
        raw = self._invoke(HIGHLIGHTS_SYSTEM, user_prompt)

        # Parse bullet points
        bullets = [
            line.lstrip("•-* ").strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return {"highlights": bullets[:6]}

    # ------------------------------------------------------------------
    # PDF generation
    # ------------------------------------------------------------------

    def _render_pdf(self, report: dict[str, Any]) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pdf_path = REPORTS_DIR / f"peris_weekly_{date_str}.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        styles = getSampleStyleSheet()
        story: list[Any] = []

        # Title
        title_style = ParagraphStyle(
            "title", parent=styles["Title"], fontSize=20, spaceAfter=6
        )
        story.append(Paragraph("PERIS — Weekly Intelligence Report", title_style))
        story.append(Paragraph(f"Generated: {date_str}", styles["Normal"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.15 * inch))

        # Highlights
        story.append(Paragraph("Key Highlights", styles["Heading2"]))
        for h in report.get("highlights", []):
            story.append(Paragraph(f"• {h}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # Pipeline funnel
        story.append(Paragraph("Pipeline Funnel", styles["Heading2"]))
        funnel = report.get("pipeline_funnel", {})
        if funnel:
            table_data = [["Stage", "Count"]] + [
                [stage, str(count)] for stage, count in sorted(funnel.items())
            ]
            t = Table(table_data, colWidths=[3 * inch, 1.5 * inch])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ])
            )
            story.append(t)
        else:
            story.append(Paragraph("No pipeline data.", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # Top companies table
        story.append(Paragraph("Top Scored Companies", styles["Heading2"]))
        top = report.get("top_companies", [])
        if top:
            table_data = [["Company", "Sector", "Score", "Source"]] + [
                [
                    c.get("name", "")[:30],
                    (c.get("sector") or "")[:20],
                    str(round(c.get("score", 0), 1)),
                    (c.get("source") or "")[:15],
                ]
                for c in top
            ]
            t = Table(table_data, colWidths=[2.5 * inch, 1.5 * inch, 0.8 * inch, 1.2 * inch])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (2, 0), (2, -1), "CENTER"),
                ])
            )
            story.append(t)
        else:
            story.append(Paragraph("No scored companies yet.", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # Recent signals
        story.append(Paragraph("Recent Signals", styles["Heading2"]))
        for sig in report.get("recent_signals", [])[:10]:
            if isinstance(sig, dict):
                stype = sig.get("signal_type", "")
                summary = sig.get("summary", "")[:200]
                story.append(Paragraph(f"[{stype.upper()}] {summary}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # Footer stats
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(
            Paragraph(
                f"Total companies tracked: {report.get('total_companies', 0)} | "
                f"KPIs recorded: {report.get('kpi_count', 0)}",
                styles["Normal"],
            )
        )

        doc.build(story)
        return pdf_path
