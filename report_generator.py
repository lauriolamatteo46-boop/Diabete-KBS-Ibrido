"""
report_generator.py
===================

Esporta un referto clinico didattico in formato PDF (ReportLab)
con i parametri del paziente, la diagnosi differenziale, i fattori e le raccomandazioni.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT_DIR = os.path.dirname(__file__)


def export_pdf(values: dict[str, Any], result: dict[str, Any]) -> str:
    """Genera e salva il referto PDF nella cartella reports/."""
    folder = os.path.join(ROOT_DIR, "reports")
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"referto_clinico_{timestamp}.pdf")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#12304a"),
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#087e8b"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#222222"),
    )

    story = [
        Paragraph("Diabete KBS Ibrido — Referto Valutazione Clinica", title_style),
        Paragraph(f"Data e ora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", body_style),
        Spacer(1, 10),
    ]

    # Tabella parametri paziente
    rows = [["Parametro Clinico", "Valore Rilevato"]]
    for k, v in values.items():
        val_str = f"{v:.1f}" if isinstance(v, float) else str(v)
        rows.append([str(k), val_str])

    table = Table(rows, colWidths=[240, 240])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ])
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # Esito Diagnostico
    onto = result.get("ontology", {})
    rules = result.get("rule_result", {})
    display_name = result.get("display_name", result.get("label", "N/D"))

    story.append(Paragraph(f"Esito Valutazione: <b>{display_name}</b>", h2_style))
    story.append(
        Paragraph(
            f"<b>Categoria:</b> {onto.get('category', 'N/D')} &nbsp;|&nbsp; "
            f"<b>Livello Gravità:</b> {onto.get('severity', 'N/D').capitalize()} &nbsp;|&nbsp; "
            f"<b>Score di Fusione:</b> {result.get('final_score', 0):.2f}",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    # Inquadramento e Differenziazione
    reason = rules.get("reason", onto.get("description", ""))
    story.append(Paragraph(f"<b>Inquadramento Differenziale:</b> {reason}", body_style))
    story.append(Spacer(1, 6))

    # Fattori rilevati
    factors = rules.get("factors", [])
    factors_str = ", ".join(factors) if factors else "Nessun fattore clinico alterato rilevato."
    story.append(Paragraph(f"<b>Fattori Clinici Attivi:</b> {factors_str}", body_style))
    story.append(Spacer(1, 8))

    # Raccomandazioni
    story.append(Paragraph("Raccomandazioni dell'Ontologia Clinica:", h2_style))
    for rec in onto.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 12))
    warning_style = ParagraphStyle(
        "WarningText",
        parent=body_style,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#888888"),
    )
    story.append(
        Paragraph(
            "Avvertenza: Questo documento è prodotto da un sistema didattico a supporto decisionale "
            "(Ingegneria della Conoscenza) e NON sostituisce in alcun modo la valutazione medica o specialistica.",
            warning_style,
        )
    )

    SimpleDocTemplate(path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story)
    return path
