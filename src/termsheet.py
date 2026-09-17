"""One-page term sheet PDF: what the cover is, what it costs, and on what evidence."""
import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#1F2933")
ACCENT = colors.HexColor("#2C7BB6")
RULE = colors.HexColor("#DED7C8")


def _table(rows, widths):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5A6672")),
        ("TEXTCOLOR", (1, 0), (-1, -1), INK),
        ("FONTNAME", (1, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def build_term_sheet(terms, stats: dict, price: dict, basis: dict, context: dict) -> bytes:
    """Returns PDF bytes. `context` carries the season range, data build date and experience basis."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="msIndex parametric maize cover — term sheet",
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=17, textColor=INK, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#5A6672"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, textColor=ACCENT, spaceBefore=12, spaceAfter=4)
    note = ParagraphStyle("note", parent=styles["Normal"], fontSize=8.2, textColor=colors.HexColor("#5A6672"), leading=11)

    story = [
        Paragraph("msIndex parametric maize cover", h1),
        Paragraph(f"Indicative term sheet · {date.today().isoformat()} · not an offer of insurance", sub),
        Paragraph("Cover", h2),
        _table([
            ["Peril", "Seasonal maize yield shortfall in the South African maize triangle"],
            ["Index", "Predicted provincial yield anomaly against trend, from satellite and weather data"],
            ["Insured unit", f"{terms.parcels} districts, {terms.hectares:,.0f} ha in total"],
            ["Sum insured", f"R {terms.sum_insured_ha:,.0f} per hectare"],
            ["Attachment", f"{terms.attachment:+.0%} yield shortfall (cover starts)"],
            ["Exhaustion", f"{terms.exhaustion:+.0%} yield shortfall (full limit)"],
            ["Limit", f"{terms.limit_share:.0%} of sum insured"],
            ["Minimum payout", f"{terms.min_payout_share:.0%} of sum insured"],
            ["Settlement", "Payment batch generated from the signed decision ledger after season close"],
        ], [38 * mm, 130 * mm]),

        Paragraph("Pricing", h2),
        _table([
            ["Expected loss", f"R {price['expected_loss_ha']:,.0f} / ha  ({stats['expected_loss']:.1%} of cover)"],
            ["Capital charge", f"R {price['capital_charge_ha']:,.0f} / ha"],
            ["Gross premium", f"R {price['gross_premium_ha']:,.0f} / ha   ·   rate on line {price['rate_on_line']:.1%}"],
            ["Book premium", f"R {price['book_premium']:,.0f}"],
            ["Capital held", f"R {price['book_capital']:,.0f} (modelled 1-in-200 tail, TVaR {stats['tvar_995']:.0%})"],
            ["Worst season", f"{stats['worst_observed']:.0%} of cover ({context.get('worst_season_label', '—')})"],
        ], [38 * mm, 130 * mm]),

        Paragraph("Evidence and cover quality", h2),
        _table([
            ["Experience", f"{context.get('basis', '')} · {stats['seasons']} seasons · "
                           f"{stats['payout_seasons']} with a payout"],
            ["Data", context.get("data_note", "")],
            ["Losses paid", f"{basis['paid_when_loss']:.0f}% of district-seasons with a genuine shortfall"],
            ["Median cover", f"{basis['coverage_ratio']:.0%} of the loss suffered"],
            ["Downside basis risk", f"{basis['under_half_covered']:.0f}% of losses covered less than half"],
            ["Payout / loss correlation", f"{basis['payout_loss_corr']:.2f}"],
        ], [38 * mm, 130 * mm]),

        Spacer(1, 10),
        Paragraph(
            "Basis risk: payouts follow the index, not a measured farm yield, so a farmer can suffer a loss the index "
            "does not pay. Yields are published per province, so all districts in a province share one reported yield. "
            "The tail used for capital is modelled from a short history and is indicative. Figures are illustrative "
            "and subject to underwriting, reinsurance terms and regulatory approval.", note),
    ]
    doc.build(story)
    return buffer.getvalue()
