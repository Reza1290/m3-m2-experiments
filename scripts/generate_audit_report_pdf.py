"""
scripts/generate_audit_report_pdf.py
====================================
Generates a formal, comprehensive PDF report of the Stage 2 Evaluation Audit:
- Executive Summary & Thesis Context
- Stage 1 Discrepancy Root Cause & Case-by-Case Audit
- Recalculated 4-Tier Evaluation Metrics Tables
- Above CEJ Diagnostic Breakdown & Algorithmic Fix
- High-Resolution Multi-View Case Visualizations (Above, At, Below, Independent)
- Leakage Prevention & Reproducibility Verification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 810, "CDSS M3–M2 CBCT Relational Analysis | Stage 2 Evaluation Audit Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 804, 559, 804)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 559, 42)
        
        self.drawString(36, 30, "CONFIDENTIAL RESEARCH AID — PROVISIONAL DEVELOPMENT EVALUATION")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 30, page_str)
        self.restoreState()


def build_audit_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
    )

    story = []

    # Title Block
    story.append(Paragraph("TECHNICAL AUDIT & EVALUATION REPORT", title_style))
    story.append(Paragraph(
        "<b>Investigation of Stage 1 Gating Consistency, 4-Tier Metric Decoupling, and CEJ-Referenced Contact Classification</b><br/>"
        "Project: Clinical Decision Support System for M3–M2 CBCT Relational Analysis<br/>"
        "Author: M. Reza (NRP: 3126640060) | Advisor: Nana Ramadijanti, S.Kom., M.Kom. | PENS 2025",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=10))

    # Executive Summary / Disclaimer Box
    disclaimer_text = (
        "<b>DISCLAIMER & RESEARCH STATUS:</b> This audit report documents experimental verification and code-level "
        "reconciliation for Option B (Hierarchical Deep Learning) and Option C (Hybrid DL + 3D Geometry). "
        "All 4-class labels are provisional heuristic baselines ('heuristic_pseudo'). The system is a research prototype "
        "and does not constitute medical diagnosis without expert clinical adjudication."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, callout_style)]], colWidths=[523])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#3b82f6")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. Root Cause Investigation of Stage 1 Inconsistency", h1_style))
    story.append(Paragraph(
        "In preliminary benchmark reporting, a divergence was noted between Option B (Binary Acc: 75.76%, Macro-F1: 0.7331) "
        "and Option C (Binary Acc: 81.82%, Macro-F1: 0.8068), despite both pipelines allegedly utilizing an identical Stage 1 "
        "Vision Transformer Ensemble. Code inspection of <code>src/models/hierarchical_pipeline.py</code> and per-sample audit of all 66 "
        "validation samples confirmed that <b>Stage 1 ViT Ensemble models, checkpoints, weights, and raw probabilities were 100% identical</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Root Cause:</b> The divergence stemmed exclusively from post-processing rules in Option C. When Stage 1 predicted "
        "<code>Contact</code>, Option C computed 3D KD-Tree surface distance <i>D</i><sub>min</sub>. For cases where <i>D</i><sub>min</sub> &gt; 0.50 mm, "
        "Option C's geometric classifier marked <code>is_contact = False</code> and reverted the prediction to <code>Independent</code> (Class 0). "
        "This rule pruned exactly 4 False Positives of Stage 1, eliminating all false positives in Option C and artificially lifting its "
        "collapsed binary accuracy to 81.82%. Option B possessed no such geometric override, so those 4 cases remained classified as <code>Below_CEJ</code>.",
        body_style
    ))

    # Diff Table
    diff_data = [
        [
            Paragraph("Sample ID", table_header_style),
            Paragraph("Ground Truth", table_header_style),
            Paragraph("Stage 1 P(Contact)", table_header_style),
            Paragraph("Stage 1 Pred", table_header_style),
            Paragraph("Option B Final", table_header_style),
            Paragraph("Option C Final", table_header_style),
            Paragraph("D_min (mm)", table_header_style),
            Paragraph("CEJ Status", table_header_style),
        ],
        [
            Paragraph("ToothFairy2P_025_37_38", table_cell_bold),
            Paragraph("Independent", table_cell_style),
            Paragraph("0.718", table_cell_style),
            Paragraph("Contact (FP)", table_cell_style),
            Paragraph("Below_CEJ (Contact)", table_cell_style),
            Paragraph("<b>Independent</b> (Override)", table_cell_bold),
            Paragraph("0.671", table_cell_style),
            Paragraph("CEJ_DETECTED", table_cell_style),
        ],
        [
            Paragraph("ToothFairy2P_025_47_48", table_cell_bold),
            Paragraph("Independent", table_cell_style),
            Paragraph("0.640", table_cell_style),
            Paragraph("Contact (FP)", table_cell_style),
            Paragraph("Below_CEJ (Contact)", table_cell_style),
            Paragraph("<b>Independent</b> (Override)", table_cell_bold),
            Paragraph("0.600", table_cell_style),
            Paragraph("CEJ_DETECTED", table_cell_style),
        ],
        [
            Paragraph("ToothFairy2P_071_27_28", table_cell_bold),
            Paragraph("Independent", table_cell_style),
            Paragraph("0.814", table_cell_style),
            Paragraph("Contact (FP)", table_cell_style),
            Paragraph("Below_CEJ (Contact)", table_cell_style),
            Paragraph("<b>Independent</b> (Override)", table_cell_bold),
            Paragraph("0.520", table_cell_style),
            Paragraph("CEJ_UNCERTAIN", table_cell_style),
        ],
        [
            Paragraph("ToothFairy2P_091_37_38", table_cell_bold),
            Paragraph("Independent", table_cell_style),
            Paragraph("0.752", table_cell_style),
            Paragraph("Contact (FP)", table_cell_style),
            Paragraph("Below_CEJ (Contact)", table_cell_style),
            Paragraph("<b>Independent</b> (Override)", table_cell_bold),
            Paragraph("0.520", table_cell_style),
            Paragraph("CEJ_DETECTED", table_cell_style),
        ],
    ]
    diff_table = Table(diff_data, colWidths=[105, 60, 55, 55, 75, 75, 45, 53])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(diff_table)
    story.append(Spacer(1, 10))

    # Section 2
    story.append(Paragraph("2. Recalculated Multi-Tier Evaluation Benchmark (N = 66)", h1_style))
    story.append(Paragraph(
        "To ensure fair and rigorous scientific reporting, metrics are now strictly decoupled into four operational tiers. "
        "Option C is evaluated under both <b>Strict Stage 1 Preservation</b> (no binary override; Stage 1 decisions are immutable) "
        "and <b>Hybrid Geometric Pruning</b> (bilateral verification where 3D geometry prunes DL false positives).",
        body_style
    ))

    tier_data = [
        [
            Paragraph("Evaluation Tier", table_header_style),
            Paragraph("Pipeline Configuration", table_header_style),
            Paragraph("Accuracy", table_header_style),
            Paragraph("Bal. Acc", table_header_style),
            Paragraph("Macro-F1", table_header_style),
            Paragraph("Precision", table_header_style),
            Paragraph("Recall", table_header_style),
        ],
        [
            Paragraph("<b>Tier 1: Stage 1 Pure Binary</b>", table_cell_bold),
            Paragraph("Swin+ViT Ensemble (Identical)", table_cell_style),
            Paragraph("75.76%", table_cell_style),
            Paragraph("74.19%", table_cell_style),
            Paragraph("<b>0.7331</b>", table_cell_bold),
            Paragraph("0.7265", table_cell_style),
            Paragraph("0.7671", table_cell_style),
        ],
        [
            Paragraph("<b>Tier 2: Output Binary</b>", table_cell_bold),
            Paragraph("Option B (Hierarchical DL)", table_cell_style),
            Paragraph("75.76%", table_cell_style),
            Paragraph("74.19%", table_cell_style),
            Paragraph("0.7331", table_cell_style),
            Paragraph("0.7265", table_cell_style),
            Paragraph("0.7671", table_cell_style),
        ],
        [
            Paragraph("<i>(Collapsed from 4-Class)</i>", table_cell_style),
            Paragraph("Option C (Strict Preservation)", table_cell_style),
            Paragraph("75.76%", table_cell_style),
            Paragraph("74.19%", table_cell_style),
            Paragraph("0.7331", table_cell_style),
            Paragraph("0.7265", table_cell_style),
            Paragraph("0.7671", table_cell_style),
        ],
        [
            Paragraph("", table_cell_style),
            Paragraph("Option C (Hybrid Geometric Pruning)", table_cell_style),
            Paragraph("81.82%", table_cell_style),
            Paragraph("87.23%", table_cell_style),
            Paragraph("0.8068", table_cell_style),
            Paragraph("0.8065", table_cell_style),
            Paragraph("0.8723", table_cell_style),
        ],
        [
            Paragraph("<b>Tier 3: Final 4-Class</b>", table_cell_bold),
            Paragraph("Option B (Hierarchical DL)", table_cell_style),
            Paragraph("69.70%", table_cell_style),
            Paragraph("39.11%", table_cell_style),
            Paragraph("0.3592", table_cell_style),
            Paragraph("0.3802", table_cell_style),
            Paragraph("0.3911", table_cell_style),
        ],
        [
            Paragraph("<i>(Indep, Above, At, Below)</i>", table_cell_style),
            Paragraph("Option C (Strict Preservation)", table_cell_style),
            Paragraph("<b>74.24%</b>", table_cell_bold),
            Paragraph("<b>59.95%</b>", table_cell_bold),
            Paragraph("<b>0.6376</b>", table_cell_bold),
            Paragraph("0.6559", table_cell_style),
            Paragraph("0.5995", table_cell_style),
        ],
        [
            Paragraph("", table_cell_style),
            Paragraph("Option C (Hybrid Geometric Pruning)", table_cell_style),
            Paragraph("80.30%", table_cell_style),
            Paragraph("65.21%", table_cell_style),
            Paragraph("0.6762", table_cell_style),
            Paragraph("0.6874", table_cell_style),
            Paragraph("0.6521", table_cell_style),
        ],
        [
            Paragraph("<b>Tier 4: Stage 2 Isolated</b>", table_cell_bold),
            Paragraph("Option B Stage 2 DL Classifier", table_cell_style),
            Paragraph("85.11%", table_cell_style),
            Paragraph("33.33%", table_cell_style),
            Paragraph("0.3065", table_cell_style),
            Paragraph("0.2837", table_cell_style),
            Paragraph("0.3333", table_cell_style),
        ],
        [
            Paragraph("<i>(True Contacts Only, N=47)</i>", table_cell_style),
            Paragraph("Option C Stage 2 3D Geometry", table_cell_style),
            Paragraph("<b>74.47%</b>", table_cell_bold),
            Paragraph("<b>64.72%</b>", table_cell_bold),
            Paragraph("<b>0.5649</b>", table_cell_bold),
            Paragraph("0.5648", table_cell_style),
            Paragraph("0.6472", table_cell_style),
        ],
    ]
    tier_table = Table(tier_data, colWidths=[110, 145, 50, 50, 55, 55, 58])
    tier_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0369a1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(tier_table)
    story.append(Spacer(1, 10))

    # Section 3
    story.append(Paragraph("3. Diagnosis of 'Above CEJ' Zero-F1 Failure and Calibrated Fix", h1_style))
    story.append(Paragraph(
        "<b>Why Option B Failed:</b> Contact training distribution is heavily skewed (87.9% Below, 7.8% Above, 4.3% At). "
        "Stage 2 DL was trained for only 3 epochs without class-balanced loss or focal loss, resulting in total mode collapse "
        "to the majority class (predicting <code>Below_CEJ</code> for 100% of contact validation cases).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why Option C Semula Failed:</b> Of the 4 ground-truth Above CEJ validation cases, 2 were gated out by Stage 1 False Negatives. "
        "The remaining 2 cases (<code>ToothFairy2P_395_17_18</code>, &Delta;z = +2.10 mm; <code>ToothFairy2P_540_27_28</code>, &Delta;z = +2.70 mm) "
        "successfully reached Stage 2. However, because 94.4% of Above CEJ cases reside in maxillary molars (FDI 17/27) where scan field-of-view "
        "often truncates at Z = 0, cervical constriction ratio was shallow (&gt; 0.95), triggering <code>CEJ_UNCERTAIN</code>. "
        "A defensive code block in <code>src/geometry/cej_classifier.py</code> blindly forced all uncertain detections to fallback to <code>At_CEJ</code>, "
        "ignoring the strongly coronal +2.1 mm and +2.7 mm elevations.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Applied Algorithmic Fix:</b> <code>GeometricCEJClassifier</code> was updated to evaluate physical signed elevation (&Delta;z) objectively "
        "under threshold &tau;<sub>CEJ</sub> = 1.0 mm even when constriction is shallow, penalizing confidence without corrupting the anatomical class. "
        "Following this fix, Option C correctly classifies both coronal contact cases, boosting Above CEJ F1 from <b>0.0000 to 0.5714</b> "
        "(100% recall on cases reaching Stage 2) and lifting 4-class Macro-F1 from 0.4489 to <b>0.6376</b> (Strict).",
        body_style
    ))

    # Page Break for Visualizations
    story.append(PageBreak())

    # Section 4 - Visualizations
    story.append(Paragraph("4. Multi-View Anatomical Visual Diagnostics", h1_style))
    story.append(Paragraph(
        "Below are publication-grade multi-plane CBCT visualizations generated from the validation set, showing bone-windowed anatomy, "
        "dual-tooth segmentations (M2 lime green, M3 cyan), detected CEJ reference plane (gold dashed line), and physical contact vectors (red/magenta).",
        body_style
    ))

    fig_dir = Path("reports/figures/cej_case_visualizations")
    cases_figs = [
        ("Above CEJ (Coronal Contact) — ToothFairy2P_540 (Tooth 27 & 28)", fig_dir / "cej_case_ToothFairy2P_540_27_28_Above_CEJ.png"),
        ("Above CEJ (Coronal Contact) — ToothFairy2P_395 (Tooth 17 & 18)", fig_dir / "cej_case_ToothFairy2P_395_17_18_Above_CEJ.png"),
        ("At CEJ (Cervical Margin Contact) — ToothFairy2P_091 (Tooth 17 & 18)", fig_dir / "cej_case_ToothFairy2P_091_17_18_At_CEJ.png"),
        ("Below CEJ (Root Dentin Impingement / ERR Risk) — ToothFairy2F_054 (Tooth 37 & 38)", fig_dir / "cej_case_ToothFairy2F_054_37_38_Below_CEJ.png"),
        ("Independent (Intact Periodontal Ligament Gap) — ToothFairy2P_025 (Tooth 37 & 38)", fig_dir / "cej_case_ToothFairy2P_025_37_38_Independent.png"),
    ]

    for title, img_p in cases_figs:
        if img_p.exists():
            story.append(Paragraph(f"<b>{title}</b>", h2_style))
            # Image size to fit A4 width nicely (520 pt width, approx 292 pt height for 16:9)
            story.append(RLImage(str(img_p), width=520, height=292.5))
            story.append(Spacer(1, 10))

    # Section 5 - Leakage & Integrity
    story.append(KeepTogether([
        Paragraph("5. Data Leakage Prevention and Reproducibility Audit", h1_style),
        Paragraph(
            "<b>Patient-Level Separation:</b> ToothFairy2's 287 unique patients were partitioned deterministically using seed 42 into "
            "Train (230 patients, 580 pairs), Validation (29 patients, 66 pairs), and Test (28 patients, 74 pairs). Pairwise set intersection "
            "confirmed <b>zero patient overlap</b> across splits.<br/>"
            "<b>Checkpoint Integrity:</b> Weights for Stage 1 models (<code>swin_s_mha_binary_best.pth</code>, <code>swin_small_binary_best.pth</code>, "
            "<code>vit_base_binary_best.pth</code>) were strictly trained on the 230 training patients and never exposed to validation/test patients.<br/>"
            "<b>Test Set Locking:</b> The entire debugging and threshold calibration procedure was conducted strictly on the validation set. "
            "The final test set (N = 74) remains untouched and locked for definitive scientific reporting.",
            body_style
        ),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=6),
        Paragraph("Antigravity Autonomous Clinical AI Agent • Antigravity 2.0 • PENS Informatics Engineering", subtitle_style)
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built PDF report at: {output_path}")


if __name__ == "__main__":
    out_pdf = Path("reports/STAGE2_EVALUATION_AUDIT_REPORT.pdf")
    build_audit_pdf(out_pdf)
