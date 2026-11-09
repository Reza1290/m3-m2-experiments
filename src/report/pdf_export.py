"""Standardized Dental Radiology PDF Report Generator Module.

Compiles case metadata, 3D geometric measurements, ensemble classification
probabilities, and clinical triage status into a clinical PDF document using ReportLab.
"""

from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_radiology_report_pdf(
    case_id: str,
    quadrant: str,
    m2_fdi: int,
    m3_fdi: int,
    dmin_mm: float,
    sagittal_slice_idx: int,
    prediction_result: Dict[str, any],
    roi_image: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """Generate a dental radiology PDF report.

    Args:
        case_id: Patient or Volume identifier.
        quadrant: e.g. "Maxillary Right"
        m2_fdi: 17, 27, 37, 47
        m3_fdi: 18, 28, 38, 48
        dmin_mm: Calculated physical minimum surface distance.
        sagittal_slice_idx: Selected deterministic sagittal plane.
        prediction_result: Dict containing:
            - 'class_name': str
            - 'confidence': float
            - 'triage_status': str
            - 'class_probabilities': Dict[str, float]
            - 'diagnostic_reason': str
        roi_image: 2D image (H, W) or (H, W, 3) to embed in report.
        output_path: Optional path to save the PDF file directly.

    Returns:
        bytes of the generated PDF file.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#718096"),
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("<b>CLINICAL DECISION SUPPORT SYSTEM (CDSS)</b>", title_style))
    elements.append(
        Paragraph(
            "CBCT Third–Second Molar (M3–M2) Relationship & Resorption Analysis Report",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # 2. Case Identification Table
    meta_data = [
        [
            Paragraph("<b>Case / Volume ID:</b>", body_style),
            Paragraph(str(case_id), body_style),
            Paragraph("<b>Target FDI Pair:</b>", body_style),
            Paragraph(f"{m2_fdi} (M2) &amp; {m3_fdi} (M3)", body_style),
        ],
        [
            Paragraph("<b>Quadrant:</b>", body_style),
            Paragraph(str(quadrant), body_style),
            Paragraph("<b>Modality:</b>", body_style),
            Paragraph("CBCT 3D Multiplanar Reconstruction", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # 3. 3D Geometric Evidence & ROI Extraction
    elements.append(Paragraph("<b>1. 3D Mask-Guided Geometric Measurements</b>", h2_style))
    geo_data = [
        [
            Paragraph("<b>Metric</b>", body_style),
            Paragraph("<b>Measured Value</b>", body_style),
            Paragraph("<b>Clinical Interpretation / Role</b>", body_style),
        ],
        [
            Paragraph("Physical Surface Distance (D_min)", body_style),
            Paragraph(f"<b>{dmin_mm:.3f} mm</b>", body_style),
            Paragraph(
                "Anatomical separation cue (supporting geometry; not a standalone classifier).",
                body_style,
            ),
        ],
        [
            Paragraph("Selected Sagittal Slice", body_style),
            Paragraph(f"Slice index <b>{sagittal_slice_idx}</b>", body_style),
            Paragraph("Deterministic contact midpoint plane derived from 3D tooth masks.", body_style),
        ],
        [
            Paragraph("Extracted 2D ROI Dimension", body_style),
            Paragraph("224 × 224 pixels", body_style),
            Paragraph("Interdental contact interface window centered at root-cervical elevation.", body_style),
        ],
    ]
    geo_table = Table(geo_data, colWidths=[160, 110, 270])
    geo_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(geo_table)
    elements.append(Spacer(1, 12))

    # 4. Ensemble Classification & Triage
    elements.append(Paragraph("<b>2. Vision Transformer Ensemble Classification & Triage</b>", h2_style))

    pred_class_name = prediction_result.get("class_name", "Unknown")
    confidence = prediction_result.get("confidence", 0.0)
    triage_status = prediction_result.get("triage_status", "Review Required")
    probs = prediction_result.get("class_probabilities", {})

    status_color = colors.HexColor("#276749") if triage_status == "Prediction Meets Threshold" else colors.HexColor("#C53030")
    status_bg = colors.HexColor("#C6F6D5") if triage_status == "Prediction Meets Threshold" else colors.HexColor("#FED7D7")

    triage_table_data = [
        [
            Paragraph("<b>Ensemble Predicted Class:</b>", body_style),
            Paragraph(f"<b><font size=11 color='#1A365D'>{pred_class_name.upper()}</font></b>", body_style),
            Paragraph("<b>Clinical Triage Status:</b>", body_style),
            Paragraph(f"<b><font color='{status_color.hexval()}'>{triage_status.upper()}</font></b>", body_style),
        ],
        [
            Paragraph("<b>Prediction Confidence:</b>", body_style),
            Paragraph(f"<b>{confidence:.1%}</b> (Threshold: &ge; 70%)", body_style),
            Paragraph("<b>Action Required:</b>", body_style),
            Paragraph(
                "Automated review eligible" if triage_status == "Prediction Meets Threshold" else "Mandatory specialist radiologist review",
                body_style,
            ),
        ],
    ]
    triage_table = Table(triage_table_data, colWidths=[140, 130, 130, 140])
    triage_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), status_bg),
            ("BOX", (0, 0), (-1, -1), 1, status_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(triage_table)
    elements.append(Spacer(1, 8))

    # Probability Distribution Breakdown
    prob_rows = [
        [Paragraph("<b>Class Name</b>", body_style), Paragraph("<b>Posterior Probability</b>", body_style)]
    ]
    for c_name, p_val in probs.items():
        prob_rows.append([Paragraph(c_name, body_style), Paragraph(f"{p_val:.2%}", body_style)])
    prob_table = Table(prob_rows, colWidths=[270, 270])
    prob_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    elements.append(prob_table)
    elements.append(Spacer(1, 12))

    # 5. Visual Snapshot if provided
    if roi_image is not None:
        elements.append(Paragraph("<b>3. Extracted Sagittal ROI Patch Snapshot</b>", h2_style))
        # Encode numpy image to PNG bytes
        if roi_image.max() <= 1.0:
            img_u8 = (roi_image * 255.0).astype(np.uint8)
        else:
            img_u8 = np.clip(roi_image, 0, 255).astype(np.uint8)

        is_success, buffer_img = cv2.imencode(".png", img_u8)
        if is_success:
            img_stream = BytesIO(buffer_img.tobytes())
            rl_img = RLImage(img_stream, width=2.2 * inch, height=2.2 * inch)
            rl_img.hAlign = "CENTER"
            elements.append(rl_img)
            elements.append(Spacer(1, 10))

    # 6. Medical Legal Disclaimer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=6))
    elements.append(
        Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by a research-oriented Clinical Decision Support System (CDSS) "
            "prototype developed at Politeknik Elektronika Negeri Surabaya (PENS). Output is advisory and intended solely "
            "to assist radiologic review. Model predictions do not constitute a definitive medical diagnosis and must be "
            "independently verified by a licensed oral and maxillofacial radiologist or dental practitioner.",
            disclaimer_style,
        )
    )

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
