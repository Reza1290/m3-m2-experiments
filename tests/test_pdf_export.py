"""Unit tests for src/report/pdf_export.py."""

import numpy as np
import pytest

from src.report.pdf_export import generate_radiology_report_pdf


def test_pdf_report_generation():
    roi = np.random.randint(0, 255, size=(224, 224), dtype=np.uint8)
    pred_res = {
        "class_name": "Contact",
        "confidence": 0.845,
        "triage_status": "Prediction Meets Threshold",
        "class_probabilities": {
            "Contact": 0.845,
            "Independent": 0.082,
            "Resorption": 0.073,
        },
        "diagnostic_reason": "Confidence meets operational threshold.",
    }

    pdf_bytes = generate_radiology_report_pdf(
        case_id="ToothFairy2F_008",
        quadrant="Maxillary Right",
        m2_fdi=17,
        m3_fdi=18,
        dmin_mm=0.300,
        sagittal_slice_idx=291,
        prediction_result=pred_res,
        roi_image=roi,
    )

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
