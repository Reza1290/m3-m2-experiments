# M3–M2 Relationship Classification: Model Experiments & Research Pipeline

Modul eksperimen riset untuk klasifikasi relasi molar ketiga dan kedua (M3–M2) pada CBCT menggunakan Mask-Guided Vision Transformer Ensemble dan Clinical Triage Gate.

## Struktur Direktori
- `src/`: Core library (loader CBCT, FDI presence filter, KD-Tree $D_{\min}$ preprocessor, Vision Transformer backbones, Stage-2 CEJ classifier, hierarchical pipeline, radiology PDF export).
- `scripts/`: Script eksekusi untuk audit dataset, cropping, training stage-1 & stage-2, dan evaluasi.
- `notebooks/`: Jupyter Notebook untuk eksplorasi analitik jarak spasial.
- `reports/`: Hasil metriks kuantitatif, tabel evaluasi, dan kurva ROC/PR.
- `tests/`: Unit test suite Pytest.
- `data/`: NTFS Junction ke `../../data` (0 byte extra disk).
- `weights/`: NTFS Junction ke `../../weights`.

## Menjalankan Pengujian
```powershell
pytest tests -v
```
