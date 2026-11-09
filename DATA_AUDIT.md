# DATA_AUDIT.md — Dataset Feasibility & Ground-Truth Provenance Audit

**Project:** Rancang Bangun Sistem Pendukung Keputusan Klinis Klasifikasi Relasi Gigi Molar Ketiga dan Kedua (M3-M2) pada Citra CBCT Menggunakan Mask-Guided Vision Transformer Ensemble dan Clinical Triage Gate  
**Author:** M. Reza (NRP: 3126640060)  
**Date of Audit:** 29 Oktober 2026  
**Auditor:** Autonomous Lead Research & Software Engineer Agent  

---

## 1. Executive Summary & Feasibility Decision

| Pipeline Component | Data Feasibility | Current Blocker / Status | Next Action |
| :--- | :--- | :--- | :--- |
| **Ingestion & Spatial Validation** | **HIGH (100% Ready)** | 480 volumes available in `data/toothfairy2/ToothFairy2_Dataset.zip`. | Extract volume subsets on-demand into `data/raw/` or stream directly. |
| **FDI Tooth Mask Filtering** | **HIGH (100% Ready)** | 480 3D multi-class masks (`.mha`) unpacked in `data/toothfairy2/Dataset112_ToothFairy2/labelsTr/`. | Implement FDI presence verification for 17-18, 27-28, 37-38, 47-48. |
| **3D Geometric Analysis ($D_{\min}$)** | **HIGH (100% Ready)** | Surface extraction and $k$-d tree distance computation already functional. | Refine physical coordinate mapping and tie-breaking. |
| **Mask-Guided Sagittal Cropper** | **HIGH (100% Ready)** | Aligned images and masks allow deterministic $224 \times 224$ extraction. | Implement `src/data/preprocessor.py` with boundary clamping/padding. |
| **Cropper Evaluation (ESR, DTAC, AR, CL)**| **HIGH (100% Ready)** | Full spatial geometry available to compute coverage, success rate, determinism, and latency. | Implement `src/data/eval_cropper.py` independently of classification. |
| **Supervised Transformer Training** | **BLOCKED (Clinical Labels)** | **ToothFairy2 contains ONLY anatomical masks, NOT clinical labels (`Independent`, `Contact`, `Resorption`).** | Provide annotation schema; build model architectures & mock/transfer weights; mark supervised training on real labels as blocked pending clinical annotation. |

---

## 2. Dataset Inventory

### 2.1 Public Benchmark Dataset: ToothFairy2
* **Source:** University of Modena and Reggio Emilia (UNIMORE) / ToothFairy2 Challenge (`https://ditto.ing.unimore.it/toothfairy2/`).
* **Format:** MetaImage (`.mha`), uncompressed `MET_UCHAR` for labels, 3D single-channel CBCT scans for images.
* **Archive Status:**
  - `data/toothfairy2/ToothFairy2_Dataset.zip`: 26.12 GB archive containing `Dataset112_ToothFairy2/imagesTr/` (480 volumetric CBCT scans `ToothFairy2F_001_0000.mha` to `ToothFairy2F_480_0000.mha`).
  - `data/toothfairy2/Dataset112_ToothFairy2/labelsTr/`: 480 extracted 3D label masks.
  - `data/toothfairy2/Dataset112_ToothFairy2/dataset.json`: metadata file defining 42 anatomical labels according to the Fédération Dentaire Internationale (FDI) schema.
* **Volume Resolution & Metadata:**
  - Voxel Spacing: Isotropic $0.3 \times 0.3 \times 0.3\text{ mm}$ (typical).
  - Spatial Grid: $512 \times 512 \times Z$ (axial slices $Z$ range from 250 to 350).
  - Anatomical Orientation: `RAI` (Right-Anterior-Inferior).
  - Transform Matrix: 3D identity matrix $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$.

### 2.2 Local Hospital Dataset
* **Directories:** `data/raw/` and `data/masks/`
* **Status:** Currently empty. No proprietary hospital DICOM series uploaded yet.

---

## 3. Anatomical Label Mapping & FDI Representation

According to `dataset.json`, integer labels map to tooth entities as follows:
* **Maxillary Right:**
  - Tooth 17 (Second Molar): Label `17`
  - Tooth 18 (Third Molar / Wisdom Tooth): Label `18`
* **Maxillary Left:**
  - Tooth 27 (Second Molar): Label `27`
  - Tooth 28 (Third Molar / Wisdom Tooth): Label `28`
* **Mandibular Left:**
  - Tooth 37 (Second Molar): Label `37`
  - Tooth 38 (Third Molar / Wisdom Tooth): Label `38`
* **Mandibular Right:**
  - Tooth 47 (Second Molar): Label `47`
  - Tooth 48 (Third Molar / Wisdom Tooth): Label `48`
* **Ancillary Structures:**
  - Left Maxillary Sinus: Label `5`
  - Right Maxillary Sinus: Label `6`
  - Left Inferior Alveolar Canal: Label `3`
  - Right Inferior Alveolar Canal: Label `4`

---

## 4. Critical Data Constraint — Absence of Clinical Relationship Labels

### 4.1 Provenance Gap
In our base reference paper (Karslıoğlu et al., Diagnostics 2026), the 306 M3–M2 units classified into `Contact`, `Independent`, and `Resorption` were collected from Bursa Uludağ University Faculty of Dentistry, and were individually diagnosed and annotated by radiologists using LabelMe on 2D representative sagittal slices.

**ToothFairy2 is strictly an anatomical segmentation dataset.** It does NOT contain clinical relationship annotations indicating whether an impaction exhibits:
1. `Class 0 (Contact)`: Contact without resorption.
2. `Class 1 (Independent)`: Separation with intact periodontal bone/ligament.
3. `Class 2 (Resorption / ERR)`: Pathological hard-tissue loss on the adjacent M2 root.

### 4.2 Non-Negotiable Rule Against Label Fabrication
Under Section 4 of `AGENTS.md` and Section 4.1 of `IMPLEMENTATION_PLAN.md`:
* We **MUST NOT** fabricate clinical labels.
* We **MUST NOT** assign `Class 1 (Independent)` simply because $D_{\min} > 0.5\text{ mm}$ or assign `Class 2 (Resorption)` because $D_{\min} \le 0.5\text{ mm}$.
* Doing so would create circular pseudo-labels and invalidate the scientific integrity of the research.

### 4.3 Proposed Labeling Schema & Clinical Workflow
For when adjudicated labels become available or an expert clinical annotation study is performed:
1. **Clinical Review Protocol:** Dual-observer joint review (Primary Researcher + drg. Sp.RKG) on 3D MPR CBCT reconstructions following Ericson & Kurol (1987, 1988) and Oenning et al. (2014) criteria.
2. **Metadata Contract:**
   `metadata.csv` will record:
   - `patient_id` (string)
   - `volume_id` (string)
   - `fdi_pair` (`17-18`, `27-28`, `37-38`, `47-48`)
   - `dmin_mm` (float, physical surface distance)
   - `sagittal_slice_idx` (int)
   - `crop_center_voxel` (tuple: [z, y, x])
   - `clinical_label` (`0: Contact`, `1: Independent`, `2: Resorption` or `UNLABELED`)
   - `adjudicator_id` (string)
   - `split` (`train`, `val`, `test` partitioned at patient level)

---

## 5. Feasibility Verdict & Action Plan

1. **Modules 1, 2, 4, 5 (Ingestion, Geometry, ROI Cropping, Cropper Evaluation, Triage Gate, Web CDSS UI, PDF Export):**
   **100% FEASIBLE IMMEDIATELY** on real ToothFairy2 3D CBCT volumes and masks.
2. **Module 3 (Supervised Classification on Real Ground-Truth):**
   - The model architectures (`Swin-S+MHA`, `Swin-S`, `ViT-B/16`), forward passes, soft-voting ensemble, checkpoints loading, Grad-CAM overlays, and evaluation metric pipeline will be implemented completely and tested.
   - Initial pretrained checkpoints (e.g., ImageNet weights via `timm`) will be wired up to demonstrate full inference and triage capability.
   - Supervised fine-tuning will remain marked as **Awaiting Clinical Ground-Truth Annotations** until verified clinical labels are provided.
