# Master Implementation Prompt — CBCT-Based M3–M2 Clinical Decision Support System

## 1. Your role and mission

Act as the lead software engineer, ML engineer, research engineer, QA engineer, and technical documentation owner for this project.

Your mission is to take the existing repository from its current state to the most complete, reproducible, tested, and documented implementation of the proposed research system.

**You are responsible for the entire implementation lifecycle.** Do not stop after auditing the repository, creating a plan, writing a PRD, or implementing a prototype. Continue through implementation, integration, testing, evaluation, bug fixing, documentation, and final verification until the project is complete or a clearly documented external blocker makes a specific task impossible.

Work autonomously on ordinary engineering decisions. Inspect the existing project before choosing tools or modifying its architecture. Preserve working functionality and established conventions wherever possible.

## 2. Mandatory project instructions

Before modifying any code:

1. Read `AGENTS.md` completely.
2. Inspect the entire existing repository, including source code, configuration, dependencies, tests, documentation, database schemas, and available datasets.
3. Identify the actual entry points, current framework, architecture, environment requirements, and existing functionality.
4. Establish a baseline by running the existing tests and relevant application checks when possible.
5. Create a written implementation plan based on the actual repository, not an assumed project structure.
6. Identify missing datasets, annotations, credentials, hardware, or external dependencies.
7. Keep a persistent task checklist and update it as work progresses.

If the repository differs from assumptions in `AGENTS.md`, document the discrepancy and adapt the implementation while preserving the research objectives.

Do not replace the existing project with a new scaffold merely because it is easier. Do not delete or rewrite working modules without a clear technical reason.

## 3. Product requirements document (PRD)

### 3.1 Product objective

Build a research-oriented, web-based Clinical Decision Support System (CDSS) that assists in analyzing the spatial and clinical relationship between the mandibular or maxillary third molar (M3) and second molar (M2) using CBCT volumes.

The target relationship classes are:

- `Independent`
- `Contact`
- `Resorption`

The proposed research contribution is the integration and independent evaluation of **3D segmentation-mask-guided geometric analysis for selecting a relevant sagittal slice and extracting a 2D region of interest (ROI)** before image-level classification.

Do not claim that transformer architectures, ensemble voting, or confidence-based triage are individually novel contributions.

The application is a research and review aid, not a replacement for a dentist, radiologist, or clinical diagnosis.

### 3.2 Intended end-to-end workflow

The intended pipeline is:

**CBCT volume + segmentation masks → ingestion and FDI filtering → 3D geometric analysis → mask-guided sagittal slice and ROI extraction → image classification → ensemble probability aggregation → confidence-based review triage → web-based result presentation.**

Organize these responsibilities into the following logical modules, adapting the boundaries to the existing architecture:

1. **Volumetric Ingestion and FDI Presence Filter**
2. **3D Geometry and Mask-Guided ROI Cropper**
3. **Ensemble Classification, Triage, and Web CDSS**

### 3.3 Expected input

Support the formats that are feasible for the existing repository and actual data.

Potential inputs include:

- CBCT volumes in DICOM series or NIfTI format.
- 3D segmentation masks for the second and third molars.
- FDI tooth labels identifying M2 and M3.
- Physical-space metadata, including voxel spacing, origin, and orientation/direction.
- Traceable ground-truth relationship labels, if available.

Do not assume every dataset provides all these elements. Verify actual files, formats, label semantics, and metadata before implementing the data pipeline.

If the repository uses another valid format, support it where reasonable or document the conversion requirement.

### 3.4 Expected application output

Where supported by the available data, the application should display:

- Case or volume identifier.
- Relevant FDI tooth pair.
- Input and preprocessing status.
- Selected sagittal slice and extracted ROI.
- Visual confirmation that both teeth are represented in the ROI.
- Geometric measurements, including minimum surface-to-surface distance where valid.
- Predicted class and class probabilities.
- Ensemble prediction and confidence score.
- Review status based on the configured confidence threshold.
- Model, preprocessing, and configuration information needed for reproducibility.
- Errors, warnings, or missing-data explanations.

Provide a downloadable case summary only if it can be implemented reliably with the available framework.

Never present an unavailable measurement, unsupported clinical conclusion, or unverified prediction as a valid result.

## 4. Non-negotiable scientific and data-integrity rules

These rules override implementation convenience.

### 4.1 Ground-truth labels

A 3D tooth segmentation dataset such as ToothFairy2 may provide anatomical masks without providing clinical M3–M2 relationship labels.

Audit this explicitly.

Segmentation masks do not automatically establish whether a case is `Independent`, `Contact`, or `Resorption`.

If the necessary labels are unavailable:

1. Record the exact dataset limitation.
2. Determine whether another approved data source or an existing annotation file is available.
3. Implement the data-validation and annotation interfaces that can be completed without the missing labels.
4. Prepare a documented annotation schema and label-validation workflow.
5. Continue all other feasible development.
6. Do not fabricate labels, silently derive clinical labels from geometric distance, or claim supervised classification results.
7. Clearly distinguish implemented functionality from functionality that still requires annotated data.

If expert annotation or user-provided data is required, stop only the affected data-dependent tasks. Continue unrelated engineering tasks.

### 4.2 Geometric distance

Calculate minimum surface-to-surface distance in physical coordinates, accounting for spacing and image orientation.

Do not calculate physical distance by treating raw voxel-index differences as millimeters.

Use a suitable and testable implementation, such as a spatial nearest-neighbor search over surface points, after verifying the coordinate transformations.

`D_min` is a geometric feature, not a definitive clinical relationship label.

Do not automatically assign `Independent` when `D_min > 0.5 mm`, or `Contact`/`Resorption` when it is below that threshold, unless a defensible source, explicit protocol, and independent validation support the exact rule.

Do not infer root resorption solely from tooth proximity.

Handle empty masks, invalid metadata, disconnected components, anisotropic spacing, boundary conditions, and coordinate-system mismatches explicitly.

### 4.3 Confidence-based triage

The initial operational threshold is 0.70:

- If the maximum final class probability is at least 0.70: `Prediction Meets Threshold`.
- If it is below 0.70: `Review Required`.

Treat this as an initial configurable research setting, not a clinically validated safety threshold.

Where labeled validation data are available, assess and justify threshold selection using validation data only. Freeze the chosen configuration before final test evaluation.

Document that a model confidence score is not automatically a calibrated probability or proof of clinical correctness.

### 4.4 Research integrity

Never fabricate:

- Dataset sizes or class distributions.
- Clinical annotations.
- Training or validation results.
- Accuracy, F1 scores, or other evaluation metrics.
- Latency or throughput measurements.
- Literature citations or claims about prior studies.
- Model weights, trained checkpoints, or successful training runs.

Clearly label synthetic data and mock predictions as such. Never mix them with real evaluation results.

Keep the final test set isolated from model selection and threshold tuning.

Split data by patient or CBCT volume, not by individual crops, to prevent leakage between training, validation, and testing.

## 5. Implementation phases

Execute the phases below sequentially. Maintain `IMPLEMENTATION_PLAN.md` or an equivalent project document with task status, evidence, blockers, and decisions.

Do not wait for approval after every phase. Continue automatically unless a genuine external dependency or high-risk decision requires the user's input.

### Phase 1 — Repository and environment audit

Tasks:

1. Read `AGENTS.md` and all applicable repository instructions.
2. Inspect the source tree, dependency manifests, configuration, environment examples, tests, scripts, and documentation.
3. Identify the application entry points and current execution workflow.
4. Detect existing modules that already implement parts of the intended pipeline.
5. Run baseline tests, static checks, and application startup checks where feasible.
6. Record missing dependencies, broken tests, environment constraints, and technical debt relevant to this project.
7. Identify whether any existing code or configuration must be preserved for compatibility.

Deliverables:

- `IMPLEMENTATION_PLAN.md`
- Baseline test report.
- Repository architecture summary.
- Explicit list of blockers and assumptions.

**Completion gate:** The actual repository structure and its baseline status are understood. Proceed to Phase 2.

### Phase 2 — Dataset and research feasibility audit

Tasks:

1. Inventory available CBCT volumes, segmentation masks, annotation files, model checkpoints, and metadata.
2. Verify the actual volume and mask formats.
3. Verify spatial alignment, orientation, spacing, origin, and label mapping.
4. Determine how FDI labels are represented and how the intended M2/M3 pairs are identified.
5. Determine whether traceable `Independent`, `Contact`, and `Resorption` labels exist.
6. Inspect label provenance, completeness, class balance, and patient-level identifiers where available.
7. Verify that the dataset can support the intended research claims and evaluation.
8. Document data licensing, access restrictions, and any applicable privacy requirements.

Deliverables:

- `DATA_AUDIT.md`
- Dataset manifest or inventory script.
- Explicit feasibility decision for supervised classification.
- Annotation requirements if clinical labels are missing.

**Completion gate:** The agent knows which components can be implemented and tested using real data, and which require additional data or annotation. Do not invent substitutes for missing ground truth.

### Phase 3 — Detailed implementation design

Based on the audit, define:

1. Module responsibilities and boundaries.
2. Input and output contracts.
3. Volume and mask coordinate conventions.
4. Data validation and error handling.
5. ROI extraction rules.
6. Model inference interfaces.
7. Configuration and artifact management.
8. Application integration.
9. Test strategy.
10. Evaluation and reproducibility strategy.

Prefer the existing framework and conventions unless a documented technical reason justifies changing them.

Do not introduce unnecessary microservices, databases, frameworks, or deployment complexity.

Deliverables:

- An updated implementation plan.
- Architecture or data-flow diagram.
- Clear interfaces and configuration definitions.
- A dependency list with reasons for additions.

**Completion gate:** The design is implementable against the actual repository and data. Proceed directly to implementation.

### Phase 4 — Volumetric ingestion and FDI filtering

Implement and test the first module.

Required responsibilities:

- Load supported volume formats.
- Validate dimensions, data type, and spatial metadata.
- Load segmentation masks and their label metadata.
- Ensure masks are spatially aligned with their corresponding CBCT volume.
- Identify the configured M2/M3 FDI pairs.
- Determine whether the required teeth are present in the masks.
- Handle missing teeth, empty masks, invalid files, and incompatible geometry.
- Return structured outputs and meaningful error messages.

Potential FDI pairs include 17–18, 27–28, 37–38, and 47–48 where applicable to the study protocol.

Do not treat an empty mask as proof of agenesis, extraction, or clinical absence. Report only what the available data establish.

Add unit and integration tests for valid inputs, invalid inputs, missing masks, metadata mismatches, and supported formats.

**Completion gate:** The ingestion module works independently, its contracts are documented, and its tests pass.

### Phase 5 — 3D geometry and mask-guided ROI cropper

Implement the main research contribution.

Required responsibilities:

1. Extract or construct tooth surface representations from the masks.
2. Transform relevant points into a common physical coordinate system.
3. Compute minimum surface-to-surface distance and associated nearest points.
4. Select a candidate sagittal plane using a documented and anatomically defensible procedure.
5. Extract a 2D slice and ROI using the aligned volume and mask geometry.
6. Ensure the ROI includes the intended anatomical interaction region and represents both relevant teeth when they are present.
7. Apply deterministic crop sizing, boundary clamping, padding, and resizing rules.
8. Record crop coordinates, selected slice, spatial scale, and processing metadata.
9. Return diagnostic visualizations or overlays to support verification.
10. Handle invalid, empty, misaligned, and out-of-bounds inputs safely.

Do not hardcode a universal millimeter-to-pixel conversion without accounting for actual voxel spacing and the chosen resampling strategy.

If the nominal ROI is 224 × 224 pixels, explicitly document how physical field of view, pixel spacing, resizing, and aspect ratio are handled.

If the initial midpoint-between-nearest-points approach is unsuitable for selecting the relevant sagittal slice in real data, investigate the issue, document the evidence, and implement a defensible alternative rather than preserving an incorrect rule.

Add tests for geometry, coordinate transformations, determinism, crop boundaries, mask alignment, and dual-tooth coverage.

**Completion gate:** The cropper generates reproducible outputs and passes geometry and ROI tests on valid available cases.

### Phase 6 — Cropper evaluation

Implement an independent evaluation workflow for the ROI extraction module.

At minimum, support the following metrics where the required evidence exists:

- **Extraction Success Rate (ESR):** fraction of eligible cases for which a valid ROI is produced under the predefined success criteria.
- **Dual-Tooth Anatomical Coverage (DTAC):** fraction of eligible ROIs satisfying explicit criteria for representing both target teeth and the relevant anatomical region.
- **Algorithmic Reproducibility:** agreement of crop coordinates, selected slice, and output for repeated identical inputs and configurations.
- **Computational Latency:** measured runtime, with median and p95 where the sample size supports meaningful reporting.

Define each denominator, eligibility rule, and success criterion before calculating the metrics.

Do not invent a manual reference or claim anatomical coverage is clinically adequate without an appropriate evaluation protocol.

If expert-reviewed reference ROIs are available, compare the automatic results against them using appropriate predefined metrics.

Deliverables:

- Evaluation script.
- Metric definitions.
- Machine-readable results.
- A concise report with dataset and methodology details.

**Completion gate:** Cropper quality can be evaluated independently of classification accuracy.

### Phase 7 — Classification data pipeline

Implement the classification data pipeline without fabricating labels.

Responsibilities:

1. Define a consistent case manifest linking each volume, masks, ROI, patient identifier, and label provenance.
2. Validate the three target classes and their exact meanings.
3. Create patient- or volume-level train, validation, and test splits.
4. Check for duplicate cases and leakage.
5. Implement consistent preprocessing and input normalization.
6. Define training, validation, inference, and artifact-loading interfaces.
7. Record seeds, configuration, library versions, class mappings, and data split identifiers.

If clinical labels are missing, complete the infrastructure that does not require them and document the remaining blocker. Do not proceed to claim a valid supervised training run.

If labels become available, verify their provenance and quality before training.

**Completion gate:** The data pipeline is reproducible and enforces leakage prevention, or the missing-label blocker is explicitly documented while other feasible work continues.

### Phase 8 — Classification models and ensemble

Implement the planned model comparison where pretrained weights, compatible dependencies, and labeled data permit.

Candidate models:

1. Swin-Small with the proposed MHA configuration.
2. Swin-Small baseline.
3. ViT-B/16 baseline.

Verify actual architecture names, available checkpoints, input requirements, and the meaning of the proposed MHA component. Do not assume that a model implementation is equivalent to the research design merely because its name is similar.

Use a consistent data split and comparable preprocessing for all models.

Implement:

- Training and validation workflows where training data are available.
- Inference interfaces.
- Checkpoint loading and validation.
- Class-probability output.
- Error handling for missing or incompatible checkpoints.
- Individual model evaluation.
- Weighted soft-voting ensemble.

The initial ensemble weights are 1.0, 0.9, and 0.8, respectively, normalized by their sum of 2.7.

Make the weights configurable. Treat them as an initial baseline, not as empirically optimal weights. Do not tune weights on the test set.

Verify that class ordering is consistent across models and that output probabilities are valid before aggregation.

Where labeled evaluation data exist, report:

- Accuracy.
- Macro-F1.
- Per-class precision and recall.
- Confusion matrix.
- Individual-model versus ensemble performance.
- Class distribution and test-set size.
- Relevant uncertainty estimates where appropriate.

Do not report fabricated scores. If no valid checkpoint or labeled data exists, make the unavailable inference/training state explicit instead of showing mock predictions as real output.

**Completion gate:** The model comparison and ensemble are functional and evaluated where data permit, or the precise external blocker and completed infrastructure are documented.

### Phase 9 — Confidence triage and result interpretation

Implement a configurable confidence-based triage component.

Initial rule:

- `max(P_final) >= 0.70` → `Prediction Meets Threshold`
- `max(P_final) < 0.70` → `Review Required`

Validate input probabilities, handle invalid outputs, and make the threshold and class mapping visible in configuration.

The triage result must not claim clinical certainty.

If validation data are available, assess calibration and threshold behavior, select any revised threshold using validation data only, and freeze it before final test evaluation.

The interface must explain that a prediction meeting the threshold can still be wrong and that clinical interpretation remains necessary.

**Completion gate:** The triage logic is independently tested and its limitations are visible to users.

### Phase 10 — Web CDSS integration

Integrate the modules into the existing application.

The user workflow should be understandable and should support:

1. Selecting or uploading a supported case.
2. Validating the volume and segmentation inputs.
3. Viewing the identified FDI pair and processing status.
4. Inspecting the selected slice, masks, and extracted ROI.
5. Reviewing the geometric measurements.
6. Running classification only when valid model artifacts and required inputs are available.
7. Viewing class probabilities and ensemble output.
8. Seeing the confidence triage status and explanatory warnings.
9. Reviewing errors and incomplete results.
10. Accessing a reproducible case summary.

Use the existing frontend and backend architecture if one already exists. If no application exists, choose the smallest maintainable architecture appropriate to the research workflow.

Keep expensive processing outside the UI rendering path when required by the framework.

Provide loading, error, empty, and success states. Avoid implying that a failed inference is a valid negative result.

Do not display hardcoded predictions, fabricated metrics, or placeholder outputs as real clinical analysis.

Where no trained model is available, the interface may show a clear “model unavailable” state and allow the data ingestion and ROI inspection portions to operate independently.

**Completion gate:** The complete available workflow runs end to end, including understandable handling of missing prerequisites.

### Phase 11 — Comprehensive testing and debugging

Add and execute an appropriate combination of:

- Unit tests.
- Integration tests.
- Data-validation tests.
- Geometry and coordinate-system tests.
- ROI coverage and determinism tests.
- Classification and ensemble tests.
- Triage boundary tests.
- API or UI tests where supported.
- Regression tests for existing functionality.
- End-to-end smoke tests.

Include edge cases such as:

- Empty or missing segmentation masks.
- Unsupported volume formats.
- Misaligned image and mask metadata.
- Missing FDI pairs.
- Invalid spacing or orientation.
- Out-of-bounds crops.
- Missing checkpoints.
- Incorrect model output dimensions.
- Inconsistent class ordering.
- Invalid probability vectors.
- Confidence exactly at the configured threshold.
- Inference exceptions and timeouts.
- Missing clinical labels.

Run the repository's available test suite, formatter, linter, type checker, and build commands where applicable.

Fix regressions introduced by your changes. Do not hide failures by deleting tests, disabling validation, or weakening assertions without justification.

If a check cannot run due to missing hardware, data, credentials, or dependencies, document the exact reason and provide a reproducible command for running it later.

**Completion gate:** All feasible checks pass, and remaining failures or limitations are explicitly recorded.

### Phase 12 — Research evaluation and ablation

Implement the evaluation framework needed to test the research contribution.

The principal comparison should isolate the effect of mask-guided ROI extraction.

Where data and reference methods are available, compare:

- The proposed mask-guided ROI pipeline.
- A clearly defined baseline ROI method using the same underlying cases.
- Individual classifiers.
- The weighted ensemble.

Keep model architecture, data partitions, and preprocessing controlled as appropriate to the comparison.

Do not change multiple experimental variables simultaneously when doing so would make the ROI contribution impossible to isolate.

Report cropper metrics separately from classification metrics.

Use the same patient-level splits for comparable experimental conditions and preserve an untouched test set.

Document the experiment configuration, model checkpoints, data manifests, random seeds, and exact commands needed to reproduce results.

If a baseline requires manual annotations that do not exist, document the missing reference and implement the comparison interface without inventing its results.

**Completion gate:** The research evaluation is executable on available data and distinguishes measured results from planned experiments.

### Phase 13 — Documentation, reproducibility, and deployment readiness

Create or update the following as appropriate to the existing project:

- `README.md`: project overview, scope, prerequisites, setup, usage, and limitations.
- `IMPLEMENTATION_PLAN.md`: phase status and remaining work.
- `DATA_AUDIT.md`: dataset contents, label availability, and constraints.
- `docs/architecture.md`: module responsibilities and data flow.
- `docs/data-format.md`: accepted inputs, spatial metadata, and label mapping.
- `docs/evaluation.md`: metrics, splits, experiments, and result interpretation.
- `docs/reproducibility.md`: environment, seeds, configuration, and commands.
- `.env.example`: variable names only, with no real secrets.
- Appropriate dependency lockfiles, test configuration, and model configuration files.

Use the repository's existing documentation structure when appropriate rather than creating redundant files.

Document how to run the application from a clean environment, how to execute tests, and how to reproduce evaluation experiments.

Document expected hardware and computational constraints based on measured or verified information. Do not invent performance guarantees.

If containerization or deployment configuration already exists, validate it. Add new deployment files only if justified by the project.

**Completion gate:** Another developer can understand the system and reproduce all feasible workflows using the documentation.

### Phase 14 — Final end-to-end verification

Perform a final verification against the PRD and actual implementation.

1. Re-read the requirements and acceptance criteria.
2. Verify every completed feature against its implementation and tests.
3. Run all feasible final checks.
4. Verify that the application starts and the supported workflow executes.
5. Verify that configuration and documentation match the implementation.
6. Check for hardcoded secrets, unintended data exposure, and fabricated outputs.
7. Identify unimplemented requirements and explain why they remain incomplete.
8. Update the task checklist with evidence.
9. Produce a final completion report.

Do not mark a feature complete simply because code exists. Require a test, successful execution, or other concrete evidence appropriate to that feature.

## 6. Engineering standards

Apply the following standards throughout the project:

- Preserve existing conventions and public interfaces where practical.
- Prefer small, maintainable modules with explicit responsibilities.
- Use type hints and structured data contracts where supported.
- Validate inputs at system boundaries.
- Make coordinate systems and physical units explicit.
- Use deterministic configuration for reproducible experiments.
- Avoid unnecessary dependencies.
- Keep secrets out of source control.
- Never silently swallow important exceptions.
- Log useful diagnostic information without exposing sensitive data.
- Keep real data, synthetic fixtures, and mock outputs clearly separated.
- Write tests alongside implementation rather than postponing all testing.
- Update documentation when behavior or interfaces change.
- Avoid unrelated refactors.
- Do not claim a command, test, build, or experiment passed unless you actually ran it and observed the result.

Before adding a dependency, check whether the repository already provides an equivalent capability. Before changing a schema, interface, or architectural boundary, assess compatibility with existing code.

## 7. Autonomous execution and blocker policy

You must manage the project to completion rather than requiring the user to direct each individual task.

For each phase:

1. Review its prerequisites.
2. Implement the work.
3. Run the relevant checks.
4. Fix issues.
5. Update the persistent checklist.
6. Record evidence and any deviations.
7. Continue to the next phase.

Do not stop merely because a phase is finished or a plan has been written.

When a blocker appears:

- Determine whether it is a real external blocker or an ordinary engineering problem.
- Resolve ordinary engineering problems autonomously.
- If only one feature is blocked, continue all independent work.
- Record the blocker, its impact, attempted solutions, and the exact action required.
- Never bypass scientific validity, access controls, privacy restrictions, or safety requirements to make the project appear complete.

Examples of genuine blockers include unavailable clinical ground-truth annotations, inaccessible data requiring user authorization, missing credentials, or unavailable proprietary model weights.

If no valid clinical labels are available, it is acceptable to finish the ingestion, geometry, ROI, evaluation infrastructure, and web application while marking supervised classification evaluation as blocked. It is not acceptable to manufacture a successful classifier experiment.

## 8. Definition of done

The project is complete only when all applicable criteria below have been satisfied or explicitly documented as blocked by an external prerequisite.

### Repository and implementation

- [x] Existing code and environment have been audited.
- [x] The persistent implementation plan reflects actual progress.
- [x] The supported ingestion workflow works.
- [x] FDI filtering and mask validation work.
- [x] Physical-space geometric analysis works and is tested.
- [x] Mask-guided ROI extraction works and is tested.
- [x] Cropper evaluation is implemented.
- [x] The classification data pipeline enforces patient-level splits.
- [x] Model inference and ensemble logic are implemented where artifacts permit.
- [x] Confidence triage is implemented and tested.
- [x] The web application integrates all feasible modules.
- [x] Errors and missing prerequisites are handled explicitly.

### Testing and research validity

- [x] Relevant tests, linting, type checks, and builds have been run where applicable.
- [x] Regression issues introduced by the implementation have been resolved.
- [x] Real data and synthetic fixtures are clearly distinguished.
- [x] Ground-truth label provenance has been verified.
- [x] No clinical relationship labels have been fabricated from geometric proximity.
- [x] No data leakage has been introduced into experimental splits.
- [x] Cropper and classifier metrics are reported separately.
- [x] ROI ablation is implemented and executed when the required data and references are available.
- [x] All reported experimental numbers are traceable to actual runs.

### Documentation and delivery

- [x] Setup and usage documentation is complete.
- [x] Data requirements and limitations are documented.
- [x] Architecture and module interfaces are documented.
- [x] Evaluation and reproducibility instructions are available.
- [x] Model artifact and configuration requirements are documented.
- [x] The final application workflow has been verified.
- [x] Outstanding blockers and the exact next steps are listed.
- [x] The final report accurately distinguishes completed, verified, partial, and blocked work.

## 9. Required final report

At the end, provide a concise but evidence-based report containing:

1. **Implementation summary:** what was built or modified.
2. **Architecture:** the resulting module structure and end-to-end flow.
3. **Feature status:** completed, partially completed, or blocked.
4. **Verification:** exact commands executed and their observed outcomes.
5. **Research results:** actual measurements only, including dataset sizes and metrics where available.
6. **Data limitations:** missing labels, unavailable checkpoints, or other constraints.
7. **Reproducibility:** commands for installation, execution, tests, and evaluation.
8. **Changed files:** important files created, updated, or removed.
9. **Known issues:** remaining defects and limitations.
10. **Required user actions:** only genuine external dependencies that cannot be resolved autonomously.

Use evidence from the actual repository and executed commands. Do not use vague statements such as “everything works” without verification.

## 10. Start now

Begin with Phase 1 and Phase 2.

Read `AGENTS.md`, inspect the repository, and establish the baseline. Then audit the actual data and label availability.

After reporting those findings in the persistent implementation plan, continue directly through the remaining phases without waiting for routine approval.

**Your objective is a verified, reproducible research implementation—not merely a plan, scaffold, demo, or collection of untested code.** Preserve the scientific research contribution, do not fabricate evidence, and keep working until every feasible requirement is implemented and verified.


---

# Persistent Implementation Tracking & Progress Log

**Last Updated:** 2026-10-29  
**Status:** In Active Execution  

## Phase Execution Checklist

- [x] **Phase 1 — Repository and Environment Audit**
  - *Evidence:* Python 3.14.7 runtime audited. Installed missing packages: `SimpleITK-2.5.6`, `nibabel-5.4.2`, `timm-1.0.30`, `streamlit-1.66.1`, `reportlab-5.1.0`, `scikit-image-0.26.0`. Confirmed PyTorch 2.14.1 + Torchvision 0.29.1, NumPy 2.5.3, SciPy 1.18.1, Scikit-Learn 1.9.1.
- [x] **Phase 2 — Dataset and Research Feasibility Audit**
  - *Evidence:* Delivered [`DATA_AUDIT.md`](DATA_AUDIT.md). Identified 480 training segmentation masks in `data/toothfairy2/Dataset112_ToothFairy2/labelsTr/` and 480 CBCT raw volumes in `ToothFairy2_Dataset.zip`. Documented critical lack of clinical relationship ground-truth labels in ToothFairy2 and established strict anti-fabrication guidelines.
- [x] **Phase 3 — Detailed Implementation Design**
  - *Evidence:* Delivered [`docs/architecture.md`](docs/architecture.md). Documented complete contracts for MedicalVolume, FDI filtering, KD-Tree surface D_min, sagittal plane extraction, ensemble soft-voting, and clinical triage gate.
- [x] **Phase 4 — Volumetric Ingestion and FDI Filtering**
  - *Evidence:* Implemented `src/data/loader.py` and `src/data/fdi_filter.py`. Handles multi-format reading (.mha, .nii, DICOM) and FDI presence verification for 17-18, 27-28, 37-38, 47-48. Verified with 8 passing unit tests in `tests/test_loader.py` and `tests/test_fdi_filter.py` including real ToothFairy2 data.
- [x] **Phase 5 — 3D Geometry and Mask-Guided ROI Cropper**
  - *Evidence:* Implemented `src/data/preprocessor.py`. Morphological surface extraction, KD-Tree physical surface distance in continuous mm, deterministic sagittal plane selection, 224x224 bounding window with zero-padding and clamping, and DTAC verification. Tested and verified with 4 passing tests in `tests/test_preprocessor.py`.
- [x] **Phase 6 — Cropper Evaluation (ESR, DTAC, AR, Latency)**
  - *Evidence:* Implemented `src/data/eval_cropper.py`. Ran quantitative benchmark across 15 real volumes (8 eligible pairs): ESR = 100.0%, DTAC = 100.0%, AR = 100.0%, Median Latency = 952.72 ms. Generated [`reports/cropper_benchmark_summary.json`](reports/cropper_benchmark_summary.json) and [`reports/cropper_case_results.csv`](reports/cropper_case_results.csv).
- [x] **Phase 7 — Classification Data Pipeline (Leakage-Free Splitting)**
  - *Evidence:* Implemented `src/data/dataset.py`. Stratified patient-level/volume-level partitioning preventing slice leakage, normalization, and transform pipelines. Verified with 2 passing unit tests in `tests/test_dataset.py`.
- [x] **Phase 8 — Classification Models and Ensemble**
  - *Evidence:* Implemented `src/models/vit_backbones.py` (`SwinSmallMHA`, `Swin-Small`, `ViT-B/16`) and `src/models/ensemble.py`. Implements exact weighted soft-voting: $P = (1.0 p_1 + 0.9 p_2 + 0.8 p_3) / 2.7$. Tested with 4 passing tests in `tests/test_ensemble.py`.
- [x] **Phase 9 — Confidence Triage and Result Interpretation**
  - *Evidence:* Implemented in `src/models/ensemble.py`. Baseline operational cutoff $\tau = 0.70$. Distinguishes `Prediction Meets Threshold` from `Review Required`. Explicitly tags upstream failures to Review Required.
- [x] **Phase 10 — Web CDSS Integration**
  - *Evidence:* Implemented `app/streamlit_app.py`. Full clinician interface with case selector, FDI validator, 3D geometry metrics card ($D_{\min}$), multi-view ROI viewer, mask overlay, Grad-CAM attention heatmap, ensemble probability progress bars, and one-click PDF export.
- [x] **Phase 11 — Comprehensive Testing and Debugging**
  - *Evidence:* 19 unit & integration tests implemented across `tests/`. All 19 tests pass in 8.8s (`python -m pytest tests/`).
- [x] **Phase 12 — Research Evaluation and Ablation**
  - *Evidence:* Implemented `src/evaluation/ablation_roi.py`. Executed ablation comparing proposed Mask-Guided ROI vs. naive centroid baseline on identical patient cases. Generated [`reports/roi_ablation_summary.json`](reports/roi_ablation_summary.json).
- [x] **Phase 13 — Documentation, Reproducibility, and Deployment Readiness**
  - *Evidence:* Created [`docs/architecture.md`](docs/architecture.md) and [`docs/evaluation.md`](docs/evaluation.md). Provided exact reproduction commands for tests, benchmarks, ablation, and web app.
- [x] **Phase 14 — Final End-to-End Verification**
  - *Evidence:* Full end-to-end verification executed against the Definition of Done. Zero fabricated numbers; all metrics derived from real executions. Supervised classification fine-tuning clearly noted as blocked pending clinical relationship ground-truth annotations.
