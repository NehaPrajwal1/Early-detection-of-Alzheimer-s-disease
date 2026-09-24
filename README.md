# Early-detection-of-Alzheimer-s-disease
📌 Project Overview

Alzheimer's disease is a progressive neurodegenerative disorder that primarily affects memory, thinking, and cognitive abilities. Early detection can help support timely intervention and monitoring.

This project explores a machine learning and deep learning based approach for the early detection and classification of Alzheimer's disease using brain MRI scans and cognitive/speech-related information.

The project combines medical imaging with clinical and cognitive features to investigate whether machine learning models can identify patterns associated with Alzheimer's disease and its early stages.

# Alzheimer's Disease Classification from MRI (ADNI)

Multi-class (AD / CN / MCI) classification of brain MRI slices using deep learning, built on the ADNI1 dataset. This repo documents three iterations of the modeling pipeline, including a data leakage bug that was found and fixed mid-project.

## Overview

| Stage | Notebook | Architecture | Key Idea |
|---|---|---|---|
| 1 — Baseline | `01_baseline_resnet50.ipynb` | Simple CNN + ResNet50 (transfer learning) | Establish preprocessing pipeline, replicate paper baseline |
| 2 — Intermediate | `02_efficientnetb3.ipynb` | EfficientNetB3 + GeM pooling + SPP head | Focal loss, cosine LR annealing, 2-phase fine-tuning |
| 3 — Final | `03_efficientnetb5_final.ipynb` | EfficientNetB5 + MC-Dropout + TTA ensembling | Subject-level evaluation with leakage-free splits |

## Pipeline

1. **Preprocessing** — ADNI1 `.nii` volumes → middle 60 axial slices per scan, normalized and resized to 160×160 (later 224×224), saved as PNG.
2. **Splitting** — 70/15/15 train/val/test.
3. **Training** — Transfer learning from ImageNet-pretrained backbones, class-weighted loss to handle AD/CN/MCI imbalance, progressive unfreezing.
4. **Evaluation** — Both per-slice and per-subject (patient-level majority/soft voting) accuracy, since a scan yields many slices and slice-level accuracy alone is misleading.

## Key engineering finding: data leakage

Initial slice-level train/val/test splitting **did not account for the fact that multiple slices come from the same patient**. This let slices from the same subject appear in both train and test sets, inflating reported accuracy.

**Diagnosis:** Wrote a subject-ID extraction + overlap check across splits — found dozens of overlapping subjects (e.g. 38 overlapping between train/test in one run).

**Fix:** Rebuilt the dataset with **subject-level (patient-ID) splitting** — all slices from a given patient are forced into exactly one split — and switched evaluation to **subject-level soft voting** (averaging softmax probabilities across all slices of a scan) rather than trusting raw per-slice accuracy.

This is the main engineering takeaway of the project: a model can look good on paper while silently leaking patient identity across splits, and per-slice accuracy is not a trustworthy metric for scan-level classification tasks.

## Results (subject-level, leakage-free test set)

**3-class (AD / CN / MCI), EfficientNetB5, final model:**

| Metric | Value |
|---|---|
| Subject-level accuracy | 49.3% |
| Balanced accuracy | 41.5% |
| Macro AUC (OvR) | 0.60 |
| AD vs CN AUC | 0.62 |

**Binary (AD vs CN only), ResNet50, leakage-fixed split:**

| Metric | Value |
|---|---|
| Test accuracy | 67% |
| AD recall / precision | 0.56 / 0.63 |
| CN recall / precision | 0.76 / 0.70 |

Full confusion matrices, training curves, and ROC curves are in `results/`.

## Data

This project uses the **ADNI1** dataset. **No data is included in this repository.** See [`docs/DATA.md`](docs/DATA.md) for access instructions — ADNI data cannot be redistributed under its Data Use Agreement.

## Setup

```bash
git clone https://github.com/<your-username>/alzheimers-mri-classification.git
cd alzheimers-mri-classification
pip install -r requirements.txt
```

## Repository structure
notebooks/
01_baseline_resnet50.ipynb
02_efficientnetb3.ipynb
03_efficientnetb5_final.ipynb
results/
confusion_matrices.png
training_curves.png
roc_curves.png
docs/
DATA.md
requirements.txt




## Limitations

- Accuracy on the 3-class subject-level task is modest (~49%), reflecting the genuine difficulty of separating MCI (an intermediate, heterogeneous stage) from AD and CN using 2D slices alone.
- 2D slice-based classification discards 3D spatial context available in the full volume; a 3D CNN or multi-slice aggregation approach would likely improve results.
- Single-timepoint MRI was used; ADNI's longitudinal scans were not leveraged.

## Acknowledgment

Data used in this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this work. A complete listing of ADNI investigators is available at:
http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf
