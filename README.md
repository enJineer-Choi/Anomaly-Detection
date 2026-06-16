# Semiconductor Component Defect Detection via Anomaly Detection

A two-stage anomaly detection pipeline for identifying defects in semiconductor
components, built on the MVTec AD dataset (`transistor` category) using
PyTorch and [anomalib](https://github.com/openvinotoolkit/anomalib).

## Overview

<!-- TODO: 1-2 paragraphs
- Your background: deep learning developer at a semiconductor inspection
  equipment company, working on AOI (Automated Optical Inspection)
- Why anomaly detection matters for industrial inspection (unsupervised,
  trained only on "good" samples — relevant when defective samples are rare)
- What this project demonstrates -->

## Dataset

- **Source**: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — `transistor` category
- **License**: CC BY-NC-SA 4.0 (non-commercial use only)

<!-- TODO: brief stats once you've explored the data
- Number of training images (good only) / test images (good + defective)
- Defect types present (e.g. bent_lead, cut_lead, damaged_case, misplaced, ...)
- One example image (good vs. defective) -->

## Approach

This project implements a two-stage approach to compare a simple baseline
against a stronger, more recent method:

1. **Stage 1 — AutoEncoder (baseline)**
   Reconstruction-based anomaly detection: trained only on normal images,
   anomalies are detected via high reconstruction error.

2. **Stage 2 — PatchCore** ([Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265))
   Patch-level feature memory bank + nearest-neighbor search. State-of-the-art
   accuracy on MVTec AD without requiring any anomalous training samples.

<!-- TODO: 1-2 sentences on what the comparison shows (e.g. trade-off between
simplicity/speed and accuracy) -->

## Results

| Model       | Image AUROC | Pixel AUROC | F1 Score | Inference Time (ms/image) |
|-------------|:-----------:|:-----------:|:--------:|:--------------------------:|
| AutoEncoder | TBD         | TBD         | TBD      | TBD                         |
| PatchCore   | TBD         | TBD         | TBD      | TBD                         |

<!-- TODO: fill in after running evaluation -->

## Visualizations

<!-- TODO: embed Grad-CAM / anomaly heatmap examples, e.g.
![Anomaly heatmap example](results/visualizations/transistor_example.png)
Show a normal sample, a defective sample, and the model's heatmap side by side. -->

## Project Structure

```
.
├── src/
│   ├── data/         # dataset loading utilities
│   ├── models/        # AutoEncoder, PatchCore wrappers
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py   # Grad-CAM / heatmap generation
├── configs/            # model configs (image size, batch size, etc.)
├── notebooks/          # exploration / experiments
└── results/
    ├── metrics/
    └── visualizations/
```

## How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py --model autoencoder --category transistor
python src/train.py --model patchcore --category transistor
```

### Evaluate

```bash
python src/evaluate.py --model patchcore --category transistor
```

<!-- TODO: adjust commands once your script argument names are finalized -->

## Future Work

<!-- TODO: e.g.
- Compare against newer lightweight models (EfficientAD, SimpleNet)
- ONNX export + TensorRT optimization for deployment-style inference benchmarking
- Extend to additional MVTec AD categories relevant to semiconductor inspection -->

## References

- Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection," CVPR 2019.
- Roth et al., "Towards Total Recall in Industrial Anomaly Detection," CVPR 2022.
- [anomalib](https://github.com/openvinotoolkit/anomalib) — OpenVINO Toolkit

## License

<!-- TODO: choose a license for your code (e.g. MIT). Note: the MVTec AD
dataset itself remains under CC BY-NC-SA 4.0 and is not redistributed in this repo. -->
