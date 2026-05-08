# KidneyCancerPrediction

## Overview
KidneyCancerPrediction is a machine learning project focused on predicting kidney cancer tumor stage from volumetric CT scan data. The project implements a 3D Convolutional Neural Network (3D CNN) and compares its performance against a Random Forest baseline.

The pipeline includes:
- CT scan preprocessing
- Model training and evaluation
- Performance visualization using metrics and confusion matrices


## Dataset
This project uses the TCGA-KIRC (Kidney Renal Clear Cell Carcinoma) dataset from the GDC Data Portal:
[https://portal.gdc.cancer.gov/projects/TCGA-KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC).

Download the CT scan dataset and place the extracted folder:
`manifest-xxn3N2Qq630907925598003437` into the `raw_data` folder.

## Preprocessing
Before training, CT scans are:
- cropped and subsampled
- grayscale normalized
- sharpened using a Laplacian filter
- histogram equalized

These steps standardize the scans and reduce computational requirements.

## Running the CNN Pipeline
Open and run all cells in:

```bash
ML_pipeline.ipynb
```

## Running the Random Forest Baseline
Open and run all cells in:

```bash
random_forest_baseline.ipynb
```

## Outputs
The pipeline generates:
- model checkpoints
- accuracy, F1-score, and AUROC metrics
- confusion matrices
- training/loss plots

## Notes
- Training was performed primarily on an NVIDIA 3050-Ti GPU with 4GB VRAM.
- CT scans are downsampled due to memory constraints.
