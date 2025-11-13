# BioSalNet
Thank you for your attention. The relevant code will be released after the paper is accepted.
# 1.Prediction Maps
Download the test results here:  
[Google Drive Download Link](https://drive.google.com/file/d/1nFw1X7ANIi4dXNnNPmtlo9v7IJZv4mPS/view)


# BioSalNet: Biologically Inspired Saliency Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official **implementation code and resources** for the paper **"BioSalNet: Biologically Inspired Saliency Prediction"** (currently under review).

> **Note on Code Availability**: To facilitate research reproducibility and peer review, we are providing the complete training and evaluation framework, including data loaders, model backbones, and metric calculations. **The core proprietary module of the BioSalNet architecture is not included in this public release** but will be made available upon publication of the paper.

## 📖 Overview

This project implements a novel deep learning model for visual saliency prediction, inspired by biological vision mechanisms. Saliency prediction aims to identify the most visually conspicuous regions in an image, mimicking human gaze behavior.

## 🚀 Features

*   **Full Training & Evaluation Pipeline**: Complete code for training and testing saliency prediction models.
*   **Multi-Dataset Support**: Data loaders and pre-processing scripts for popular saliency benchmarks (e.g., SALICON, MIT1003, CAT2000).
*   **Comprehensive Evaluation Metrics**: Implementation of standard saliency metrics (AUC-Judd, NSS, CC, SIM, KL-Divergence, etc.).
*   **Modular Design**: Easy-to-extend code structure for integrating new models and datasets.
*   **Reproducible Results**: Scripts to replicate the experiments and comparisons reported in our paper.

## ⚙️ Installation

### Our Experiment Environment
**Note:** Other environments may also work, but the following is the exact setup we used for all experiments, which guarantees reproducibility.

```bash
# Core Deep Learning Framework
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Essential Libraries
pip install transformers==4.35.2
pip install "numpy<1.24"  # We recommend numpy==1.23.5 or 1.22.4

# Computer Vision Libraries
pip install mmcv==1.3.8 mmengine==0.10.5 mmsegmentation==0.14.1
pip install timm==1.0.19
```
## 📥 Data Acquisition

### SALICON Dataset
The **SALICON** dataset is available through the official SALICON website. Due to the dynamic nature of online resources, we recommend searching for the most current access point using the dataset's official title: **"SALICON: Saliency in Context"**.

- **Official Reference**: Look for the SALICON dataset on academic data portals or the publisher's website associated with the original paper.
- **Content**: This dataset provides the `images/` (stimuli) and `maps/` (saliency maps) directories for your saliency prediction task.

### Depth Maps
The depth maps used in our work are not part of the original SALICON dataset. They were generated using pre-trained monocular depth estimation models.

- **Recommended Tool**: A modern and effective choice is the **Depth Anything** model, which provides a robust and unified solution for monocular depth estimation.
    - **Source Code & Model**: You can find the official implementation and pre-trained models on its [GitHub repository](https://github.com/DepthAnything/DepthAnything).
    - **Usage**: You can process the SALICON images through this model to generate the corresponding depth maps for the `depth/` directory.

- **Alternative Options**: Other popular depth estimation models you could consider include **MiDaS** and **DPT** (Dense Prediction Transformer).



## 🛠️ Configuration

The training script `train.py` accepts several important command-line arguments for configuration:
### SALICON Dataset Structure
```bash
salicon/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── depth/
│   ├── train/          # Depth maps for training images
│   └── val/            # Depth maps for validation images
└── maps/
    ├── train/          # Saliency maps for training images
    └── val/            # Saliency maps for validation images
```

### Dataset Paths
- `--salicon-root`: Root directory of SALICON dataset
- `--train-csv`: Path to CSV file listing training samples (default: `dataset/salicon_train.csv`)
- `--val-csv`: Path to CSV file listing validation samples (default: `dataset/salicon_val.csv`)

### Output & Logging
- `--log-dir`: Directory to save model checkpoints and training logs (default: `outputs/checkpoints`)

### Usage Example

```bash
python train.py \
    --salicon-root /path/to/your/salicon/dataset \
    --train-csv path/to/train.csv \
    --val-csv path/to/val.csv \
    --log-dir outputs/my_experiment
```

## 📁 Repository Structure
