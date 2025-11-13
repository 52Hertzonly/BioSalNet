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

1.  Clone this repository:
    ```bash
    git clone https://github.com/52Hertzonly/BioSalNet.git
    cd BioSalNet
    ```

2.  Install the required dependencies. We recommend using a virtual environment.
    ```bash
    # Using pip
    pip install -r requirements.txt

    # Or using Conda (if you have a conda_env.yml file)
    # conda env create -f conda_env.yml
    # conda activate biosalnet
    ```

## 📁 Repository Structure
