# PDF Image Extractor & Classifier

An automated pipeline that extracts embedded images from PDF documents and classifies them as **accepted** or **rejected** using a fine-tuned ResNet18 deep learning model.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Running the Pipeline](#2-running-the-pipeline)
- [Output Structure](#output-structure)
- [Configuration](#configuration)

---

## Overview

This project provides an end-to-end solution for:

1. **Training** a binary image classifier (accept/reject) using transfer learning on ResNet18.
2. **Extracting** embedded images from multiple PDF files.
3. **Classifying** each extracted image and sorting them into accepted/rejected folders.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                        │
│                                                         │
│   dataset/                                              │
│   ├── accept/          ┌──────────────┐                 │
│   │   ├── img1.jpg ──►│   train.py    │                 │
│   │   └── img2.jpg    │              │                 │
│   └── reject/          │  ResNet18    │                 │
│       ├── img3.jpg ──►│  Fine-Tuning  │                 │
│       └── img4.jpg    └──────┬───────┘                 │
│                              │                          │
│                              ▼                          │
│                   accept_reject_model.pth               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                        │
│                                                         │
│              ┌───────────────────────┐                  │
│              │  inference_model.py   │                  │
│              │                       │                  │
│              │  - Loads ResNet18     │                  │
│              │  - Loads trained      │                  │
│              │    weights (.pth)     │                  │
│              │  - classify_image()   │                  │
│              └───────────┬───────────┘                  │
│                          │                              │
│                  Returns: (label, confidence)            │
│                  label ∈ {"accept", "reject"}            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   PIPELINE PHASE                         │
│                                                         │
│   all pdfs/              ┌──────────────┐               │
│   ├── doc1.pdf ────────►│              │               │
│   ├── doc2.pdf ────────►│   app7.py    │               │
│   └── doc3.pdf ────────►│              │               │
│                          └──────┬───────┘               │
│                                 │                       │
│                    ┌────────────┴────────────┐          │
│                    ▼                         ▼          │
│          STEP 1: Extract            STEP 2: Classify    │
│          Images (PyMuPDF)           (ResNet18 Model)    │
│                    │                         │          │
│                    ▼                         ▼          │
│           embedded_images/       ┌───────────────────┐  │
│                                  │  filtered_images/ │  │
│                                  │  rejected_images/ │  │
│                                  └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Component Details

| Component              | File                 | Description                                                                 |
|------------------------|----------------------|-----------------------------------------------------------------------------|
| **Model Training**     | `train.py`           | Fine-tunes a pretrained ResNet18 on a binary dataset (accept/reject)        |
| **Model Inference**    | `inference_model.py` | Loads trained weights and exposes `classify_image()` for predictions        |
| **Pipeline**           | `app7.py`            | Orchestrates PDF image extraction → classification → sorting                |
| **Model Weights**      | `accept_reject_model.pth` | Saved PyTorch state dict for the trained ResNet18 model                |

### Model Details

- **Architecture:** ResNet18 (pretrained on ImageNet, fine-tuned)
- **Input Size:** 224 × 224 pixels (RGB)
- **Output Classes:** 2 — `accept`, `reject`
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.0001)
- **Training Split:** 80% train / 20% validation

---

## Project Structure

```
Image_Extractor/
│
├── app7.py                     # Main pipeline — extracts & filters images from PDFs
├── inference_model.py          # Model loading & image classification function
├── train.py                    # Model training script
├── accept_reject_model.pth     # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── all pdfs/                   # (Input) Place PDF files here
│   ├── document1.pdf
│   └── document2.pdf
│
├── dataset/                    # (Training) Training dataset directory
│   ├── accept/                 #   Images to accept
│   │   ├── img1.jpg
│   │   └── ...
│   └── reject/                 #   Images to reject
│       ├── img1.jpg
│       └── ...
│
└── <pdf_name>_all_images/      # (Output) Generated per PDF
    ├── embedded_images/        #   All extracted images
    ├── filtered_images/        #   Accepted images
    └── rejected_images/        #   Rejected images
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for faster training/inference)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Image_Extractor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training the Model

Prepare your dataset in the following structure:

```
dataset/
├── accept/
│   ├── image1.jpg
│   └── image2.jpg
└── reject/
    ├── image3.jpg
    └── image4.jpg
```

Run the training script:

```bash
python train.py
```

This will:
- Load the dataset with an 80/20 train/validation split
- Fine-tune ResNet18 for 8 epochs
- Save the trained model as `accept_reject_model.pth`

### 2. Running the Pipeline

1. Place your PDF files in the `all pdfs/` directory.
2. Ensure `accept_reject_model.pth` exists (either train it or use a pre-trained one).
3. Run the pipeline:

```bash
python app7.py
```

The pipeline will:
- Extract all embedded JPEG/PNG images from each PDF
- Classify each image using the trained model
- Sort images into `filtered_images/` (accepted) and `rejected_images/` (rejected)
- Print classification results with confidence scores

---

## Output Structure

For each PDF file (e.g., `report.pdf`), the pipeline creates:

```
report_all_images/
├── embedded_images/        # All extracted images from the PDF
├── filtered_images/        # Images classified as "accept"
└── rejected_images/        # Images classified as "reject"

report/                     # Copy of accepted images (clean output folder)
```

---

## Configuration

### Training Parameters (`train.py`)

| Parameter       | Default  | Description                     |
|-----------------|----------|---------------------------------|
| `DATA_DIR`      | `dataset`| Path to the training dataset    |
| `BATCH_SIZE`    | `16`     | Training batch size             |
| `EPOCHS`        | `8`      | Number of training epochs       |
| `LEARNING_RATE` | `0.0001` | Adam optimizer learning rate    |

### Pipeline Configuration (`app7.py`)

| Parameter    | Default    | Description                              |
|--------------|------------|------------------------------------------|
| `PDF_FOLDER` | `all pdfs` | Directory containing input PDF files     |
