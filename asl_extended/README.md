# ASL Extended System

## Overview

This folder is my extended ASL recognition system with two phases:
- **Phase 1:** CNN-based fingerspelling recognition (trained from my existing processed images)
- **Phase 2:** Word-level ASL recognition (trained from WLASL videos from Hugging Face)

## Quick Start

### Installation

```bash
pip install --break-system-packages -r requirements.txt
```

### Phase 1: CNN-Based Fingerspelling (Current)

**I use my existing processed images, so I don’t need new data collection for Phase 1.**

1. **Train Model:**
   ```bash
   jupyter notebook notebooks/train_cnn_model.ipynb
   ```
   Run all cells — it uses my `dataSet/` images

2. **Run Application:**
   ```bash
   python app/cnn_app.py
   ```

### Phase 2: Word-Level Recognition (Ready!)

**I use the WLASL video dataset from Hugging Face, so I can start Phase 2 without collecting new data.**

1. **Install Dependencies:**
   ```bash
   pip install --break-system-packages fiftyone mediapipe<0.11.0
   ```

2. **Load WLASL Dataset:**
   ```bash
   python data/wlasl_loader.py
   ```

3. **Train Word Model:**
   ```bash
   jupyter notebook notebooks/train_word_model.ipynb
   ```

4. **Run Word Recognition App:**
   ```bash
   python app/word_app.py
   ```

**I documented the full Phase 2 steps in [PHASE2_START.md](PHASE2_START.md).**

## Detailed Guide

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for:
- Complete setup instructions
- Troubleshooting
- Phase 2 implementation guide (WLASL → landmarks → sequence model)
- Next steps roadmap

## Project Structure

```
asl_extended/
├── data/
│   ├── cnn_data_loader.py       # Phase 1: Load processed images for CNN
│   ├── image_processor.py        # Phase 1: Image preprocessing
│   ├── wlasl_loader.py           # Phase 2: Load WLASL from Hugging Face
│   └── sequence_loader.py        # Phase 2: Load landmark sequences
├── models/
│   ├── cnn_classifier.py        # Phase 1: CNN model
│   └── sequence_classifier.py   # Phase 2: BiLSTM model
├── utils/
│   └── holistic_extractor.py    # MediaPipe Holistic (for Phase 2 only)
├── app/
│   ├── cnn_app.py               # Phase 1: CNN-based application
│   └── word_app.py               # Phase 2: Word recognition app
└── notebooks/
    ├── train_cnn_model.ipynb    # Phase 1: Train CNN
    └── train_word_model.ipynb   # Phase 2: Train word model
```

## Features

**Phase 1 (CNN):**
- Works with my existing processed images
- No webcam needed for training
- Real-time recognition with temporal smoothing
- GUI application

**Phase 2 (Landmarks + WLASL):**
- Uses the WLASL video dataset from Hugging Face
- MediaPipe Holistic for full-body landmarks
- BiLSTM for sequence recognition
- Word-level ASL recognition

