# ASL Extended System

## Overview

This is the extended ASL recognition system with two phases:
- **Phase 1:** CNN-Based Fingerspelling Recognition (using existing processed images)
- **Phase 2:** Word-Level ASL Recognition (using WLASL dataset from Hugging Face)

## Quick Start

### Installation

```bash
pip install --break-system-packages -r requirements.txt
```

### Phase 1: CNN-Based Fingerspelling (Current)

**Uses your existing processed images - no new data collection needed!**

1. **Train Model:**
   ```bash
   jupyter notebook notebooks/train_cnn_model.ipynb
   ```
   Run all cells - uses your existing `dataSet/` images

2. **Run Application:**
   ```bash
   python app/cnn_app.py
   ```

### Phase 2: Word-Level Recognition (Ready!)

**Uses WLASL dataset from Hugging Face - no webcam needed!**

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

**See [PHASE2_START.md](PHASE2_START.md) for detailed guide.**

## Detailed Guide

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for:
- Complete setup instructions
- Troubleshooting
- Phase 2 implementation guide
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
- Works with your existing processed images
- No webcam needed for training
- Real-time recognition with temporal smoothing
- GUI application

**Phase 2 (Landmarks + WLASL):**
- Uses WLASL dataset from Hugging Face
- MediaPipe Holistic for full-body landmarks
- BiLSTM for sequence recognition
- Word-level ASL recognition

