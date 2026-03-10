# Setup and Running Guide

## Phase 1: CNN-Based Fingerspelling - Running Steps

### Step 1: Install Dependencies

```bash
cd asl_extended
pip install --break-system-packages -r requirements.txt
```

### Step 2: Train the CNN Model

**Uses your existing dataset - no new data collection needed!**

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/train_cnn_model.ipynb
```

**In the notebook:**
1. Run all cells sequentially
2. The notebook will automatically load images from `../../dataSet/trainingData/` and `../../dataSet/testingData/`
3. Preprocesses images (grayscale, blur, threshold) same as original system
4. Trains CNN model on your existing data
5. Model will be saved to `models/cnn_model.json` and `models/cnn_model.h5`
6. Training history plot will be saved to `models/cnn_training_history.png`

**Note:** If you already have trained models in `../Models/`, the app will use those automatically.

### Step 3: Run the Application

Once the model is trained:

```bash
python app/cnn_app.py
```

**Features:**
- Real-time fingerspelling recognition (A-Z + blank)
- Uses your existing CNN model
- Works with processed binary images
- Word and sentence building
- Visual feedback with processed ROI display

---

## Phase 2: Word-Level ASL Recognition - Getting Started

### Overview

Phase 2 extends the system to recognize full ASL words using sequences of landmarks (motion + hands).

**Uses WLASL dataset from Hugging Face - no webcam needed!**

### Prerequisites

1. Complete Phase 1 successfully
2. Install additional dependencies:
   ```bash
   pip install --break-system-packages fiftyone
   ```

### Phase 2 Implementation Steps

#### Step 1: Set Up WLASL Dataset Access

Create a new script to download and process WLASL dataset:

```bash
# Create the script
touch data/wlasl_loader.py
```

The script should:
- Use FiftyOne to load WLASL dataset from Hugging Face
- Filter to selected word subset (50-100 common words)
- Extract MediaPipe Holistic landmarks from videos
- Save landmark sequences for training

#### Step 2: Create Holistic Landmark Extractor

Extend the landmark extractor to use MediaPipe Holistic:

```bash
# Create new extractor
touch utils/holistic_extractor.py
```

This should extract:
- 21 hand landmarks (both hands)
- 33 body landmarks (upper body)
- 468 face landmarks (for future use)

#### Step 3: Build Sequence Data Collector

Create a data collector for word-level signs:

```bash
# Create collector
touch data/word_collector.py
```

This should:
- Record video sequences (1-3 seconds per sign)
- Extract holistic landmarks frame-by-frame
- Save sequences with word labels

#### Step 4: Create Sequence Model

Build BiLSTM/GRU model for sequence classification:

```bash
# Create model
touch models/sequence_classifier.py
```

Architecture:
- Input: sequences of 16-32 frames
- BiLSTM layers (128-256 units)
- Dense layers for classification
- Output: word class probabilities

#### Step 5: Create Training Notebook

Create notebook for training word-level model:

```bash
# Create notebook
touch notebooks/train_word_model.ipynb
```

Should include:
- Data loading from WLASL + custom data
- Sequence preprocessing
- Model training
- Evaluation metrics

#### Step 6: Extend Application

Update the application to support word recognition mode:

```bash
# Create extended app
touch app/word_app.py
```

Features:
- Mode switching: Fingerspelling vs. Word Recognition
- Frame buffer for sequence processing
- Word prediction display

### Phase 2 File Structure

```
asl_extended/
├── data/
│   ├── wlasl_loader.py          # NEW: WLASL dataset loader
│   ├── word_collector.py        # NEW: Word-level data collection
│   └── sequence_loader.py       # NEW: Sequence data loading
├── models/
│   └── sequence_classifier.py   # NEW: BiLSTM/GRU model
├── utils/
│   └── holistic_extractor.py    # NEW: MediaPipe Holistic
├── app/
│   └── word_app.py              # NEW: Word recognition app
└── notebooks/
    └── train_word_model.ipynb   # NEW: Word model training
```

### Phase 2 Workflow

1. **Data Collection:**
   - Download WLASL subset (50-100 words)
   - Extract holistic landmarks from videos
   - Collect custom word samples if needed

2. **Model Training:**
   - Prepare sequence datasets
   - Train BiLSTM model
   - Evaluate on test set

3. **Integration:**
   - Add word recognition mode to app
   - Implement frame buffering
   - Test real-time performance

### Target Words for Phase 2

Start with these common ASL words:
- HELLO, THANK YOU, SORRY, PLEASE
- YES, NO, HELP, WATER
- BATHROOM, FOOD, DRINK, MORE
- GOOD, BAD, HAPPY, SAD
- I, YOU, WE, THEY
- HOW, WHAT, WHERE, WHEN
- (Expand to 50-100 words total)

### Expected Timeline

- **Week 1-2:** Dataset setup and landmark extraction
- **Week 3-4:** Model architecture and training
- **Week 5-6:** Application integration and testing

---

## Troubleshooting

### Phase 1 Issues

**Problem:** No hand detected
- **Solution:** Ensure good lighting, clear background, hand fully visible

**Problem:** Low accuracy
- **Solution:** Collect more training data (100+ samples per class)

**Problem:** Model not loading
- **Solution:** Check file paths, ensure model files exist in `models/` directory

### Phase 2 Preparation

**Problem:** WLASL dataset too large
- **Solution:** Start with small subset (20-30 words), expand gradually

**Problem:** Sequence processing slow
- **Solution:** Optimize frame buffer size, use smaller sequence length initially

---

## Next Steps After Phase 2

- Phase 3: Facial Expression Recognition
- Phase 4: Multi-Modal Fusion
- Continuous Sign Recognition

See `FURTHER_PLAN.md` for complete roadmap.

