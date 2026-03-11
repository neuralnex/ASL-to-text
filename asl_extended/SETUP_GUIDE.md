# Setup and Running Guide

## Phase 1: CNN-Based Fingerspelling - Running Steps

### Step 1: Install Dependencies

```bash
cd asl_extended
pip install --break-system-packages -r requirements.txt
```

### Step 2: Train the CNN Model

**I use my existing dataset, so I don’t need new data collection for Phase 1.**

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

In Phase 2, I extend the system to recognize full ASL words using **video sequences** (motion over time), not single images.

**I use the WLASL dataset from Hugging Face, so I can start Phase 2 even if I cannot collect new data.**

### Prerequisites

1. Complete Phase 1 successfully
2. Install additional dependencies:
   ```bash
   pip install --break-system-packages fiftyone
   ```

### Phase 2 Implementation Steps

#### Step 1: Download WLASL and Extract Landmarks

I run the loader that downloads WLASL and converts videos into landmark sequences:

```bash
cd asl_extended
python3 data/wlasl_loader.py
```

This will:
- Load WLASL via FiftyOne + Hugging Face
- Filter to my selected words
- Extract MediaPipe landmarks from sampled video frames
- Save sequences to `data/wlasl_landmarks/`

#### Step 2: Landmark Extraction Details

I use `utils/holistic_extractor.py` to turn frames into numeric features (hands and pose; a small face subset when available). This produces a consistent feature vector per frame, which is what the sequence model learns from.

#### Step 3: Train the Word Model

After landmarks exist in `data/wlasl_landmarks/`, I train the sequence model:

```bash
jupyter notebook notebooks/train_word_model.ipynb
```

#### Step 4: Run the Word App

After training, I run the Phase 2 app:

```bash
python3 app/word_app.py
```

### Notes for Phase 2

- WLASL is a **video** dataset. I rely on motion, which is why Phase 2 uses sequence models.
- The first run can be slow because it downloads videos and extracts landmarks.
- For faster iteration, I start with fewer words and a smaller `max_samples_per_word`, then expand.

### Phase 2 File Structure

```
asl_extended/
├── data/
│   ├── wlasl_loader.py          # NEW: WLASL dataset loader
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

