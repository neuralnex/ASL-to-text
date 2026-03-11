# Phase 2: Word-Level ASL Recognition - Getting Started

## Overview

In Phase 2, I extend the system to recognize full ASL words using **video sequences** and holistic landmarks (hands + pose + a small set of face points) from the WLASL dataset.

## Prerequisites

1. ✅ I have Phase 1 (CNN fingerspelling) completed
2. I install the Phase 2 dependencies:
   ```bash
   pip install --break-system-packages fiftyone mediapipe<0.11.0
   ```

## Step-by-Step Guide

### Step 1: Load WLASL Dataset and Extract Landmarks

The WLASL dataset contains 11,980 video samples of 2,000 ASL words. I extract landmarks from a selected subset of words.

```bash
cd asl_extended
python data/wlasl_loader.py
```

**What this does:**
- Downloads WLASL dataset from Hugging Face (first time only)
- Extracts MediaPipe Holistic landmarks from video frames
- Saves landmark sequences to `data/wlasl_landmarks/`
- Processes 50 samples per word by default

**Customization:**
Edit `data/wlasl_loader.py` to:
- Change `selected_words` list (default: 35 common words)
- Adjust `max_samples_per_word` (default: 50)
- Modify `num_frames` per sequence (default: 30)

**Expected output:**
```
Loading WLASL dataset from Hugging Face...
Dataset loaded: 11980 samples
Filtered to 1750 samples for selected words
Processing...
Saved HELLO sample 0
Saved HELLO sample 1
...
Processing complete!
HELLO: 50 samples
THANK_YOU: 50 samples
...
```

### Step 2: Train the Word Recognition Model

I open and run the training notebook:

```bash
jupyter notebook notebooks/train_word_model.ipynb
```

**In the notebook:**
1. Run all cells sequentially
2. The model will:
   - Load landmark sequences from `data/wlasl_landmarks/`
   - Train a BiLSTM network on sequences
   - Save model to `models/word_model.json` and `models/word_model.h5`
   - Save class mapping to `models/word_class_mapping.json`
3. Training typically takes 30-60 minutes depending on dataset size
4. Monitor validation accuracy - should reach 70-85% for good word sets

**Model Architecture:**
- Input: Sequences of 30 frames × 160 features (holistic landmarks)
- BiLSTM layers: 128 → 64 units
- Dense layers: 128 → 64 → num_classes
- Output: Word class probabilities

### Step 3: Run Word Recognition Application

Once the model is trained, I run:

```bash
python app/word_app.py
```

**Features:**
- Real-time holistic landmark detection
- Sequence buffer (30 frames) for word recognition
- Word-level predictions with confidence threshold
- Sentence building from detected words
- Visual feedback with landmark overlay

**Usage:**
- I sign words clearly in front of the camera
- I hold each sign for about 1–2 seconds
- The system builds a sentence from detected words
- Press window close button to exit

## File Structure

```
asl_extended/
├── data/
│   ├── wlasl_loader.py          # Step 1: Load WLASL, extract landmarks
│   └── sequence_loader.py        # Step 2: Load sequences for training
├── models/
│   ├── sequence_classifier.py    # BiLSTM model definition
│   ├── word_model.json           # Trained model (after Step 2)
│   ├── word_model.h5             # Model weights (after Step 2)
│   └── word_class_mapping.json   # Class labels (after Step 2)
├── utils/
│   └── holistic_extractor.py    # MediaPipe Holistic extraction
├── app/
│   └── word_app.py               # Step 3: Word recognition app
└── notebooks/
    └── train_word_model.ipynb    # Step 2: Training notebook
```

## Troubleshooting

### Issue: "FiftyOne not installed"
**Solution:**
```bash
pip install --break-system-packages fiftyone
```

### Issue: "No data found" in training notebook
**Solution:**
- Run `python data/wlasl_loader.py` first
- Check that `data/wlasl_landmarks/` contains word folders with `.npy` files

### Issue: "MediaPipe model not found"
**Solution:**
- MediaPipe 0.10.x uses tasks API and requires model file
- The holistic extractor should work without model file (uses solutions API if available)
- If issues persist, install: `pip install --break-system-packages "mediapipe<0.10.0"`

### Issue: Low accuracy in training
**Solutions:**
- Increase dataset size (more samples per word)
- Add more diverse words
- Adjust model architecture (more LSTM units, different dropout)
- Check data quality (landmarks extracted correctly)

### Issue: Poor real-time recognition
**Solutions:**
- Lower `prediction_threshold` in `word_app.py` (default: 0.7)
- Increase sequence buffer stability (adjust counter thresholds)
- Ensure good lighting and clear signing
- Sign more slowly and clearly

## Next Steps After Phase 2

1. **Evaluate Performance:**
   - Test on held-out test set
   - Measure per-word accuracy
   - Create confusion matrix

2. **Expand Vocabulary:**
   - Add more words to `selected_words` in `wlasl_loader.py`
   - Retrain model with expanded vocabulary

3. **Collect Custom Data (Optional):**
   - Record your own word videos
   - Add to dataset for better personalization

4. **Move to Phase 3:**
   - Facial expression recognition
   - Multi-modal fusion
   - Advanced grammar understanding

## Performance Expectations

- **Training Accuracy:** 75-85% (depends on word selection and dataset size)
- **Real-time FPS:** 20-30 FPS (with MediaPipe Holistic)
- **Latency:** ~100-200ms per word prediction
- **Memory:** ~500MB-1GB for model + landmarks

## Notes

- WLASL dataset is large (~several GB). First download may take time.
- Landmark extraction is CPU-intensive. Processing all videos may take hours.
- Start with 20-30 words for faster iteration, then expand.
- Word recognition requires clear, complete signs. Partial signs may not work well.

