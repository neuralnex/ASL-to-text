# Dataset Usage Roadmap

## Current Status

### Phase 1: Fingerspelling (Current)
- **Dataset:** My existing processed images (`dataSet/trainingData/`, `dataSet/testingData/`)
- **Format:** Binary/grayscale cropped hand images (310x310px)
- **Status:** ✅ Working with CNN model (95-98% accuracy)
- **MediaPipe Landmarks:** ❌ Not compatible (images are processed, not natural)

**Decision:** I keep using the original CNN approach for Phase 1 because it fits my existing dataset.

---

## Phase 2: Word-Level ASL Recognition

### When to Use WLASL Dataset

**Timeline:** I start Phase 2 after Phase 1 is complete and documented.

**Purpose:**
- Train models to recognize **full ASL words** (not just letters)
- Learn from **motion and sequences** (not just static handshapes)
- Build foundation for real-world ASL communication

### WLASL Dataset Details

**Source:** Hugging Face Hub - `Voxel51/WLASL`
- **Size:** 11,980 video samples
- **Words:** 2,000 different ASL words
- **Format:** Natural RGB videos (not processed/binary)
- **License:** Academic use only (C-UDA)

**Access:**
```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
dataset = fouh.load_from_hub("Voxel51/WLASL")
```

### Phase 2 Implementation Plan

#### Step 1: Install Dependencies
```bash
pip install --break-system-packages fiftyone
```

#### Step 2: Select Target Words (50-100 words)
Start with common words:
- Basic: HELLO, THANK YOU, SORRY, PLEASE, YES, NO
- Needs: HELP, WATER, BATHROOM, FOOD, DRINK
- Emotions: HAPPY, SAD, GOOD, BAD
- Pronouns: I, YOU, WE, THEY
- Questions: HOW, WHAT, WHERE, WHEN, WHY
- Social: NAME, NICE, MEET, AGAIN, BYE

#### Step 3: Extract Landmarks from WLASL Videos
Use `data/wlasl_loader.py`:
```bash
python data/wlasl_loader.py
```

This will:
- Load WLASL dataset from Hugging Face
- Filter to your selected words
- Extract MediaPipe Holistic landmarks (hands + body + face) from each video
- Save landmark sequences to `data/wlasl_landmarks/`

#### Step 4: Collect Custom Data (Optional but Recommended)
- Record your own videos of selected words
- Use `data/word_collector.py` (to be created)
- Ensures model works with your specific signing style/environment

#### Step 5: Train Sequence Model
- Use `notebooks/train_word_model.ipynb` (to be created)
- Train BiLSTM on landmark sequences
- Evaluate on test set

---

## Phase 2 File Structure

```
asl_extended/
├── data/
│   ├── wlasl_loader.py          # ✅ Created - Loads WLASL from Hugging Face
│   ├── word_collector.py        # ⏳ To create - Collect custom word videos
│   └── sequence_loader.py       # ⏳ To create - Load sequence data for training
├── utils/
│   └── holistic_extractor.py   # ✅ Created - Extracts holistic landmarks
├── models/
│   └── sequence_classifier.py   # ⏳ To create - BiLSTM for word recognition
└── notebooks/
    └── train_word_model.ipynb  # ⏳ To create - Training notebook
```

---

## Timeline

### Phase 1 (Current - Complete)
- ✅ Original CNN model working
- ✅ Dataset ready
- ✅ Application functional

### Phase 2 (Next Steps)
1. **Week 1:** Set up WLASL dataset access, extract landmarks for 50-100 words
2. **Week 2:** Collect custom word videos (optional)
3. **Week 3-4:** Build and train BiLSTM sequence model
4. **Week 5:** Integrate word recognition into application
5. **Week 6:** Testing and evaluation

---

## Why WLASL for Phase 2?

1. **Natural Videos:** Real signers, natural lighting, diverse backgrounds
2. **Motion Data:** Full signing sequences, not just static frames
3. **Large Scale:** 11,980 samples provides good training data
4. **Academic Standard:** Widely used in research, good for citations
5. **Complementary:** Works with your custom data for better generalization

---

## Next Actions

1. **Complete Phase 1 documentation** (current system)
2. **Install FiftyOne:** `pip install --break-system-packages fiftyone`
3. **Test WLASL access:** Run `data/wlasl_loader.py` to verify dataset loading
4. **Select word vocabulary:** Choose 50-100 words for initial prototype
5. **Extract landmarks:** Process WLASL videos to landmark sequences
6. **Build sequence model:** Create BiLSTM architecture
7. **Train and evaluate:** Follow Phase 2 training pipeline

---

## Important Notes

- **Phase 1 (Fingerspelling):** Use your existing CNN - it works great!
- **Phase 2 (Words):** Use WLASL + landmarks - designed for motion/sequences
- **Don't mix:** Your processed images won't work with MediaPipe landmarks
- **New data needed:** Phase 2 requires natural RGB videos, not processed images

