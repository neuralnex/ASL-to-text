# Full System Plan: Real-World ASL Recognition with Motion and Facial Expressions

**Engineer:** Zion  
**Project:** Sign Language to Text Conversion - Extended System  
**Date:** 2026  
**Foundation:** This project extends the original work by Nikhil Gupta (2021), who developed the initial fingerspelling recognition system.

---

## Executive Summary

This document outlines the comprehensive plan to evolve the current **fingerspelling-only system** into a **full-featured American Sign Language (ASL) recognition system** capable of recognizing:
- **Hand signs** (word-level ASL vocabulary)
- **Motion and trajectory** (dynamic gestures)
- **Facial expressions** (non-manual markers for grammar)

The plan leverages the existing foundation (CNN-based fingerspelling) and extends it using modern computer vision techniques, sequence models, and large-scale datasets.

---

## Current System Capabilities

### What We Have Now
- ✅ **Static fingerspelling recognition** (A-Z + blank)
- ✅ **Real-time webcam processing** with GUI
- ✅ **Custom dataset** (hand-cropped images from wrist to palm)
- ✅ **CNN model** achieving high accuracy on collected data
- ✅ **Temporal smoothing** for stable predictions

### Current Limitations
- ❌ Only recognizes **static handshapes** (no motion)
- ❌ Dataset limited to **hand-only images** (no body/face context)
- ❌ Cannot handle **word-level ASL signs**
- ❌ No **facial expression** recognition
- ❌ Limited generalization to new users/environments

---

## Target System Architecture

### Phase 1: Landmark-Based Fingerspelling (Foundation Upgrade)

**Goal:** Modernize the current fingerspelling system using hand landmarks instead of raw pixels.

**Technical Approach:**
- Replace ROI-based image processing with **MediaPipe Hands**
- Extract **21 hand landmarks** per frame (normalized coordinates)
- Train a lightweight **MLP classifier** on landmark features
- Maintain compatibility with existing GUI and real-time pipeline

**Benefits:**
- More robust to lighting and background changes
- Faster inference (smaller feature space)
- Better generalization across users
- Foundation for motion-based recognition

**Deliverables:**
- Updated `Application.py` with MediaPipe integration
- New training script for landmark-based model
- Performance comparison: old CNN vs. new landmark model

**Timeline:** 2-3 weeks

---

### Phase 2: Word-Level ASL Recognition (Motion + Hands)

**Goal:** Recognize common ASL words using sequences of hand/body landmarks.

**Technical Approach:**
- Use **MediaPipe Holistic** to extract:
  - 21 hand landmarks (both hands)
  - 33 body landmarks (upper body, arms)
  - 468 face landmarks (for future use)
- Collect or use **WLASL dataset** (Word-Level ASL) via FiftyOne/Hugging Face
- Select **50-100 common ASL words** for initial prototype:
  - HELLO, THANK YOU, SORRY, PLEASE, YES, NO, HELP, WATER, BATHROOM, etc.
- Build **sequence model**:
  - Input: sequences of 16-32 frames (landmark vectors)
  - Architecture: **BiLSTM** or **GRU** (2-3 layers, ~128-256 units)
  - Output: word class probabilities
- Integrate into live application:
  - Mode switching: "Fingerspelling" vs. "Word Recognition"
  - Frame buffer for sequence processing

**Data Strategy:**
- Use **WLASL dataset** (2,000 words, 11,980 samples) via:
  ```python
  import fiftyone as fo
  import fiftyone.utils.huggingface as fouh
  dataset = fouh.load_from_hub("Voxel51/WLASL")
  ```
- Filter to selected word subset
- Extract landmarks from all video clips
- Split: train/validation/test (user-based split if possible)
- Augment with own recordings from diverse signers

**Model Architecture:**
```
Input: [batch_size, sequence_length=30, features=hand_landmarks + body_landmarks]
  ↓
BiLSTM(128 units) → Dropout(0.3)
  ↓
BiLSTM(64 units) → Dropout(0.3)
  ↓
Dense(128) → ReLU → Dropout(0.4)
  ↓
Dense(num_words) → Softmax
```

**Deliverables:**
- Landmark extraction pipeline from WLASL videos
- Trained BiLSTM model for word recognition
- Updated GUI with word recognition mode
- Evaluation metrics: accuracy, confusion matrix, per-word performance

**Timeline:** 4-6 weeks

---

### Phase 3: Facial Expression Recognition (Non-Manual Markers)

**Goal:** Detect facial expressions that convey grammatical information in ASL.

**Technical Approach:**
- Extract **face landmarks** from MediaPipe Holistic (468 points)
- Derive **facial features**:
  - Eyebrow raise (distance between eyebrows and eyes)
  - Mouth shape (open/closed, smile/frown)
  - Head tilt (left/right/neutral)
  - Eye gaze direction
- Train a **small classifier** for expression categories:
  - NEUTRAL
  - QUESTION (eyebrow raise)
  - NEGATION (head shake, negative expression)
  - EMPHASIS (exaggerated features)
- Integrate expression predictions:
  - Modify text output (add "?" for questions)
  - Use as additional context for word disambiguation

**Model Architecture:**
```
Face Landmarks (468 points) → Normalize
  ↓
MLP: Dense(128) → ReLU → Dropout(0.3)
  ↓
Dense(64) → ReLU → Dropout(0.3)
  ↓
Dense(num_expressions) → Softmax
```

**Deliverables:**
- Facial expression classifier
- Integration with word recognition pipeline
- Updated GUI showing detected expressions

**Timeline:** 2-3 weeks

---

### Phase 4: Multi-Modal Fusion and Continuous Sign Recognition

**Goal:** Combine hand, body, and face features for robust recognition of continuous signing.

**Technical Approach:**
- **Multi-stream architecture**:
  - Stream 1: Hand landmarks → Hand feature extractor (MLP)
  - Stream 2: Body landmarks → Body feature extractor (MLP)
  - Stream 3: Face landmarks → Face feature extractor (MLP)
  - Stream 4: Facial expression classifier output
- **Fusion layer**:
  - Concatenate all stream outputs
  - Feed into BiLSTM for temporal modeling
  - Output: sequence of sign labels (for continuous signing)
- **Sequence-to-sequence model** (optional, advanced):
  - Use Transformer encoder-decoder for full sentence recognition
  - Requires larger dataset and more compute

**Deliverables:**
- Multi-stream feature extraction pipeline
- Fused BiLSTM model
- Continuous signing recognition demo

**Timeline:** 4-6 weeks (advanced)

---

## Dataset Strategy

### Primary Dataset: WLASL (Word-Level ASL)

**Source:** Hugging Face Hub (`Voxel51/WLASL`)  
**Size:** 11,980 video samples, 2,000 word classes  
**License:** Computational Use of Data Agreement (C-UDA) - academic use only

**Usage:**
- Load via FiftyOne: `fouh.load_from_hub("Voxel51/WLASL")`
- Filter to common words subset (50-100 words)
- Extract MediaPipe landmarks from all videos
- Split into train/validation/test sets

### Secondary Datasets (Research/Reference)

- **MS-ASL:** Large-scale word-level ASL dataset
- **How2Sign:** Multimodal ASL corpus with multiple views
- **PHOENIX-14T:** German Sign Language (for methodology reference)

### Custom Data Collection

**Purpose:** Improve generalization to real-world deployment scenarios

**Protocol:**
- Record videos from **10-20 diverse signers**:
  - Different skin tones, hand sizes, signing styles
  - Various backgrounds (plain wall, office, outdoor)
  - Different lighting conditions
- **Target signs:** 50-100 common ASL words
- **Format:** Short video clips (1-3 seconds per sign)
- **Annotation:** Label each clip with sign class

**Tools:**
- Modified version of `TrainingDataCollection.py` for video capture
- CVAT or Label Studio for annotation
- MediaPipe for landmark extraction and storage

---

## Technical Stack

### Core Libraries
- **MediaPipe Holistic:** Hand, body, and face landmark extraction
- **TensorFlow/Keras:** Model training and inference
- **FiftyOne:** Dataset management and visualization
- **OpenCV:** Video processing and GUI
- **NumPy:** Numerical operations

### Model Frameworks
- **Keras/TensorFlow:** For BiLSTM and MLP models
- **Hugging Face Transformers:** (Optional) Pre-trained models for reference
- **ONNX/TensorFlow Lite:** (Future) Model optimization for deployment

### Development Tools
- **Jupyter Notebooks:** Model experimentation
- **Git:** Version control
- **Python 3.8+:** Primary language

---

## Implementation Roadmap

### Milestone 1: Foundation Upgrade (Weeks 1-3)
- [ ] Integrate MediaPipe Hands into `Application.py`
- [ ] Train landmark-based fingerspelling classifier
- [ ] Compare performance: CNN vs. landmark model
- [ ] Update documentation

### Milestone 2: Word Recognition (Weeks 4-9)
- [ ] Set up WLASL dataset access via FiftyOne
- [ ] Implement landmark extraction pipeline
- [ ] Select and filter 50-100 common ASL words
- [ ] Design and train BiLSTM sequence model
- [ ] Integrate word recognition mode into GUI
- [ ] Evaluate on test set

### Milestone 3: Facial Expressions (Weeks 10-12)
- [ ] Extract face landmarks from MediaPipe Holistic
- [ ] Design facial feature extraction
- [ ] Train expression classifier
- [ ] Integrate with word recognition pipeline
- [ ] Update GUI to show expressions

### Milestone 4: Multi-Modal Fusion (Weeks 13-18)
- [ ] Implement multi-stream architecture
- [ ] Design fusion layer
- [ ] Train combined model
- [ ] Test on continuous signing sequences
- [ ] Performance optimization

### Milestone 5: Real-World Testing (Weeks 19-20)
- [ ] Collect custom dataset from diverse signers
- [ ] Fine-tune models on custom data
- [ ] Evaluate generalization performance
- [ ] Document limitations and future work

---

## Expected Outcomes

### Performance Targets
- **Fingerspelling:** Maintain >95% accuracy (upgraded landmark model)
- **Word Recognition:** Achieve 70-80% accuracy on 50-100 word vocabulary
- **Facial Expressions:** >85% accuracy on 4 expression classes
- **Real-time Performance:** <200ms latency per prediction

### Deliverables
1. **Upgraded Application:** Full-featured GUI with fingerspelling + word recognition + expressions
2. **Trained Models:** Landmark-based fingerspelling, BiLSTM word classifier, expression classifier
3. **Documentation:** Updated README, model architecture diagrams, usage guide
4. **Evaluation Report:** Performance metrics, confusion matrices, limitations analysis

---

## Challenges and Mitigation

### Challenge 1: Dataset Size and Diversity
- **Issue:** WLASL may not cover all signs needed; limited diversity
- **Mitigation:** Combine WLASL with custom data collection; use data augmentation

### Challenge 2: Real-Time Performance
- **Issue:** Sequence models (BiLSTM) slower than single-frame CNNs
- **Mitigation:** Optimize model size, use TensorFlow Lite, reduce sequence length

### Challenge 3: Generalization
- **Issue:** Models may not work well for new signers/environments
- **Mitigation:** Collect diverse training data; use user-based train/test splits

### Challenge 4: Continuous Sign Recognition
- **Issue:** Segmenting continuous signing into individual signs is complex
- **Mitigation:** Start with isolated signs; use temporal smoothing and blank detection

---

## Future Extensions (Beyond Current Scope)

1. **Full Sentence Recognition:** Sequence-to-sequence models for continuous ASL
2. **Multi-Language Support:** Extend to other sign languages (BSL, Auslan, etc.)
3. **Mobile Deployment:** Optimize for Android/iOS using TensorFlow Lite
4. **Sign-to-Speech:** Text-to-speech integration for bidirectional communication
5. **Educational Applications:** Interactive learning tools for ASL students

---

## References and Resources

### Datasets
- WLASL: https://github.com/dxli94/WLASL
- Hugging Face WLASL: https://huggingface.co/datasets/Voxel51/WLASL
- MS-ASL: Microsoft Research
- How2Sign: https://how2sign.github.io/

### Tools and Libraries
- MediaPipe: https://mediapipe.dev/
- FiftyOne: https://voxel51.com/docs/fiftyone/
- TensorFlow: https://www.tensorflow.org/
- Hugging Face: https://huggingface.co/

### Research Papers
- Li et al. (2020): "Word-level Deep Sign Language Recognition from Video"
- Transfer learning for sign language recognition (CVPR 2020)

---

## Conclusion

This plan provides a structured path from the current **fingerspelling prototype** to a **comprehensive ASL recognition system** capable of handling real-world communication needs. By leveraging modern computer vision tools (MediaPipe), large-scale datasets (WLASL), and sequence modeling techniques (BiLSTM), we can build a system that recognizes not just static handshapes, but dynamic signs with facial expressions.

The phased approach allows for incremental progress, with each milestone building on the previous one. The focus on **real-world deployment** (diverse users, various environments) ensures the final system will be practical and useful for actual communication scenarios.

---

**Document Version:** 1.0  
**Last Updated:** 2026  
**Author:** Zion  
**Foundation:** Nikhil Gupta (2021) - Original fingerspelling recognition system

