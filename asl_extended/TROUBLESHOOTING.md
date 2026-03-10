# Troubleshooting Guide

## Fixed Issues

### 1. NumPy 2.x Compatibility
**Problem:** NumPy 2.2.6 incompatible with TensorFlow, MediaPipe, scipy, sklearn (compiled with NumPy 1.x)

**Solution:** 
- Pinned `numpy<2.0` in requirements.txt
- Downgraded to `numpy==1.26.4`
- Also downgraded `opencv-python<4.12.0` (newer versions require NumPy 2.x)

**Install:**
```bash
pip install --break-system-packages "numpy<2.0" "opencv-python<4.12.0"
```

### 2. MediaPipe API Change
**Problem:** MediaPipe 0.10.x uses tasks API, not solutions API

**Solution:**
- Updated `landmark_extractor.py` to support both APIs
- Downloads default hand landmarker model automatically
- Model saved to `models/mediapipe/hand_landmarker.task`

**If model download fails:**
```bash
cd asl_extended
mkdir -p models/mediapipe
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o models/mediapipe/hand_landmarker.task
```

## Current Status

✅ NumPy compatibility fixed
✅ MediaPipe tasks API implemented
✅ Model file downloaded
✅ Script should now work

## Running the Conversion

```bash
cd asl_extended
python3 data/image_to_landmarks.py
```

The script will:
1. Process all images from `../../dataSet/trainingData/` and `../../dataSet/testingData/`
2. Extract landmarks using MediaPipe
3. Save to `data/landmarks/` directory
4. Only save when hand is detected

**Note:** Processing may take several minutes depending on dataset size.

