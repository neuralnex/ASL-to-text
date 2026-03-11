# Troubleshooting Guide

## Fixed Issues

### 1. NumPy 2.x Compatibility
**Problem:** NumPy 2.2.6 incompatible with TensorFlow, MediaPipe, scipy, sklearn (compiled with NumPy 1.x)

**Solution:** 
- I pinned `numpy<2.0` in `requirements.txt`
- I downgraded to `numpy==1.26.4`
- I also downgraded `opencv-python<4.12.0` (newer versions require NumPy 2.x)
- I pinned `scipy<1.12.0` (scipy 1.13+ expects NumPy 2.x)

**Install:**
```bash
pip install --break-system-packages "numpy<2.0" "scipy<1.12.0" "opencv-python<4.12.0"
```

### 2. MediaPipe API Change
**Problem:** MediaPipe 0.10.x uses tasks API, not solutions API

**Solution:**
- I updated `holistic_extractor.py` to support both APIs
- It downloads the default hand landmarker model automatically on first use
- Model saved to `models/mediapipe/hand_landmarker.task`

**If model download fails:**
```bash
cd asl_extended
mkdir -p models/mediapipe
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o models/mediapipe/hand_landmarker.task
```

### 3. SciPy/NumPy Binary Incompatibility
**Problem:** `ValueError: numpy.dtype size changed, may indicate binary incompatibility` when importing fiftyone

**Solution:**
- My system scipy was compiled against a different NumPy version
- I installed compatible versions: `numpy==1.26.4` and `scipy<1.12.0`
- `holistic_extractor.py` automatically downloads the MediaPipe model if it’s missing

**Fix:**
```bash
pip install --break-system-packages "numpy==1.26.4" "scipy<1.12.0" --force-reinstall
```

### 4. MongoDB Required for FiftyOne
**Problem:** `ServiceExecutableNotFound: Could not find 'mongod'` or `FiftyOneConfigError: MongoDB could not be installed`

**Solution:**
FiftyOne requires MongoDB to manage datasets. This is what I install:

**On Debian/Ubuntu/Kali Linux:**
```bash
sudo apt update
sudo apt install mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

**On Fedora/RHEL:**
```bash
sudo dnf install mongodb-server
sudo systemctl start mongod
sudo systemctl enable mongod
```

**On macOS (with Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**On Windows:**
1. Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
2. Run the installer
3. MongoDB will start automatically as a service

**Verify Installation:**
```bash
mongod --version
```

**Start MongoDB (if not running):**
```bash
# On systemd-based systems (most Linux distributions):
sudo systemctl start mongodb
sudo systemctl enable mongodb  # Enable auto-start on boot

# Check status:
sudo systemctl status mongodb

# On systems without systemd:
sudo service mongodb start

# Manual start (if needed):
mongod --dbpath /var/lib/mongodb
```

**Important:** MongoDB must be running before using the WLASL loader. If you see "Could not find 'mongod'" errors, make sure MongoDB is started.

After installing and starting MongoDB, the WLASL loader should work.

## Current Status

✅ NumPy compatibility fixed (1.26.4)
✅ SciPy compatibility fixed (<1.12.0)
✅ MediaPipe tasks API implemented
✅ Model file auto-download implemented
✅ WLASL loader imports successfully
✅ MongoDB installed and running
✅ FiftyOne can connect to MongoDB

## Testing Phase 2 Components

**Test WLASL Loader:**
```bash
cd asl_extended
python3 -c "from data.wlasl_loader import WLASLLoader; print('Success')"
```

**Test Holistic Extractor:**
```bash
cd asl_extended
python3 -c "from utils.holistic_extractor import HolisticLandmarkExtractor; e = HolisticLandmarkExtractor(); print('Success')"
```

