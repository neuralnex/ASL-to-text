# Starting MongoDB for Phase 2

MongoDB is required for FiftyOne to load the WLASL dataset. These are the steps I follow on my machine:

## 1. Start MongoDB Service

**On Debian/Ubuntu/Kali Linux:**
```bash
sudo systemctl start mongodb
sudo systemctl enable mongodb  # Optional: enable auto-start on boot
```

**Check if MongoDB is running:**
```bash
sudo systemctl status mongodb
```

You should see `Active: active (running)` if it's working.

## 2. Verify MongoDB is Accessible

```bash
mongod --version
```

This should show the MongoDB version (e.g., `db version v7.0.14`).

## 3. Test WLASL Loader

Once MongoDB is running, I test the loader:

```bash
cd asl_extended
python3 -c "from data.wlasl_loader import WLASLLoader; print('Success!')"
```

## 4. Run WLASL Data Processing

```bash
cd asl_extended
python3 data/wlasl_loader.py
```

This will:
- Load the WLASL dataset from Hugging Face
- Filter to selected words
- Extract MediaPipe Holistic landmarks from videos
- Save landmark sequences to `data/wlasl_landmarks/`

## Troubleshooting

**If you get "Could not find 'mongod'" error:**
- Make sure MongoDB is installed: `mongod --version`
- Start the service: `sudo systemctl start mongodb`

**If you get "Connection refused" error:**
- MongoDB might not be running: `sudo systemctl start mongodb`
- Check MongoDB logs: `sudo journalctl -u mongodb`

**If MongoDB won't start:**
- Check if port 27017 is in use: `sudo lsof -i :27017`
- Check MongoDB logs: `sudo journalctl -u mongodb -n 50`

