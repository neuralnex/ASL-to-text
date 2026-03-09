## Sign Language to Text Conversion

### 1. Project Overview

This repository contains Zion's final year project: a real‑time system that converts American Sign Language (ASL) fingerspelling into English text using computer vision and deep learning.  
The system is designed to support basic communication between Deaf or Hard‑of‑Hearing users and non‑signers by recognizing static hand gestures for the 26 letters of the alphabet and composing them into words and sentences.

The current implementation focuses on:
- Static ASL fingerspelling (A–Z and a blank/no‑sign class).
- Real‑time recognition from a standard webcam.
- A desktop graphical user interface (GUI) that displays the live video, the segmented hand region, the predicted character, the current word, and the full sentence.

### 2. Problem Statement and Objectives

#### 2.1 Problem Statement

Sign language is a rich visual language, but most hearing people do not know it, and human interpreters are not always available. This creates a communication barrier in everyday situations such as education, healthcare, and public services.  
The goal of this project is to prototype an assistive tool that can recognize ASL fingerspelling in real time and translate it into text, reducing the dependence on a human interpreter for simple interactions.

#### 2.2 Project Objectives

- Build a **custom dataset** of ASL fingerspelling images (A–Z and blank) using a standard webcam.
- Design and train a **Convolutional Neural Network (CNN)** to classify hand gestures into 27 classes (blank + 26 letters).
- Achieve **high accuracy** on the collected dataset while maintaining **real‑time performance** on a typical laptop.
- Implement a **GUI application** that:
  - Captures video from the webcam.
  - Extracts a region of interest (ROI) for the hand.
  - Applies image processing to segment the hand.
  - Runs the trained model for live prediction.
  - Builds words and sentences from consecutive predictions.
- Evaluate the system and discuss its **limitations** and **future improvements** for real‑world deployment.

### 3. System Overview

At a high level, the system consists of four main components:

1. **Dataset Creation**
   - Directory structure for training and testing data under `dataSet/`.
   - Scripts to capture labeled images for each class using a webcam.

2. **Pre‑processing and Image Processing**
   - Mirror the webcam frame for a more natural user experience.
   - Define a fixed **Region of Interest (ROI)** where the user places their hand.
   - Convert the ROI to grayscale, apply Gaussian blur, and then adaptive thresholding followed by Otsu’s thresholding to obtain a clean binary image of the hand.

3. **Model Training (CNN)**
   - Use TensorFlow/Keras to train a CNN on the pre‑processed dataset.
   - Input: grayscale images of size `128 × 128 × 1`.
   - Output: probability distribution over 27 classes (blank + A–Z) using a softmax layer.

4. **Real‑Time Application**
   - Load the trained model and additional disambiguation models from the `Models/` directory.
   - Continuously capture frames from the webcam, apply the same preprocessing as in training, and run inference.
   - Use temporal smoothing and specialized models to improve stability for visually similar letters.
   - Display the predictions and suggestions in a Tkinter‑based GUI.

### 4. Dataset and Data Collection

The dataset used in this project is self‑collected.

- **Folder Structure**
  - `dataSet/trainingData/0, A, B, ..., Z`
  - `dataSet/testingData/0, A, B, ..., Z`
- **Collection Scripts**
  - `FoldersCreation.py`  
    Creates the `dataSet/` directory structure if it does not exist.
  - `TrainingDataCollection.py`  
    Captures images from the webcam and saves them to the appropriate subfolder under `dataSet/trainingData/` based on the key pressed by the user (`0`, `a`–`z`).
  - `TestingDataCollection.py`  
    Similar to the training script, but saves images under `dataSet/testingData/`.

Each frame:
- Is flipped horizontally.
- Shows the current count of images per class on the screen.
- Draws a blue rectangle indicating the ROI.
- Applies grayscale conversion, Gaussian blur, adaptive thresholding, and Otsu’s thresholding to generate a binary hand image used for training.

### 5. Model Architecture and Training

The main model is implemented and trained in `Models/Model.ipynb`.

- **Data Generation**
  - Uses `ImageDataGenerator` to:
    - Rescale pixel values to `[0, 1]`.
    - Apply data augmentation (shear, zoom, horizontal flip) for training.
  - Loads images from:
    - `dataSet/trainingData` (training set),
    - `dataSet/testingData` (test set),
    - with target size `(128, 128)` and grayscale color mode.

- **CNN Architecture (summary)**
  - Two convolutional layers (32 filters, kernel size 3×3, ReLU activation) each followed by max‑pooling.
  - Flatten layer.
  - Fully connected layers:
    - Dense(128) + ReLU
    - Dropout(0.40)
    - Dense(96) + ReLU
    - Dropout(0.40)
    - Dense(64) + ReLU
    - Final Dense(27) + softmax.

- **Training Setup**
  - Loss: categorical cross‑entropy.
  - Optimizer: Adam.
  - Metric: accuracy.
  - Trained for 5 epochs on the collected dataset.

The trained model is saved as:
- `Models/model_new.json` (model architecture in JSON format).
- `Models/model_new.h5` (trained weights).

In addition, three **specialized models** are trained for visually similar letter groups:
- `Models/model-bw_dru.*` for `{D, R, U}`.
- `Models/model-bw_tkdi.*` for `{T, K, D, I}`.
- `Models/model-bw_smn.*` for `{S, M, N}`.

These models are used in the application as a second layer of decision making when the main model predicts one of these ambiguous letters.

### 6. Real‑Time Application

The main application is implemented in `Application.py`.

Key responsibilities:
- Initialize the webcam (`cv2.VideoCapture(0)`).
- Load the main model and the three specialized models from the `Models/` directory.
- Create a Tkinter GUI window with:
  - A panel for the live webcam feed with the ROI drawn.
  - A panel for the processed binary image of the ROI.
  - Text fields for the current character, the current word, and the sentence.
  - Buttons showing spelling suggestions.

Runtime pipeline:
1. Read a frame from the webcam and flip it horizontally.
2. Draw the fixed ROI and crop the ROI from the frame.
3. Convert the ROI to grayscale, apply Gaussian blur, adaptive thresholding, and Otsu’s thresholding.
4. Resize the processed image to `128 × 128` and feed it into the main CNN.
5. If the predicted letter is in one of the ambiguous sets, re‑evaluate using the corresponding specialized model.
6. Use a counter for each class to enforce temporal stability (the same prediction must be sustained for multiple frames before it is accepted).
7. Treat the `blank` class as a delimiter between words:
   - When a stable `blank` is detected, the current word is appended to the sentence and reset.
8. Use Hunspell and pyenchant for simple spelling suggestions and render them as clickable buttons in the GUI.

This design allows the system to run live on a typical laptop while smoothing out noisy frame‑to‑frame predictions.

### 7. Setup and Usage

#### 7.1 Software Requirements

- Python 3.8 or later is recommended.
- Required Python libraries (install via `pip`):
  - `numpy`
  - `opencv-python`
  - `tensorflow`
  - `keras`
  - `tk` (Tkinter)
  - `Pillow`
  - `pyenchant`
  - `cyhunspell`

You can install them with:

```bash
pip install --upgrade pip
pip install numpy opencv-python tensorflow keras tk Pillow pyenchant cyhunspell
```

#### 7.2 Running the Project

1. **Create the dataset structure (first time only):**
   - Run `FoldersCreation.py` to create the `dataSet/` directory layout.

2. **Collect training and testing images (optional if you use the provided dataset):**
   - Run `TrainingDataCollection.py` and capture samples for each class.
   - Run `TestingDataCollection.py` to collect test images.

3. **Train the model (optional if you use the provided models):**
   - Open `Models/Model.ipynb` in Jupyter Notebook.
   - Adjust dataset paths if needed.
   - Run all cells to train and save the CNN model.

4. **Run the real‑time application:**

```bash
python Application.py
```

Place your hand inside the on‑screen ROI and perform ASL fingerspelling gestures.  
The GUI will display the current character, the accumulated word, the sentence, and spelling suggestions.

### 8. Results, Limitations and Future Work

#### 8.1 Results

On the collected dataset, the system achieves:
- High accuracy on the training and test sets for the 27‑class classification problem.
- Improved performance for confusing letters when the second‑layer models (`D/R/U`, `T/K/D/I`, `S/M/N`) are used.

These results demonstrate that the approach is effective on data collected under similar conditions (same camera, background, and signer).

#### 8.2 Limitations

- The dataset is limited in terms of:
  - Number of different users (hand shapes, skin tones).
  - Backgrounds and lighting conditions.
  - Camera types and resolutions.
- The system focuses only on **static fingerspelling** (A–Z) and does not handle:
  - Dynamic signs,
  - Whole‑word signs,
  - Facial expressions or body posture.
- Real‑world performance may degrade when:
  - Used by new signers who were not part of the training data.
  - Used in environments very different from the training setup.

#### 8.3 Future Work

Possible extensions and improvements include:
- Collecting a larger and more diverse dataset across multiple users and environments.
- Using modern hand‑tracking and landmark‑based methods (e.g. hand keypoints) to improve robustness to lighting and background.
- Exploring lightweight architectures suitable for deployment on mobile and embedded devices.
- Extending the system to:
  - Dynamic gestures,
  - Word‑level and phrase‑level signs,
  - Multi‑modal inputs (e.g. combining hand shape with lip reading or body pose).
- Integrating the system into real‑world scenarios such as classroom assistance or simple kiosk‑based interactions.

### 9. Acknowledgements

This project builds on well‑known concepts from computer vision and deep learning, including convolutional neural networks, image preprocessing, and data augmentation.  
Zion acknowledges **Nikhil Gupta** (2021) for laying the foundation of this project with the original fingerspelling recognition system.  
Zion also acknowledges the open‑source community for providing the core tools (Python, TensorFlow, Keras, OpenCV, Tkinter, and related libraries) that made this work possible.

## License

Copyright (c) 2021 Nikhil Gupta (original foundation)  
Copyright (c) 2026 Zion (extended development)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

  
