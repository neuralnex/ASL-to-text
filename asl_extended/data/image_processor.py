import cv2
import numpy as np

def preprocess_image(image, target_size=(128, 128), min_value=70):
    if image is None:
        return None
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, min_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    res = cv2.resize(res, target_size)
    res = res.reshape(*target_size, 1)
    res = res.astype('float32') / 255.0
    
    return res

def process_frame_for_cnn(frame, roi_coords=None):
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        roi = frame[y1:y2, x1:x2]
    else:
        h, w = frame.shape[:2]
        x1 = int(0.5 * w)
        y1 = 10
        x2 = w - 10
        y2 = int(0.5 * w)
        roi = frame[y1:y2, x1:x2]
    
    processed = preprocess_image(roi)
    return processed

