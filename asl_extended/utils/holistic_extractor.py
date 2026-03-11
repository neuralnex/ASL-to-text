import cv2
import numpy as np
import mediapipe as mp

try:
    _ = mp.solutions.holistic
    USE_SOLUTIONS_API = True
except AttributeError:
    USE_SOLUTIONS_API = False

class HolisticLandmarkExtractor:
    def __init__(self):
        if USE_SOLUTIONS_API:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            import os
            import urllib.request
            
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mediapipe')
            os.makedirs(model_dir, exist_ok=True)
            hand_model = os.path.join(model_dir, 'hand_landmarker.task')
            
            if not os.path.exists(hand_model):
                print(f"Downloading MediaPipe hand landmarker model to {hand_model}...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    hand_model
                )
                print("Download complete.")
            
            hand_base = BaseOptions(model_asset_path=hand_model)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.hands = vision.HandLandmarker.create_from_options(hand_options)
            self.holistic = None
            self.mp_drawing = None
    
    def extract_sequence_features(self, image):
        if image is None or image.size == 0:
            return None, False
        
        h, w = image.shape[:2]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if h < 256 or w < 256:
            scale = max(256 / h, 256 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if USE_SOLUTIONS_API:
            results = self.holistic.process(image_rgb)
            features = []
            
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0.0] * 63)
            
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0.0] * 63)
            
            if results.pose_landmarks:
                upper_body_indices = [11, 12, 13, 14, 15, 16]
                for idx in upper_body_indices:
                    if idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[idx]
                        features.extend([landmark.x, landmark.y, landmark.z])
                    else:
                        features.extend([0.0] * 3)
            else:
                features.extend([0.0] * 18)
            
            if results.face_landmarks:
                key_points = [10, 151, 9, 175, 61, 291, 0, 17]
                for idx in key_points:
                    if idx < len(results.face_landmarks.landmark):
                        landmark = results.face_landmarks.landmark[idx]
                        features.extend([landmark.x, landmark.y])
                    else:
                        features.extend([0.0] * 2)
            else:
                features.extend([0.0] * 16)
            
            detected = any([
                results.left_hand_landmarks,
                results.right_hand_landmarks,
                results.pose_landmarks
            ])
            
            return np.array(features), detected
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.hands.detect(mp_image)
            
            features = []
            if results.hand_landmarks:
                for hand in results.hand_landmarks[:2]:
                    for landmark in hand:
                        features.extend([landmark.x, landmark.y, landmark.z])
                    if len(hand) < 21:
                        features.extend([0.0] * (63 - len(hand) * 3))
            else:
                features.extend([0.0] * 126)
            
            return np.array(features), len(results.hand_landmarks) > 0
    
    def close(self):
        if hasattr(self.holistic, 'close'):
            self.holistic.close()
        if hasattr(self.hands, 'close'):
            self.hands.close()

