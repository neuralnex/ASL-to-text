import cv2
import numpy as np
import tkinter as tk
import os
import sys
import json
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.holistic_extractor import HolisticLandmarkExtractor
from models.sequence_classifier import load_model

class WordApplication:
    def __init__(self):
        self.extractor = HolisticLandmarkExtractor()
        self.load_model()
        
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        
        self.sequence_buffer = []
        self.sequence_length = 30
        self.prediction_threshold = 0.7
        
        self.ct = {}
        self.current_word = "None"
        
        self.setup_gui()
        self.video_loop()
    
    def load_model(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(base_dir, "models", "word_model.json")
        weights_path = os.path.join(base_dir, "models", "word_model.h5")
        mapping_path = os.path.join(base_dir, "models", "word_class_mapping.json")
        
        if not all(os.path.exists(p) for p in [json_path, weights_path, mapping_path]):
            raise FileNotFoundError("Word model not found. Train the model first using notebooks/train_word_model.ipynb")
        
        json_file = open(json_path, "r")
        model_json = json_file.read()
        json_file.close()
        
        self.model = model_from_json(model_json)
        self.model.load_weights(weights_path)
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
            self.classes = mapping['classes']
        
        print(f"Word model loaded successfully. {len(self.classes)} word classes.")
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("ASL Word Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x700")
        
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=640, height=480)
        
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="ASL Word Recognition", font=("Courier", 30, "bold"))
        
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=500, y=500)
        
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=500)
        self.T1.config(text="Detected Word:", font=("Courier", 25, "bold"))
        
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=350, y=550)
        
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=550)
        self.T2.config(text="Sentence:", font=("Courier", 25, "bold"))
        
        self.sentence = ""
        self.word_buffer = []
    
    def video_loop(self):
        ok, frame = self.vs.read()
        
        if ok:
            cv2image = cv2.flip(frame, 1)
            
            features, detected = self.extractor.extract_sequence_features(frame)
            
            if detected and features is not None:
                if len(features) < 160:
                    padding = np.zeros(160 - len(features))
                    features = np.concatenate([features, padding])
                elif len(features) > 160:
                    features = features[:160]
                
                self.sequence_buffer.append(features)
                
                if len(self.sequence_buffer) > self.sequence_length:
                    self.sequence_buffer.pop(0)
                
                if len(self.sequence_buffer) == self.sequence_length:
                    self.predict_word()
                
                annotated = self.extractor.draw_landmarks(cv2image, features) if hasattr(self.extractor, 'draw_landmarks') else cv2image
            else:
                annotated = cv2image
                if len(self.sequence_buffer) > 0:
                    self.sequence_buffer = []
            
            self.current_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            self.panel2.config(text=self.current_word, font=("Courier", 25))
            self.panel3.config(text=self.sentence, font=("Courier", 20))
        
        self.root.after(5, self.video_loop)
    
    def predict_word(self):
        sequence = np.array(self.sequence_buffer)
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        predictions = self.model.predict(sequence, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        if confidence > self.prediction_threshold:
            word = self.idx_to_class[predicted_idx]
            
            if word not in self.ct:
                self.ct[word] = 0
            
            self.ct[word] += 1
            
            if self.ct[word] > 10:
                if word != self.current_word:
                    if self.current_word != "None" and self.current_word not in self.word_buffer:
                        self.word_buffer.append(self.current_word)
                    
                    self.current_word = word
                    self.ct = {}
                    
                    if len(self.word_buffer) >= 3:
                        self.sentence = " ".join(self.word_buffer)
                        self.word_buffer = []
    
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
        self.extractor.close()

if __name__ == '__main__':
    print("Starting ASL Word Recognition Application...")
    print("Make sure you have trained the word model first!")
    app = WordApplication()
    app.root.mainloop()

