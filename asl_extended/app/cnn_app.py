import cv2
import numpy as np
import tkinter as tk
import os
import sys
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
import operator
from string import ascii_uppercase

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.image_processor import process_frame_for_cnn
from data.cnn_data_loader import CNNDataLoader

class CNNApplication:
    def __init__(self):
        self.loader = CNNDataLoader()
        self.class_to_idx, self.idx_to_class = self.loader.get_class_mapping()
        
        self.load_model()
        
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        self.setup_gui()
        self.video_loop()
    
    def load_model(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(base_dir, "models", "cnn_model.json")
        weights_path = os.path.join(base_dir, "models", "cnn_model.h5")
        
        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            parent_json = os.path.join(base_dir, "..", "Models", "model_new.json")
            parent_weights = os.path.join(base_dir, "..", "Models", "model_new.h5")
            
            if os.path.exists(parent_json) and os.path.exists(parent_weights):
                json_path = parent_json
                weights_path = parent_weights
            else:
                raise FileNotFoundError("Model files not found. Train the model first.")
        
        json_file = open(json_path, "r")
        model_json = json_file.read()
        json_file.close()
        
        self.model = model_from_json(model_json)
        self.model.load_weights(weights_path)
        print("CNN Model loaded successfully")
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("ASL CNN-Based Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")
        
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)
        
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)
        
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="ASL CNN Recognition", font=("Courier", 30, "bold"))
        
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)
        
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character:", font=("Courier", 30, "bold"))
        
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)
        
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word:", font=("Courier", 30, "bold"))
        
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)
        
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence:", font=("Courier", 30, "bold"))
        
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
    
    def video_loop(self):
        ok, frame = self.vs.read()
        
        if ok:
            cv2image = cv2.flip(frame, 1)
            
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            
            cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            
            self.current_image = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            processed = process_frame_for_cnn(frame, (x1, y1, x2, y2))
            
            if processed is not None:
                self.predict(processed)
                
                vis_image = (processed * 255).astype(np.uint8).reshape(128, 128)
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGBA)
                self.current_image2 = Image.fromarray(vis_image)
                imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2)
            
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))
        
        self.root.after(5, self.video_loop)
    
    def predict(self, processed_image):
        result = self.model.predict(processed_image.reshape(1, 128, 128, 1), verbose=0)
        
        prediction = {}
        for idx, class_name in self.idx_to_class.items():
            prediction[class_name] = result[0][idx]
        
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        
        self.ct[self.current_symbol] += 1
        
        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol
    
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Starting CNN-Based ASL Application...")
    app = CNNApplication()
    app.root.mainloop()

