import numpy as np
import os
import cv2
from string import ascii_uppercase
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from data.image_processor import preprocess_image

class CNNDataLoader:
    def __init__(self, data_dir='../../dataSet', target_size=(128, 128)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.classes = ['blank'] + list(ascii_uppercase)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
    
    def load_data(self):
        X = []
        y = []
        
        training_dir = os.path.join(self.data_dir, 'trainingData')
        testing_dir = os.path.join(self.data_dir, 'testingData')
        
        for class_name in self.classes:
            source_class = '0' if class_name == 'blank' else class_name
            class_idx = self.class_to_idx[class_name]
            
            for data_type in ['trainingData', 'testingData']:
                class_dir = os.path.join(self.data_dir, data_type, source_class)
                if not os.path.exists(class_dir):
                    continue
                
                files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for file in files:
                    filepath = os.path.join(class_dir, file)
                    image = cv2.imread(filepath)
                    
                    if image is None:
                        continue
                    
                    processed = preprocess_image(image, target_size=self.target_size)
                    if processed is not None:
                        X.append(processed)
                        y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        y_categorical = to_categorical(y, num_classes=len(self.classes))
        
        return X, y_categorical, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_class_mapping(self):
        return self.class_to_idx, self.idx_to_class

