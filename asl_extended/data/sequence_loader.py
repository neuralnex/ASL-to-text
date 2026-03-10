import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class SequenceDataLoader:
    def __init__(self, data_dir='data/wlasl_landmarks', sequence_length=30):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_class_mapping()
    
    def _load_class_mapping(self):
        if not os.path.exists(self.data_dir):
            return
        
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
    
    def load_data(self):
        X = []
        y = []
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return None, None, None
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            class_idx = self.class_to_idx[class_name]
            
            for file in files:
                filepath = os.path.join(class_dir, file)
                sequence = np.load(filepath)
                
                if sequence.shape[0] < 10:
                    continue
                
                if sequence.shape[0] > self.sequence_length:
                    sequence = sequence[:self.sequence_length]
                elif sequence.shape[0] < self.sequence_length:
                    padding = np.zeros((self.sequence_length - sequence.shape[0], sequence.shape[1]))
                    sequence = np.vstack([sequence, padding])
                
                X.append(sequence)
                y.append(class_idx)
        
        if len(X) == 0:
            print("No data found. Run wlasl_loader.py first to extract landmarks.")
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        y_categorical = to_categorical(y, num_classes=len(self.classes))
        
        print(f"Loaded {len(X)} sequences")
        print(f"Sequence shape: {X.shape[1:]} (length={X.shape[1]}, features={X.shape[2]})")
        print(f"Number of classes: {len(self.classes)}")
        
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

