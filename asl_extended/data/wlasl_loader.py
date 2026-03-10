import os
import sys
import cv2
import numpy as np
from string import ascii_uppercase

try:
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("FiftyOne not installed. Install with: pip install --break-system-packages fiftyone")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.holistic_extractor import HolisticLandmarkExtractor
except ImportError:
    print("Warning: holistic_extractor not available. Install mediapipe for Phase 2.")
    HolisticLandmarkExtractor = None

class WLASLLoader:
    def __init__(self, output_dir='data/wlasl_landmarks', selected_words=None):
        self.output_dir = output_dir
        self.selected_words = selected_words or self.get_default_words()
        self.extractor = HolisticLandmarkExtractor()
        self.setup_directories()
    
    def get_default_words(self):
        return [
            'HELLO', 'THANK YOU', 'SORRY', 'PLEASE', 'YES', 'NO', 'HELP',
            'WATER', 'BATHROOM', 'FOOD', 'DRINK', 'MORE', 'GOOD', 'BAD',
            'HAPPY', 'SAD', 'I', 'YOU', 'WE', 'THEY', 'HOW', 'WHAT',
            'WHERE', 'WHEN', 'WHY', 'NAME', 'NICE', 'MEET', 'AGAIN',
            'BYE', 'SORRY', 'EXCUSE', 'ME', 'PLEASE', 'THANK', 'WELCOME'
        ]
    
    def setup_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for word in self.selected_words:
            word_dir = os.path.join(self.output_dir, word.upper().replace(' ', '_'))
            if not os.path.exists(word_dir):
                os.makedirs(word_dir)
    
    def load_wlasl_dataset(self):
        if not FIFTYONE_AVAILABLE:
            raise ImportError("FiftyOne required. Install: pip install --break-system-packages fiftyone")
        
        print("Loading WLASL dataset from Hugging Face...")
        dataset = fouh.load_from_hub("Voxel51/WLASL")
        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset
    
    def filter_by_words(self, dataset):
        filtered_samples = []
        for sample in dataset:
            if hasattr(sample, 'ground_truth') and sample.ground_truth:
                label = sample.ground_truth.label
                if label and label.upper() in [w.upper() for w in self.selected_words]:
                    filtered_samples.append(sample)
        print(f"Filtered to {len(filtered_samples)} samples for selected words")
        return filtered_samples
    
    def extract_landmarks_from_video(self, video_path, num_frames=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)) == 0:
                landmarks, detected = self.extractor.extract_sequence_features(frame)
                if detected:
                    landmarks_sequence.append(landmarks)
                frames.append(frame)
            
            frame_count += 1
            if len(landmarks_sequence) >= num_frames:
                break
        
        cap.release()
        
        if len(landmarks_sequence) < 10:
            return None
        
        while len(landmarks_sequence) < num_frames:
            landmarks_sequence.append(landmarks_sequence[-1])
        
        return np.array(landmarks_sequence[:num_frames])
    
    def process_wlasl_dataset(self, max_samples_per_word=50):
        if not FIFTYONE_AVAILABLE:
            print("FiftyOne not available. Cannot load WLASL dataset.")
            return
        
        dataset = self.load_wlasl_dataset()
        filtered = self.filter_by_words(dataset)
        
        word_counts = {word: 0 for word in self.selected_words}
        
        for sample in filtered:
            if not hasattr(sample, 'ground_truth') or not sample.ground_truth:
                continue
            
            label = sample.ground_truth.label.upper()
            word_key = label.replace(' ', '_')
            
            if word_key not in [w.upper().replace(' ', '_') for w in self.selected_words]:
                continue
            
            if word_counts.get(label, 0) >= max_samples_per_word:
                continue
            
            if hasattr(sample, 'filepath') and sample.filepath:
                video_path = sample.filepath
            elif hasattr(sample, 'video_path'):
                video_path = sample.video_path
            else:
                continue
            
            if not os.path.exists(video_path):
                continue
            
            landmarks_seq = self.extract_landmarks_from_video(video_path)
            
            if landmarks_seq is not None:
                word_dir = os.path.join(self.output_dir, word_key)
                count = word_counts.get(label, 0)
                filepath = os.path.join(word_dir, f"{count}.npy")
                np.save(filepath, landmarks_seq)
                word_counts[label] = count + 1
                print(f"Saved {label} sample {count}")
        
        print("\nProcessing complete!")
        for word, count in word_counts.items():
            print(f"{word}: {count} samples")
    
    def get_statistics(self):
        stats = {}
        for word in self.selected_words:
            word_key = word.upper().replace(' ', '_')
            word_dir = os.path.join(self.output_dir, word_key)
            if os.path.exists(word_dir):
                count = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
                stats[word] = count
        return stats

if __name__ == '__main__':
    loader = WLASLLoader()
    loader.process_wlasl_dataset(max_samples_per_word=50)

