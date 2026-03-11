import os
import sys
import cv2
import numpy as np
from string import ascii_uppercase

try:
    import os
    os.environ['PATH'] = '/usr/bin:/usr/local/bin:' + os.environ.get('PATH', '')
    
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    
    fo.config.database_uri = "mongodb://localhost:27017"
    fo.config.database_name = "fiftyone"
    
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("FiftyOne not installed. Install with: pip install --break-system-packages fiftyone")
except Exception as e:
    if "mongod" in str(e).lower() or "database" in str(e).lower() or "mongodb" in str(e).lower():
        FIFTYONE_AVAILABLE = False
        print("\n" + "="*60)
        print("ERROR: MongoDB is required for FiftyOne")
        print("="*60)
        print("\nFiftyOne requires MongoDB to manage datasets.")
        print("Please install MongoDB:")
        print("\n  On Debian/Ubuntu/Kali:")
        print("    sudo apt update")
        print("    sudo apt install mongodb")
        print("    sudo systemctl start mongodb")
        print("\n  On other systems:")
        print("    Visit: https://www.mongodb.com/try/download/community")
        print("\nAfter installing MongoDB, restart this script.")
        print("="*60 + "\n")
    else:
        FIFTYONE_AVAILABLE = False
        raise

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
            raise RuntimeError(
                "FiftyOne not available. This may be due to:\n"
                "1. FiftyOne not installed: pip install --break-system-packages fiftyone\n"
                "2. MongoDB not installed (required for FiftyOne)\n"
                "   Install: sudo apt install mongodb && sudo systemctl start mongodb"
            )
        
        print("Loading WLASL dataset from Hugging Face...")
        try:
            dataset = fouh.load_from_hub("Voxel51/WLASL", persistent=False)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            is_mongo_error = (
                "mongod" in error_str or 
                "database" in error_str or 
                "mongodb" in error_str or
                "FiftyOneConfigError" in error_type or
                "ServiceExecutableNotFound" in error_type or
                "none" in error_str and "path" in error_str
            )
            
            if is_mongo_error:
                print("\n" + "="*60)
                print("ERROR: MongoDB is required for FiftyOne")
                print("="*60)
                print("\nFiftyOne requires MongoDB to manage datasets.")
                print("Please install MongoDB:")
                print("\n  On Debian/Ubuntu/Kali:")
                print("    sudo apt update")
                print("    sudo apt install mongodb")
                print("    sudo systemctl start mongodb")
                print("\n  On other systems:")
                print("    Visit: https://www.mongodb.com/try/download/community")
                print("\nAfter installing MongoDB, restart this script.")
                print("="*60 + "\n")
                raise RuntimeError("MongoDB not installed. Please install MongoDB to use FiftyOne.") from e
            else:
                print(f"\nUnexpected error: {error_type}: {e}")
                raise
        
        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset
    
    def filter_by_words(self, dataset):
        filtered_samples = []
        for sample in dataset:
            label = None
            if hasattr(sample, 'gloss') and sample.gloss:
                if hasattr(sample.gloss, 'label'):
                    label = sample.gloss.label
                else:
                    label = str(sample.gloss)
            elif hasattr(sample, 'ground_truth') and sample.ground_truth:
                if hasattr(sample.ground_truth, 'label'):
                    label = sample.ground_truth.label
                else:
                    label = str(sample.ground_truth)
            
            if label:
                label_upper = str(label).upper().strip()
                selected_upper = [w.upper().strip() for w in self.selected_words]
                if label_upper in selected_upper:
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
        
        if len(landmarks_sequence) == 0:
            return None
        
        normalized_landmarks = []
        for lm in landmarks_sequence:
            if isinstance(lm, np.ndarray):
                flat_lm = lm.flatten()
            else:
                flat_lm = np.array(lm).flatten()
            normalized_landmarks.append(flat_lm)
        
        if len(normalized_landmarks) == 0:
            return None
        
        feature_dim = len(normalized_landmarks[0])
        if feature_dim == 0:
            return None
        
        for i, lm in enumerate(normalized_landmarks):
            if len(lm) != feature_dim:
                normalized_landmarks[i] = np.zeros(feature_dim)
        
        while len(normalized_landmarks) < num_frames:
            normalized_landmarks.append(normalized_landmarks[-1].copy())
        
        landmarks_array = np.array(normalized_landmarks[:num_frames])
        
        return landmarks_array
    
    def process_wlasl_dataset(self, max_samples_per_word=50):
        if not FIFTYONE_AVAILABLE:
            print("FiftyOne not available. Cannot load WLASL dataset.")
            return
        
        dataset = self.load_wlasl_dataset()
        filtered = self.filter_by_words(dataset)
        
        word_counts = {word: 0 for word in self.selected_words}
        
        for sample in filtered:
            label = None
            if hasattr(sample, 'gloss') and sample.gloss:
                if hasattr(sample.gloss, 'label'):
                    label = sample.gloss.label
                else:
                    label = str(sample.gloss)
            elif hasattr(sample, 'ground_truth') and sample.ground_truth:
                if hasattr(sample.ground_truth, 'label'):
                    label = sample.ground_truth.label
                else:
                    label = str(sample.ground_truth)
            
            if not label:
                continue
            
            label = str(label).upper().strip()
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

