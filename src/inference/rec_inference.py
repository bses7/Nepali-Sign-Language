import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont 

from src.utils import PoseExtractor
from src.models.sign_classifier import NSLClassifier
from src.data_preprocessing.tokenizer import NSLTokenizer

class NSLRecognizer:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = NSLTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = NSLClassifier(num_classes=checkpoint['num_classes']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.frame_buffer = deque(maxlen=30)
        self.pred_history = deque(maxlen=10) 

        self.font_path = "C:/Windows/Fonts/Nirmala.ttc" 
        if not Path(self.font_path).exists():
            self.font_path = "arial.ttf"

    def draw_nepali(self, img, text, position=(20, 10), font_size=45, color=(0, 255, 0)):
        """Helper to draw Unicode text on OpenCV image."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
            
        draw.text(position, text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def preprocess_landmarks(self, pose, lh, rh):
        """Standardizes coordinates exactly like the training data."""
        pose_coords = pose[:, :3] / 0.5
        lh_coords = lh * 5.0
        rh_coords = rh * 5.0

        mid_x = (pose_coords[11, 0] + pose_coords[12, 0]) / 2
        mid_y = (pose_coords[11, 1] + pose_coords[12, 1]) / 2
        pose_coords[:, 0] -= mid_x
        pose_coords[:, 1] -= mid_y

        return np.concatenate([pose_coords.flatten(), lh_coords.flatten(), rh_coords.flatten()])

    def predict(self):
        if len(self.frame_buffer) < 30:
            return "प्रतीक्षा गर्नुहोस् (Collecting...)"
            
        input_tensor = torch.tensor(list(self.frame_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(prob, dim=1)
            
            if confidence.item() > 0.7:
                label = self.tokenizer.idx2char[class_idx.item()]
                self.pred_history.append(label)
            else:
                return "चिन्ह देखाउनुहोस् (Show Sign)"

        if self.pred_history:
            return max(set(self.pred_history), key=self.pred_history.count)
        return "..."

def run_realtime():
    config = {'mediapipe': {'static_image_mode': False, 'model_complexity': 0, 'min_detection_confidence': 0.5}}
    extractor = PoseExtractor(config)
    
    recognizer = NSLRecognizer(
        model_path="experiments/recognition/best_recognizer.pth",
        vocab_path="vocab.json"
    )

    cap = cv2.VideoCapture(0)
    print("🚀 Real-time NSL Recognizer Active.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 

        pose, lh, rh, _, _ = extractor.process_frame(frame)
        features = recognizer.preprocess_landmarks(pose, lh, rh)
        recognizer.frame_buffer.append(features)

        prediction = recognizer.predict()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        frame = recognizer.draw_nepali(frame, f"Sign: {prediction}")

        cv2.imshow("Nepali Sign Language Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()