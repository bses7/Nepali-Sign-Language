import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont 

import time
from src.recognition.feedback_engine import NSLFeedbackEngine

from src.utils import PoseExtractor
from src.models.sign_classifier import NSLClassifier
from src.data_preprocessing.tokenizer import NSLTokenizer

class NSLRecognizer:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = NSLTokenizer()
        self.tokenizer.load_vocab(vocab_path)

        num_classes = len(self.tokenizer.vocab)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = NSLClassifier(num_classes=num_classes).to(self.device)
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

def run_realtime(model_path, vocab_path):
    # Standard config for PoseExtractor
    extractor_config = {
        'mediapipe': {
            'static_image_mode': False, 
            'model_complexity': 0, 
            'min_detection_confidence': 0.5
        }
    }
    extractor = PoseExtractor(extractor_config)
    
    recognizer = NSLRecognizer(
        model_path=model_path,
        vocab_path=vocab_path
    )

    cap = cv2.VideoCapture(0)
    print("🚀 Real-time NSL Recognizer Active. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)  

        pose, lh, rh, _, _ = extractor.process_frame(frame)
        features = recognizer.preprocess_landmarks(pose, lh, rh)
        recognizer.frame_buffer.append(features)

        prediction = recognizer.predict()

        # Add a dark overlay bar at the top for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw the Nepali text
        frame = recognizer.draw_nepali(frame, f"Sign: {prediction}")

        cv2.imshow("Nepali Sign Language Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_practice_session(target_char, model_path, vocab_path, duration=60):
    extractor = PoseExtractor({'mediapipe': {'static_image_mode': False, 'model_complexity': 0, 'min_detection_confidence': 0.5}})
    recognizer = NSLRecognizer(model_path, vocab_path)
    feedback_engine = NSLFeedbackEngine("reference_library.npz")
    
    cap = cv2.VideoCapture(0)
    session_start = time.time()
    
    hold_start_time = None
    REQUIRED_HOLD = 3.0  
    
    best_capture = None
    max_confidence = 0.0

    print(f"🎯 Practice Mode: Show the sign for '{target_char}' and hold it for {REQUIRED_HOLD}s")

    while True:
        elapsed = time.time() - session_start
        if elapsed > duration: 
            print("⏰ Time's up!")
            break
        
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        pose, lh, rh, _, _ = extractor.process_frame(frame)
        features = recognizer.preprocess_landmarks(pose, lh, rh)
        recognizer.frame_buffer.append(features)

        pred_char = "..."
        conf_val = 0.0
        if len(recognizer.frame_buffer) == 30:
            input_tensor = torch.tensor(list(recognizer.frame_buffer), dtype=torch.float32).unsqueeze(0).to(recognizer.device)
            with torch.no_grad():
                output = recognizer.model(input_tensor)
                prob = torch.softmax(output, dim=1)
                conf, idx = torch.max(prob, dim=1)
                pred_char = recognizer.tokenizer.idx2char[idx.item()]
                conf_val = conf.item()

        is_correct = (pred_char == target_char and conf_val > 0.85)
        
        if is_correct:
            if hold_start_time is None:
                hold_start_time = time.time()
            
            time_held = time.time() - hold_start_time
            remaining_hold = max(0, REQUIRED_HOLD - time_held)
            
            if conf_val > max_confidence:
                max_confidence = conf_val
                best_capture = (pose, lh, rh, pred_char, conf_val)

            color = (0, 255, 0) 
            status_text = f"CORRECT! Hold for {remaining_hold:.1f}s"
            
            if time_held >= REQUIRED_HOLD:
                print(f"✅ Sign held successfully for {REQUIRED_HOLD}s")
                break
        else:
            hold_start_time = None
            status_text = f"Target: {target_char} | Looking for sign..."
            color = (255, 255, 255) 

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        frame = recognizer.draw_nepali(frame, f"अभ्यास (Target): {target_char}", (20, 10))
        cv2.putText(frame, status_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Session: {int(duration-elapsed)}s", (frame.shape[1]-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("NSL Practice Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    if best_capture:
        p, l, r, pred, c = best_capture
        report = feedback_engine.compute_feedback(p, l, r, target_char, pred, c)
        feedback_engine.save_report(report)
        
        print("\n" + "="*30)
        print(f"FINAL RESULT FOR '{target_char}'")
        print(f"Status: {report['status']}")
        print(f"Score: {report['overall_score']:.1f}/100")
        print("-" * 30)
        for msg in report['feedback']:
            print(f" • {msg}")
        print("="*30 + "\n")
    else:
        print("\n❌ No stable sign was detected. Try again!")

if __name__ == "__main__":
    run_realtime()