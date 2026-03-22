import cv2
import numpy as np
import base64
import torch
import time
import sys
from app.core.config import settings, PROJECT_ROOT

sys.path.append(str(PROJECT_ROOT))

from src.inference.rec_inference import NSLRecognizer
from src.utils import PoseExtractor
from src.recognition.feedback_engine import NSLFeedbackEngine

class PracticeManager:
    def __init__(self):
        self.recognizer = NSLRecognizer(
            model_path=settings.RECOGNIZER_MODEL_PATH,
            vocab_path=settings.VOCAB_PATH
        )

        ml_cfg = settings.ml_config
        self.extractor = PoseExtractor(ml_cfg['mediapipe'])

        self.feedback_engine = NSLFeedbackEngine(str(PROJECT_ROOT / "reference_library.npz"))
        
        self.sessions = {}

    def process_frame(self, session_id, base64_frame, target_char): 
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "buffer": [], 
                "hold_start": None, 
                "best": None
            }
        
        session = self.sessions[session_id]

        try:
            if "," in base64_frame:
                base64_frame = base64_frame.split(",")[1]
            
            nparr = np.frombuffer(base64.b64decode(base64_frame), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1) # Mirror for webcam feel
        except Exception as e:
            return {"status": "error", "message": f"Decode failed: {str(e)}"}

        pose, lh, rh, _, _ = self.extractor.process_frame(frame)
        features = self.recognizer.preprocess_landmarks(pose, lh, rh)

        session["buffer"].append(features)
        if len(session["buffer"]) > 30:
            session["buffer"].pop(0)

        prediction = "..."
        confidence = 0.0
        if len(session["buffer"]) == 30:
            input_tensor = torch.tensor([session["buffer"]], dtype=torch.float32).to(self.recognizer.device)
            with torch.no_grad():
                output = self.recognizer.model(input_tensor)
                prob = torch.softmax(output, dim=1)
                conf, idx = torch.max(prob, dim=1)
                prediction = self.recognizer.tokenizer.idx2char[idx.item()]
                confidence = conf.item()

        is_correct = (prediction == target_char and confidence > 0.85)
        progress = 0.0
        status = "searching"

        if is_correct:
            if session["hold_start"] is None:
                session["hold_start"] = time.time()
            
            elapsed = time.time() - session["hold_start"]
            progress = min(1.0, elapsed / 3.0)
            
            if progress >= 1.0:
                status = "completed"
                session["best"] = (pose, lh, rh, prediction, confidence)
        else:
            session["hold_start"] = None

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "progress": round(progress, 2),
            "status": status
        }
    
    def generate_report(self, session_id, target_char):
        session = self.sessions.get(session_id)
        if not session or not session["best"]: return None
        p, l, r, pred, c = session["best"]
        return self.feedback_engine.compute_feedback(p, l, r, target_char, pred, c)

practice_manager = PracticeManager()