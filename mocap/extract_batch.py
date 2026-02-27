import cv2
import mediapipe as mp
import json
from pathlib import Path

class BatchExtractor:
    def __init__(self, USE_POSE=True, USE_HANDS=True):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=2) if USE_POSE else None
        self.hands = self.mp_hands.Hands(max_num_hands=2) if USE_HANDS else None

    def extract(self, video_path, output_json):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_keypoints = {
            'metadata': {'fps': fps, 'video': Path(video_path).name},
            'frames': []
        }
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = {'frame': frame_idx, 'pose': None, 'hands': {'left': None, 'right': None}}
            
            if self.pose:
                results = self.pose.process(frame_rgb)
                if results.pose_landmarks:
                    frame_data['pose'] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
            
            if self.hands:
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hl, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label.lower()
                        real_label = 'left' if label == 'right' else 'right' # Anatomical Swap
                        frame_data['hands'][real_label] = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hl.landmark]
            
            all_keypoints['frames'].append(frame_data)
            frame_idx += 1
            
        cap.release()
        with open(output_json, 'w') as f:
            json.dump(all_keypoints, f)