import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, config):
        self.config = config
        self.mp_pose = mp.solutions.pose.Pose(model_complexity=2)
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    def normalize_hand(self, hand_landmarks):
        """
        Returns:
            pts: Normalized coordinates (wrist at 0,0,0)
            meta: [wrist_x, wrist_y, wrist_z, scale] (To reconstruct original position)
        """
        if not hand_landmarks:
            return np.zeros((21, 3)), np.zeros(4)
        
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
        wrist = pts[0].copy() # Original wrist position
        pts = pts - wrist

        scale = np.linalg.norm(pts[0] - pts[9])
        if scale > 0:
            pts = pts / scale
        else:
            scale = 1.0
            
        meta = np.array([wrist[0], wrist[1], wrist[2], scale], dtype=np.float32)
        return pts, meta

    def process_frame(self, frame, is_cropped=False):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.mp_hands.process(frame_rgb)
        
        pose_data = np.zeros((33, 4))
        if not is_cropped:
            pose_res = self.mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks.landmark])

        lh_data, rh_data = np.zeros((21, 3)), np.zeros((21, 3))
        lh_meta, rh_meta = np.zeros(4), np.zeros(4) # [wrist_x, wrist_y, wrist_z, scale]

        if hand_res.multi_hand_landmarks:
            for res, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = handedness.classification[0].label.lower()
                side = 'left' if label == 'right' else 'right'
                
                norm_pts, meta = self.normalize_hand(res)
                if side == 'left':
                    lh_data, lh_meta = norm_pts, meta
                else:
                    rh_data, rh_meta = norm_pts, meta
                
        return pose_data, lh_data, rh_data, lh_meta, rh_meta