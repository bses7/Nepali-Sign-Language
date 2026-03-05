import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, config):
        self.config = config
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=2, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7,
            static_image_mode=False 
        )

    def normalize_hand(self, hand_landmarks):
        """
        Wrist-centric normalization for finger shapes.
        """
        if not hand_landmarks:
            return np.zeros((21, 3)), np.zeros(4)
        
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
        wrist = pts[0].copy() 
        pts = pts - wrist 

        scale = np.linalg.norm(pts[0] - pts[9])
        if scale > 1e-6:
            pts = pts / scale
        else:
            scale = 1.0
            
        meta = np.array([wrist[0], wrist[1], wrist[2], scale], dtype=np.float32)
        return pts, meta

    def process_frame(self, frame, is_cropped=False):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        hand_res = self.mp_hands.process(frame_rgb)
        
        pose_data = np.zeros((33, 4))
        shoulder_center = np.array([0.5, 0.5, 0.0])

        if not is_cropped:
            pose_res = self.mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks.landmark])
                
                l_shoulder = pose_data[11, :3]
                r_shoulder = pose_data[12, :3]
                shoulder_center = (l_shoulder + r_shoulder) / 2

        lh_data, rh_data = np.zeros((21, 3)), np.zeros((21, 3))
        lh_meta, rh_meta = np.zeros(4), np.zeros(4) 

        if hand_res.multi_hand_landmarks:
            for res, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = handedness.classification[0].label.lower()
                side = 'left' if label == 'right' else 'right'
                
                norm_pts, meta = self.normalize_hand(res)
                
                if not is_cropped:
                    meta[0] -= shoulder_center[0]
                    meta[1] -= shoulder_center[1] 
                    meta[2] -= shoulder_center[2] 

                if side == 'left':
                    lh_data, lh_meta = norm_pts, meta
                else:
                    rh_data, rh_meta = norm_pts, meta
        
        if not is_cropped:
            for i in range(33):
                pose_data[i, :3] -= shoulder_center

        return pose_data, lh_data, rh_data, lh_meta, rh_meta