import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, config):
        self.config = config
        # Increase confidence to reduce jitter in limited data
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=2, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7,
            static_image_mode=False # Better for video sequences
        )

    def normalize_hand(self, hand_landmarks):
        """
        Wrist-centric normalization for finger shapes.
        """
        if not hand_landmarks:
            return np.zeros((21, 3)), np.zeros(4)
        
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
        wrist = pts[0].copy() 
        pts = pts - wrist # Center wrist at (0,0,0)

        # Use distance from wrist (0) to middle-finger-mcp (9) as unit scale
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
        
        # 1. Process Hand Landmarks (Primary for fingerspelling)
        hand_res = self.mp_hands.process(frame_rgb)
        
        # 2. Process Pose Landmarks (For arm/shoulder movement)
        pose_data = np.zeros((33, 4))
        shoulder_center = np.array([0.5, 0.5, 0.0]) # Default center

        if not is_cropped:
            pose_res = self.mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks.landmark])
                
                # IMPORTANT: Use the midpoint of shoulders (11, 12) as the "Origin"
                # This keeps the torso stable while fingerspellng
                l_shoulder = pose_data[11, :3]
                r_shoulder = pose_data[12, :3]
                shoulder_center = (l_shoulder + r_shoulder) / 2

        lh_data, rh_data = np.zeros((21, 3)), np.zeros((21, 3))
        lh_meta, rh_meta = np.zeros(4), np.zeros(4) 

        if hand_res.multi_hand_landmarks:
            for res, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                # Correct for Mediapipe's handedness reversal
                label = handedness.classification[0].label.lower()
                side = 'left' if label == 'right' else 'right'
                
                norm_pts, meta = self.normalize_hand(res)
                
                # If we have pose data, we adjust the 'meta' wrist position 
                # to be relative to the shoulder center.
                if not is_cropped:
                    meta[0] -= shoulder_center[0] # wrist_x relative to shoulder
                    meta[1] -= shoulder_center[1] # wrist_y relative to shoulder
                    meta[2] -= shoulder_center[2] # wrist_z relative to shoulder

                if side == 'left':
                    lh_data, lh_meta = norm_pts, meta
                else:
                    rh_data, rh_meta = norm_pts, meta
        
        # Shift entire pose relative to shoulder center for consistency
        if not is_cropped:
            for i in range(33):
                pose_data[i, :3] -= shoulder_center

        return pose_data, lh_data, rh_data, lh_meta, rh_meta