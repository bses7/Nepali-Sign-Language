import numpy as np
import json

class NSLFeedbackEngine:
    def __init__(self, library_path="reference_library.npz"):
        self.library = np.load(library_path, allow_pickle=True)
        self.finger_map = {
            "Thumb": [0, 1, 2, 4], 
            "Index": [0, 5, 6, 8], 
            "Middle": [0, 9, 10, 12],
            "Ring": [0, 13, 14, 16],
            "Pinky": [0, 17, 18, 20]
        }
        self.TOLERANCE = 15.0     
        self.SLIGHT_LIMIT = 30.0  
        self.MAJOR_LIMIT = 55.0   

    def _get_angle(self, a, b, c):
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_intensity(self, diff_mag):
        if diff_mag <= self.TOLERANCE: return None
        if diff_mag <= self.SLIGHT_LIMIT: return "slightly"
        if diff_mag <= self.MAJOR_LIMIT: return "more"
        return "significantly"

    def compute_feedback(self, user_pose, user_lh, user_rh, target_char, recognized_char, confidence):
        report = {
            "target_char": target_char,
            "detected_char": recognized_char,
            "overall_score": 0, 
            "accuracy_percentage": 0,
            "status": "Incomplete",
            "feedback": [],
            "timestamp_iso": np.datetime64('now').astype(str)
        }

        if target_char not in self.library:
            report["feedback"].append("Reference not available.")
            return report

        ref_data = self.library[target_char].item()
        ref_hand = ref_data['rh']
        
        is_using_rh = np.sum(np.abs(user_rh)) > np.sum(np.abs(user_lh))
        user_hand = user_rh if is_using_rh else user_lh
        if not is_using_rh:
            user_hand = user_hand.copy()
            user_hand[:, 0] *= -1 

        total_error_deg = 0
        
        for finger, idxs in self.finger_map.items():
            u_ang = self._get_angle(user_hand[idxs[1]], user_hand[idxs[2]], user_hand[idxs[3]])
            r_ang = self._get_angle(ref_hand[idxs[1]], ref_hand[idxs[2]], ref_hand[idxs[3]])
            
            diff = u_ang - r_ang
            mag = abs(diff)
            intensity = self._get_intensity(mag)

            if intensity:
                action = "Bend" if diff > 0 else "Straighten"
                report["feedback"].append(f"{action} your {finger} finger {intensity}.")
                total_error_deg += (mag - self.TOLERANCE)

        u_wrist_vec = user_hand[9] - user_hand[0]
        r_wrist_vec = ref_hand[9] - ref_hand[0]
        dot = np.dot(u_wrist_vec, r_wrist_vec) / (np.linalg.norm(u_wrist_vec) * np.linalg.norm(r_wrist_vec) + 1e-6)
        wrist_error = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        
        if wrist_error > self.TOLERANCE:
            intensity = self._get_intensity(wrist_error)
            report["feedback"].append(f"Rotate your wrist {intensity}.")
            total_error_deg += (wrist_error - self.TOLERANCE)

        # --- SCORING LOGIC ---
        # 100 points max. Each degree of error (outside tolerance) is a 0.5 point deduction.
        # Recognition of wrong character results in a massive penalty.
        
        base_score = 100.0
        deduction = total_error_deg * 0.5
        
        final_score = max(0, base_score - deduction)
        
        if recognized_char != target_char:
            final_score = min(final_score, 40.0)
            report["status"] = "Incorrect Sign"
        elif final_score >= 90:
            report["status"] = "Excellent"
            report["feedback"] = ["Great job! The sign is correct. Move on to next."]
        elif final_score >= 70:
            report["status"] = "Good"
        else:
            report["status"] = "Needs Improvement"

        report["overall_score"] = round(float(final_score), 2)
        report["accuracy_percentage"] = round(float(confidence * 100), 2)

        return report

    def save_report(self, report, filename="feedback_report.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)