import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# --- CONSTANTS ---
VOWELS = ["A", "AA", "I", "II", "U", "UU", "RI", "E", "AI", "O", "AU", "AM", "AH"]
CONSONANTS = ["KA", "KHA", "GA", "GHA", "NGA",
    "CHA", "CHHA", "JA", "JHA", "YAN",
    "TA", "THA", "DA", "DHA", "NA",
    "TAA", "THAA", "DAA", "DHAA", "NAA",
    "PA", "PHA", "BA", "BHA", "MA",
    "YA", "RA", "LA", "WA",
    "T_SHA", "M_SHA", "D_SHA",
    "HA", "KSHA", "TRA", "GYA"
]

class NSLAnnotator:
    def __init__(self, mode):
        self.mode = mode
        self.sequence = VOWELS if mode == "vowel" else CONSONANTS
        self.base_data_path = Path("data")
        self.anno_base_path = Path("data/annotations")

    def find_videos(self):
        video_list = []
        if self.mode == "vowel":
            vowel_dir = self.base_data_path / "NSL_Vowel"
            prefixes = ["S3", "S4", "S5", "S6", "S14"]
            for folder in vowel_dir.iterdir():
                if folder.is_dir() and any(folder.name.startswith(p) for p in prefixes):
                    for vid in folder.glob("*.MOV"):
                        video_list.append(vid)
        else:
            paths_to_search = [
                self.base_data_path / "NSL_Consonant_Part_1",
                self.base_data_path / "NSL_Consonant_Part_3"
            ]
            target_folders = [
                "S3_NSL_Consonant_Prepared", "S3_NSL_Consonant_Real_World",
                "S4_NSL_Consonant_Prepared", "S5_NSL_Consonant_Prepared",
                "S6_NSL_Consonant_Prepared", "S6_NSL_Consonant_RealWorld",
                "S14_NSL_Consonant_RealWorld"
            ]
            for base_path in paths_to_search:
                if not base_path.exists(): continue
                for folder in base_path.iterdir():
                    if folder.is_dir() and folder.name in target_folders:
                        for vid in folder.glob("*.MOV"):
                            video_list.append(vid)
        return video_list

    def run_ui(self, video_path):
        # --- NEW: Manual Crop Selection ---
        print(f"\n🎥 Video: {video_path.name}")
        crop_input = input("Is this video CROPPED (only hand visible)? (y/n): ").lower()
        is_cropped = True if crop_input == 'y' else False
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        current_idx = 0
        annotations = []
        temp_start = None
        
        print(f"   Mode: {'[CROPPED]' if is_cropped else '[FULL-VIEW]'}")
        print("   Controls: [S]tart, [E]nd, [W]ipe Start, [D/F] Fwd, [A/R] Bwd, [Q]uit Video")

        while True:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                continue

            display = frame.copy()
            label = self.sequence[current_idx] if current_idx < len(self.sequence) else "DONE"
            
            # UI Overlay
            cv2.rectangle(display, (0, 0), (700, 160), (0, 0, 0), -1)
            color = (0, 255, 0) if temp_start is not None else (0, 255, 255)
            status = f"START: {temp_start} | Waiting for [E]" if temp_start is not None else "Press [S] at start of sign"
            
            crop_label = "CROPPED (Hand Only)" if is_cropped else "FULL (Pose+Hand)"
            cv2.putText(display, f"Label: {label} | {crop_label}", (20, 40), 1, 1.8, (255, 255, 255), 2)
            cv2.putText(display, status, (20, 90), 1, 1.5, color, 2)
            cv2.putText(display, f"Frame: {frame_idx}/{total_frames} | Video: {video_path.name}", (20, 140), 1, 1, (200, 200, 200), 1)

            cv2.imshow("NSL Professional Annotator", display)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'): 
                temp_start = frame_idx
            elif key == ord('e'):
                if temp_start is not None and frame_idx > temp_start:
                    annotations.append({
                        'label': label,
                        'start_frame': temp_start,
                        'end_frame': frame_idx,
                        'is_cropped': is_cropped, # Stores your manual choice
                        'video_name': video_path.name,
                        'folder_name': video_path.parent.name
                    })
                    temp_start = None
                    current_idx += 1
                    print(f"✅ Captured {label}")
                    if current_idx >= len(self.sequence): 
                        print("🎉 All letters in sequence finished!")
                        break
            elif key == ord('w'): 
                temp_start = None
                print("🗑️ Start point cleared")
            elif key == ord('d'): cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
            elif key == ord('f'): cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 15)
            elif key == ord('a'): cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            elif key == ord('r'): cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 15))
            elif key == ord('q'): 
                print("👋 Video closed by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return annotations

    def save(self, video_path, data):
        if not data: return
        save_dir = self.anno_base_path / video_path.parent.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{video_path.stem}.csv"
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"💾 Saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vowel", "consonant"], required=True)
    args = parser.parse_args()

    annotator = NSLAnnotator(args.mode)
    videos = annotator.find_videos()
    
    print(f"🔍 Found {len(videos)} videos for {args.mode}")
    
    for vid in videos:
        check_path = annotator.anno_base_path / vid.parent.name / f"{vid.stem}.csv"
        if check_path.exists():
            print(f"⏩ Skipping {vid.name} (Already annotated)")
            continue
            
        print("-" * 50)
        ans = input(f"Do you want to annotate {vid.name}? (y/n/q): ")
        if ans.lower() == 'q': break
        if ans.lower() != 'y': continue
        
        data = annotator.run_ui(vid)
        annotator.save(vid, data)

if __name__ == "__main__":
    main()