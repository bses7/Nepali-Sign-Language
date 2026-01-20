import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import PoseExtractor
from .build_vowel_multi import save_npz # Reuse the same save function

def build_consonant_multi(config):
    extractor = PoseExtractor(config)
    anno_root = Path("data/annotations")
    base_out_dir = Path(config['paths']['sequences_dir']) / "NSL_Consonant_Multi"
    
    metadata = []

    # FIX: Paths to search (Part 1 and Part 3)
    data_sources = [
        Path(config['paths']['raw_data']) / "NSL_Consonant_Part_1",
        Path(config['paths']['raw_data']) / "NSL_Consonant_Part_3"
    ]

    for anno_folder in anno_root.iterdir():
        if not anno_folder.is_dir(): continue
        
        # Only process folders containing "Consonant"
        if "Consonant" not in anno_folder.name:
            continue
            
        print(f"\n📂 Processing Consonant Multi: {anno_folder.name}")

        for csv_path in anno_folder.glob("*.csv"):
            df = pd.read_csv(csv_path)
            if df.empty: continue

            video_name = df.iloc[0]['video_name']
            
            # Find the video in either Part 1 or Part 3
            video_path = None
            for source in data_sources:
                potential_path = source / anno_folder.name / video_name
                if potential_path.exists():
                    video_path = potential_path
                    break
            
            if not video_path:
                print(f"⚠️ Video not found for: {video_name}")
                continue

            print(f"   🎥 Extracting: {video_name}")
            video_out_dir = base_out_dir / anno_folder.name / video_path.stem
            video_out_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            is_cropped = df.iloc[0]['is_cropped']

            frames_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                p, lh, rh, lm, rm = extractor.process_frame(frame, is_cropped=is_cropped)
                frames_keypoints.append({'pose':p, 'lh':lh, 'rh':rh, 'lh_meta':lm, 'rh_meta':rm})
            cap.release()

            df = df.sort_values('start_frame')
            for i, row in df.iterrows():
                start, end = int(row['start_frame']), int(row['end_frame'])
                label = row['label']
                nepali_char = config['processing']['consonant_label_map'].get(label, "Unknown")
                signer_id = anno_folder.name.split('_')[0]

                sign_seg = frames_keypoints[start:end+1]
                sign_fn = f"{label}_{start}_{end}.npz"
                save_npz(video_out_dir / sign_fn, sign_seg, [fps, w, h])
                
                metadata.append({
                    'relative_path': str(video_out_dir.relative_to(base_out_dir.parent.parent) / sign_fn),
                    'char': nepali_char,
                    'roman_label': label,         # e.g. "KA"
                    'frames': len(sign_seg),
                    'is_cropped': is_cropped,
                    'signer': signer_id,
                    'type': 'sign'                # Identifies this is a stable sign
                })

                if i + 1 < len(df):
                    nxt = df.iloc[i+1]
                    t_s, t_e = end + 1, int(nxt['start_frame']) - 1
                    if t_e > t_s:
                        t_seg = frames_keypoints[t_s:t_e+1]
                        t_fn = f"trans_{label}_to_{nxt['label']}.npz"
                        save_npz(video_out_dir / t_fn, t_seg, [fps, w, h])
                        metadata.append({
                            'relative_path': str(video_out_dir.relative_to(base_out_dir.parent.parent) / sign_fn),
                            'char': nepali_char,
                            'roman_label': label,         # e.g. "KA"
                            'frames': len(sign_seg),
                            'is_cropped': is_cropped,
                            'signer': signer_id,
                            'type': 'sign'                # Identifies this is a stable sign
                        })
    return metadata