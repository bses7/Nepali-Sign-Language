import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import PoseExtractor

def build_vowel_multi(config):
    extractor = PoseExtractor(config)
    
    # Path setup
    vowel_root = Path(config['paths']['raw_data']) / "NSL_Vowel"
    anno_root = Path("data/annotations")
    base_out_dir = Path(config['paths']['sequences_dir']) / "NSL_Vowel_Multi"
    
    metadata = []

    # Iterate through all your annotated folders
    for anno_folder in anno_root.iterdir():
        if not anno_folder.is_dir(): continue
        
        # FIX: Only process folders containing "Vowel"
        if "Vowel" not in anno_folder.name:
            continue
            
        print(f"\n📂 Processing Vowel Multi: {anno_folder.name}")

        for csv_path in anno_folder.glob("*.csv"):
            df = pd.read_csv(csv_path)
            if df.empty: continue

            video_name = df.iloc[0]['video_name']
            # Search for video in NSL_Vowel/<Annotation_Folder_Name>/<Video_Name>
            video_path = vowel_root / anno_folder.name / video_name
            
            if not video_path.exists():
                print(f"⚠️ Video not found: {video_path}")
                continue

            print(f"   🎥 Extracting: {video_name}")
            video_out_dir = base_out_dir / anno_folder.name / video_path.stem
            video_out_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            is_cropped = df.iloc[0]['is_cropped']

            # Process all frames
            frames_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                p, lh, rh, lm, rm = extractor.process_frame(frame, is_cropped=is_cropped)
                frames_keypoints.append({'pose':p, 'lh':lh, 'rh':rh, 'lh_meta':lm, 'rh_meta':rm})
            cap.release()

            # Slice segments
            df = df.sort_values('start_frame')
            for i, row in df.iterrows():
                start, end = int(row['start_frame']), int(row['end_frame'])
                label = row['label']
                nepali_char = config['processing']['label_map'].get(label, "Unknown")
                signer_id = anno_folder.name.split('_')[0]

                # Save Sign
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

                # Save Transition
                if i + 1 < len(df):
                    nxt = df.iloc[i+1]
                    t_s, t_e = end + 1, int(nxt['start_frame']) - 1
                    if t_e > t_s:
                        t_seg = frames_keypoints[t_s:t_e+1]
                        t_fn = f"trans_{label}_to_{nxt['label']}.npz"
                        save_npz(video_out_dir / t_fn, t_seg, [fps, w, h])
                        metadata.append({
                            'relative_path': str(video_out_dir.relative_to(base_out_dir.parent.parent) / t_fn),
                            'char': 'transition',
                            'roman_label': f"trans_{label}_to_{nxt['label']}",
                            'frames': len(t_seg),
                            'is_cropped': is_cropped,
                            'signer': signer_id,
                            'type': 'transition'      # Identifies this is a movement
                        })
    return metadata

def save_npz(path, frames, info):
    np.savez_compressed(path,
        pose=np.array([f['pose'] for f in frames], dtype=np.float32),
        lh=np.array([f['lh'] for f in frames], dtype=np.float32),
        rh=np.array([f['rh'] for f in frames], dtype=np.float32),
        lh_meta=np.array([f['lh_meta'] for f in frames], dtype=np.float32),
        rh_meta=np.array([f['rh_meta'] for f in frames], dtype=np.float32),
        video_info=np.array(info, dtype=np.float32))