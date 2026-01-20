import cv2
import numpy as np
from pathlib import Path
from src.utils import PoseExtractor

def build_vowel_single(config):
    extractor = PoseExtractor(config)

    vowel_root = Path(config['paths']['raw_data']) / "NSL_Vowel"
    base_out_dir = Path(config['paths']['sequences_dir'])
    
    metadata = []
    
    if not vowel_root.exists():
        print(f" Error: Root folder {vowel_root} does not exist.")
        return []

    for folder in vowel_root.iterdir():
        if not folder.is_dir(): continue
        
        if not (folder.name.startswith("S1_") or folder.name.startswith("S2_")):
            continue
        
        target_subfolder = base_out_dir / "NSL_Vowel" / folder.name
        target_subfolder.mkdir(parents=True, exist_ok=True)
        
        is_cropped = any(cid in folder.name for cid in config['processing']['cropped_identifiers'])
        print(f"\n Processing Folder: {folder.name}")
        print(f"   Output: {target_subfolder}")
        print(f"   Mode: {'Cropped' if is_cropped else 'Full'}")

        for video_path in folder.glob("*.MOV"):
            filename_parts = video_path.stem.split('_')
            if len(filename_parts) < 2:
                continue
                
            raw_label = filename_parts[1] 
            nepali_char = config['processing']['label_map'].get(raw_label, "Unknown")

            print(f"      -> {video_path.name} -> {raw_label}.npz")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            seq_pose, seq_lh, seq_rh = [], [], []
            seq_lh_meta, seq_rh_meta = [], []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Extraction logic from utils.py
                pose, lh, rh, l_meta, r_meta = extractor.process_frame(frame, is_cropped=is_cropped)
                
                seq_pose.append(pose)
                seq_lh.append(lh)
                seq_rh.append(rh)
                seq_lh_meta.append(l_meta)
                seq_rh_meta.append(r_meta)
            
            cap.release()

            save_path = target_subfolder / f"{raw_label}.npz"
            
            np.savez_compressed(
                save_path,
                pose=np.array(seq_pose, dtype=np.float32),
                lh=np.array(seq_lh, dtype=np.float32),
                rh=np.array(seq_rh, dtype=np.float32),
                lh_meta=np.array(seq_lh_meta, dtype=np.float32),
                rh_meta=np.array(seq_rh_meta, dtype=np.float32), 
                video_info=np.array([fps, w, h], dtype=np.float32) 
            )
            
            metadata.append({
                'relative_path': str(Path("sequences") / "NSL_Vowel" / folder.name / f"{raw_label}.npz"),
                'char': nepali_char,
                'roman_label': raw_label,
                'frames': len(seq_pose),
                'is_cropped': is_cropped,
                'signer': folder.name.split('_')[0], 
                'type': 'sign'
            })
            
    return metadata

