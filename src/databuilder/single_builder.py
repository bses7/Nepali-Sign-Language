import cv2
import pandas as pd
from pathlib import Path
from .base_builder import BaseBuilder

class SingleBuilder(BaseBuilder):
    def build(self, category):
        """category: 'vowel' or 'consonant'"""
        metadata = []
        conf = self.config['processing']
        
        # Determine paths based on category
        if category == "vowel":
            root_paths = [Path(self.config['paths']['raw_data']) / "NSL_Vowel"]
            label_map = conf['label_map']
        else:
            root_paths = [
                Path(self.config['paths']['raw_data']) / "NSL_Consonant_Part_1",
                Path(self.config['paths']['raw_data']) / "NSL_Consonant_Part_3"
            ]
            label_map = conf['consonant_label_map']

        for root in root_paths:
            if not root.exists(): continue
            for folder in root.iterdir():
                if not folder.is_dir() or not (folder.name.startswith("S1_") or folder.name.startswith("S2_")):
                    continue
                
                # Output directory setup
                out_rel_path = Path(category) / folder.name
                target_dir = Path(self.config['paths']['sequences_dir']) / out_rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                
                is_cropped = any(c in folder.name for c in conf['cropped_identifiers'])
                print(f"📦 Processing {category.upper()} Single: {folder.name}")

                for vid_path in folder.glob("*.MOV"):
                    parts = vid_path.stem.split('_')
                    if len(parts) < 2: continue
                    
                    raw_label = parts[1]
                    nepali_char = label_map.get(raw_label, "Unknown")
                    
                    cap = cv2.VideoCapture(str(vid_path))
                    info = [cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
                    
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        p, lh, rh, lm, rm = self.extractor.process_frame(frame, is_cropped)
                        frames.append({'pose':p, 'lh':lh, 'rh':rh, 'lh_meta':lm, 'rh_meta':rm})
                    cap.release()

                    # Save using base class method
                    save_path = target_dir / f"{raw_label}.npz"
                    self.save_npz(save_path, frames, info)
                    
                    metadata.append({
                        'relative_path': str(Path("sequences") / out_rel_path / f"{raw_label}.npz"),
                        'char': nepali_char,
                        'roman_label': raw_label,
                        'frames': len(frames),
                        'is_cropped': is_cropped,
                        'signer': self.get_signer_id(folder.name),
                        'type': 'sign'
                    })
        return metadata