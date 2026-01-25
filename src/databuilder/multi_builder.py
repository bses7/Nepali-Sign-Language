import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .base_builder import BaseBuilder

class MultiBuilder(BaseBuilder):
    def build(self, category):
        metadata = []
        conf = self.config['processing']
        label_map = conf['label_map'] if category == "vowel" else conf['consonant_label_map']
        
        anno_root = Path("data/annotations")
        base_out_dir = Path(self.config['paths']['sequences_dir']) / f"NSL_{category.capitalize()}_Multi"

        # Folders to search for videos
        data_sources = [Path(self.config['paths']['raw_data']) / f"NSL_{category.capitalize()}"]
        if category == "consonant":
            data_sources = [
                Path(self.config['paths']['raw_data']) / "NSL_Consonant_Part_1",
                Path(self.config['paths']['raw_data']) / "NSL_Consonant_Part_3"
            ]

        anno_folders = [
            f for f in anno_root.iterdir()
            if f.is_dir() and category.capitalize() in f.name
        ]

        for anno_folder in tqdm(
            anno_folders,
            desc=f"{category.upper()} annotation folders",
            unit="folder"
        ):
            csv_files = list(anno_folder.glob("*.csv"))

            for csv_path in tqdm(
                csv_files,
                desc=f"📄 {anno_folder.name}",
                unit="csv",
                leave=False
            ):
                df = pd.read_csv(csv_path, keep_default_na=False)
                if df.empty:
                    continue
                
                video_name = df.iloc[0]['video_name']
                video_path = next(
                    (s / anno_folder.name / video_name for s in data_sources
                     if (s / anno_folder.name / video_name).exists()),
                    None
                )
                
                if not video_path:
                    continue

                video_out_dir = base_out_dir / anno_folder.name / video_path.stem
                video_out_dir.mkdir(parents=True, exist_ok=True)

                cap = cv2.VideoCapture(str(video_path))
                info = [
                    cap.get(cv2.CAP_PROP_FPS),
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                ]
                is_cropped = df.iloc[0]['is_cropped']

                # Extract all frames once
                all_frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    p, lh, rh, lm, rm = self.extractor.process_frame(frame, is_cropped)
                    all_frames.append({
                        'pose': p,
                        'lh': lh,
                        'rh': rh,
                        'lh_meta': lm,
                        'rh_meta': rm
                    })
                cap.release()

                df = df.sort_values('start_frame')
                signer_id = self.get_signer_id(anno_folder.name)

                for i, row in tqdm(
                    df.iterrows(),
                    total=len(df),
                    desc=f"✂️ Segments ({video_path.stem})",
                    unit="segment",
                    leave=False
                ):
                    s, e, lbl = int(row['start_frame']), int(row['end_frame']), row['label']
                    
                    # 1. Save Sign
                    sign_seg = all_frames[s:e+1]
                    sign_fn = f"{lbl}_{s}_{e}.npz"
                    self.save_npz(video_out_dir / sign_fn, sign_seg, info)
                    metadata.append(
                        self._create_meta(
                            video_out_dir, sign_fn,
                            label_map.get(lbl, "Unknown"),
                            lbl, len(sign_seg),
                            is_cropped, signer_id, 'sign'
                        )
                    )

                    # 2. Save Transition
                    if i + 1 < len(df):
                        nxt = df.iloc[i + 1]
                        ts, te = e + 1, int(nxt['start_frame']) - 1
                        if te > ts:
                            t_seg = all_frames[ts:te+1]
                            t_fn = f"trans_{lbl}_to_{nxt['label']}.npz"
                            self.save_npz(video_out_dir / t_fn, t_seg, info)
                            metadata.append(
                                self._create_meta(
                                    video_out_dir, t_fn,
                                    'transition',
                                    f"trans_{lbl}_to_{nxt['label']}",
                                    len(t_seg),
                                    is_cropped, signer_id, 'transition'
                                )
                            )

        return metadata

    def _create_meta(self, out_dir, fn, char, roman, frames, cropped, signer, mtype):
        return {
            'relative_path': str(
                out_dir.relative_to(
                    Path(self.config['paths']['sequences_dir']).parent
                ) / fn
            ),
            'char': char,
            'roman_label': roman,
            'frames': frames,
            'is_cropped': cropped,
            'signer': signer,
            'type': mtype
        }
