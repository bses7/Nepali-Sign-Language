import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, max_seq_len=200, augment=False):
        """
        Updated Dataset for NSL.
        Features: 99 (Pose) + 63 (LH) + 63 (RH) + 3 (LH Meta) + 3 (RH Meta) = 231 Total.
        """
        full_df = pd.read_csv(metadata_path, keep_default_na=False)
        self.df = full_df[full_df['type'].isin(['sign', 'transition'])].reset_index(drop=True)
        
        self.sequences_root = Path(sequences_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        print(f"Dataset initialized. Features: 231 dimensions (99 Pose, 126 Hands, 6 Meta).")

    def __len__(self):
        return len(self.df)

    def rotate_point_cloud(self, batch_pts, max_angle=0.05):
        angle = np.random.uniform(-max_angle, max_angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        R = torch.tensor([
            [cos_a, -sin_a, 0], 
            [sin_a, cos_a, 0], 
            [0, 0, 1]
        ]).float() 
        
        frames = batch_pts.shape[0]
        pts = batch_pts.view(frames, 21, 3)
        pts = torch.matmul(pts, R.to(batch_pts.device))
        return pts.reshape(frames, -1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path(row['relative_path'])
        
        if not npz_path.exists():
            npz_path = self.sequences_root.parent / row['relative_path']

        data = np.load(npz_path)
        is_cropped = bool(row['is_cropped'])

        # 1. Load Pose, LH, RH (Total 225)
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) 
        lh = data['lh'].reshape(data['lh'].shape[0], -1)
        rh = data['rh'].reshape(data['rh'].shape[0], -1)

        lh_meta = data['lh_meta'][:, :3] 
        rh_meta = data['rh_meta'][:, :3]

        # Scaling for normalization
        lh = lh * 5.0
        rh = rh * 5.0
        pose = pose / 0.5 
        
        if not is_cropped:
            mid_x = (pose[:, 11*3] + pose[:, 12*3]) / 2
            mid_y = (pose[:, 11*3+1] + pose[:, 12*3+1]) / 2
            for c in range(0, 99, 3): 
                pose[:, c] -= mid_x
                pose[:, c+1] -= mid_y

        features = np.concatenate([pose, lh, rh, lh_meta, rh_meta], axis=1)
        features = torch.tensor(features, dtype=torch.float32)

        mode = "sign" if row['type'] == 'sign' else "trans"
        
        if mode == "trans":
            char_text = str(row['char']) 
            if len(char_text) < 2 and idx > 0:
                prev_row = self.df.iloc[idx-1]
                char_text = str(prev_row['char']) + char_text
        else:
            char_text = str(row['char'])
        
        token_ids = torch.tensor(
            self.tokenizer.tokenize(char_text, mode=mode), 
            dtype=torch.long
        )

        if self.augment:
            features += torch.randn_like(features) * 0.0005
            
            # Augment LH (99-162) and RH (162-225)
            if torch.rand(1) > 0.5:
                features[:, 99:162] = self.rotate_point_cloud(features[:, 99:162]) 
            if torch.rand(1) > 0.5:
                features[:, 162:225] = self.rotate_point_cloud(features[:, 162:225])  
            
            # local wrist remains 0,0,0
            features[:, 99:102] = 0.0 
            features[:, 162:165] = 0.0
        
        return {
            'features': features, 
            'token_ids': token_ids, 
            'type': row['type'], 
            'is_cropped': is_cropped
        }

def nsl_collate_fn(batch):
    batch.sort(key=lambda x: x['features'].shape[0], reverse=True)
    
    features = torch.nn.utils.rnn.pad_sequence(
        [item['features'] for item in batch], batch_first=True
    )
    token_ids = torch.nn.utils.rnn.pad_sequence(
        [item['token_ids'] for item in batch], batch_first=True
    )
    
    return {
        'features': features,
        'token_ids': token_ids,
        'types': [item['type'] for item in batch],
        'is_cropped': torch.tensor([item['is_cropped'] for item in batch], dtype=torch.bool)
    }