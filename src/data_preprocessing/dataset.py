import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, max_seq_len=200, augment=False):
        """
        Specialized Dataset for Single Sign Perfection.
        """
        full_df = pd.read_csv(metadata_path, keep_default_na=False)
        self.df = full_df[full_df['type'] == 'sign'].reset_index(drop=True)
        
        self.sequences_root = Path(sequences_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        print(f"📊 Dataset initialized with {len(self.df)} SINGLE sign samples.")

    def __len__(self):
        return len(self.df)

    def rotate_point_cloud(self, batch_pts, max_angle=0.1):
        """Randomly rotates the hands slightly to improve orientation robustness."""
        angle = np.random.uniform(-max_angle, max_angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # ADDED .float() here to match the features precision
        R = torch.tensor([
            [cos_a, -sin_a, 0], 
            [sin_a, cos_a, 0], 
            [0, 0, 1]
        ]).float() 
        
        frames = batch_pts.shape[0]
        # Reshape to [Frames, 21 landmarks, 3 coordinates]
        pts = batch_pts.view(frames, 21, 3)
        
        # Apply rotation and flatten back
        pts = torch.matmul(pts, R.to(batch_pts.device))
        return pts.reshape(frames, -1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path(row['relative_path'])
        if not npz_path.exists():
            npz_path = self.sequences_root.parent / row['relative_path']

        data = np.load(npz_path)
        is_cropped = bool(row['is_cropped'])
        
        # 1. Extract Components
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) 
        lh = data['lh'].reshape(data['lh'].shape[0], -1) * 5.0 # BACK TO 5.0
        rh = data['rh'].reshape(data['rh'].shape[0], -1) * 5.0 # BACK TO 5.0

        pose = pose / 0.5
        
        if not is_cropped:
            mid_x = (pose[:, 11*3] + pose[:, 12*3]) / 2
            mid_y = (pose[:, 11*3+1] + pose[:, 12*3+1]) / 2
            
            for c in range(0, 99, 3): 
                pose[:, c] -= mid_x
                pose[:, c+1] -= mid_y
        
        features = np.concatenate([pose, lh, rh], axis=1)
        
        features = torch.tensor(features, dtype=torch.float32)
        token_ids = torch.tensor(self.tokenizer.tokenize(row['char']), dtype=torch.long)

        if self.augment:
            features += torch.randn_like(features) * 0.001
            
            features[:, 99:162] = self.rotate_point_cloud(features[:, 99:162]) 
            features[:, 162:] = self.rotate_point_cloud(features[:, 162:])  
            
            features[:, 99:102] = 0.0 
            features[:, 162:165] = 0.0
        
        return {
            'features': features, 
            'token_ids': token_ids, 
            'char': row['char'], 
            'is_cropped': is_cropped
        }

def nsl_collate_fn(batch):
    batch.sort(key=lambda x: x['features'].shape[0], reverse=True)
    features = [item['features'] for item in batch]
    token_ids = [item['token_ids'] for item in batch]
    is_cropped = torch.tensor([item['is_cropped'] for item in batch], dtype=torch.bool)

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True)
    
    return {
        'features': features_padded,
        'token_ids': tokens_padded,
        'lengths': torch.tensor([f.shape[0] for f in features]),
        'is_cropped': is_cropped
    }