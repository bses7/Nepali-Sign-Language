import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, max_seq_len=200, augment=False):
        """
        Specialized Dataset for NSL.
        Handles both static signs and transitions between them.
        """
        full_df = pd.read_csv(metadata_path, keep_default_na=False)
        # Filter for the types we have
        self.df = full_df[full_df['type'].isin(['sign', 'transition'])].reset_index(drop=True)
        
        self.sequences_root = Path(sequences_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        print(f"📊 Dataset initialized with {len(self.df)} samples (Signs & Transitions).")

    def __len__(self):
        return len(self.df)

    def rotate_point_cloud(self, batch_pts, max_angle=0.05):
        """Randomly rotates the hands slightly. Reduced angle for stability."""
        angle = np.random.uniform(-max_angle, max_angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotation around Z-axis (plane of the hand)
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
        
        # Path resolution
        if not npz_path.exists():
            npz_path = self.sequences_root.parent / row['relative_path']

        data = np.load(npz_path)
        is_cropped = bool(row['is_cropped'])
        
        # Load and reshape features
        # Pose: [F, 33, 3] -> [F, 99], Hands: [F, 21, 3] -> [F, 63]
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) 
        lh = data['lh'].reshape(data['lh'].shape[0], -1)
        rh = data['rh'].reshape(data['rh'].shape[0], -1)

        # 1. Normalization Scaling
        # We scale hands up (x5) to make finger movements more 'visible' to the loss function
        lh = lh * 5.0
        rh = rh * 5.0
        pose = pose / 0.5 # Match your 0.5 rule
        
        # 2. Torso Centering (If not already handled in PoseExtractor)
        if not is_cropped:
            # Anchor to shoulder midpoint
            mid_x = (pose[:, 11*3] + pose[:, 12*3]) / 2
            mid_y = (pose[:, 11*3+1] + pose[:, 12*3+1]) / 2
            for c in range(0, 99, 3): 
                pose[:, c] -= mid_x
                pose[:, c+1] -= mid_y

        features = np.concatenate([pose, lh, rh], axis=1)
        features = torch.tensor(features, dtype=torch.float32)

        # 3. Mode-Based Tokenization
        # If type is 'transition', the 'char' in CSV is the TARGET letter (e.g., 'Kha')
        mode = "sign" if row['type'] == 'sign' else "trans"
        char = str(row['char'])
        
        token_ids = torch.tensor(
            self.tokenizer.tokenize(char, mode=mode), 
            dtype=torch.long
        )

        # 4. Augmentation
        if self.augment:
            # Add very small noise to prevent overfitting to the fixed sequences
            features += torch.randn_like(features) * 0.0005
            
            # Rotate hands independently to simulate different wrist angles
            if torch.rand(1) > 0.5:
                features[:, 99:162] = self.rotate_point_cloud(features[:, 99:162]) 
            if torch.rand(1) > 0.5:
                features[:, 162:] = self.rotate_point_cloud(features[:, 162:])  
            
            # Wrist points (99-101 and 162-164) should ideally stay at 0,0,0 
            # after normalization, but we zero them to be safe
            features[:, 99:102] = 0.0 
            features[:, 162:165] = 0.0
        
        return {
            'features': features, 
            'token_ids': token_ids, 
            'type': row['type'], 
            'is_cropped': is_cropped
        }

def nsl_collate_fn(batch):
    # Sort by sequence length for efficient padding
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