import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLRecDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, signers_to_include=None, num_frames=30, augment=False):
        """
        Args:
            signers_to_include: List of signer IDs (e.g., ['S1', 'S2', 'S14']). 
                                If None, includes all.
        """
        full_df = pd.read_csv(metadata_path, keep_default_na=False)
        
        temp_df = full_df[full_df['type'] == 'sign'].reset_index(drop=True)
        
        if signers_to_include is not None:
            self.df = temp_df[temp_df['signer'].isin(signers_to_include)].reset_index(drop=True)
        else:
            self.df = temp_df

        self.sequences_root = Path(sequences_root)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.augment = augment
        
        print(f"✅ Recognition Dataset: Loaded {len(self.df)} samples for signers {signers_to_include}")

    def __len__(self):
        return len(self.df)

    def standardize_features(self, data):
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) / 0.5
        lh = data['lh'].reshape(data['lh'].shape[0], -1) * 5.0
        rh = data['rh'].reshape(data['rh'].shape[0], -1) * 5.0

        mid_x = (pose[:, 11*3] + pose[:, 12*3]) / 2
        mid_y = (pose[:, 11*3+1] + pose[:, 12*3+1]) / 2
        for c in range(0, 99, 3):
            pose[:, c] -= mid_x
            pose[:, c+1] -= mid_y

        return np.concatenate([pose, lh, rh], axis=1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path("training_dataset") / row['relative_path']
        
        data = np.load(npz_path)
        features = self.standardize_features(data)

        total_f = features.shape[0]
        if total_f >= self.num_frames:
            start = np.random.randint(0, total_f - self.num_frames + 1) if self.augment else (total_f - self.num_frames) // 2
            features = features[start : start + self.num_frames]
        else:
            pad_size = self.num_frames - total_f
            features = np.concatenate([features, np.tile(features[-1:], (pad_size, 1))], axis=0)

        label_id = self.tokenizer.char2idx.get(row['char'], 3)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)

def nsl_rec_collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return features, labels