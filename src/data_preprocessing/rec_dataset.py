import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLRecDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, signers_to_include=None, num_frames=30, augment=False):
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
        
        print(f"✅ Recognition Dataset: Loaded {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def standardize_features(self, data, flip_lr=False):
        # Extract components
        pose = data['pose'][:, :, :3].copy() # (F, 33, 3)
        lh = data['lh'].copy() # (F, 21, 3)
        rh = data['rh'].copy() # (F, 21, 3)

        # HAND AGNOSTIC AUGMENTATION
        # If flip_lr is True, we swap Left and Right hands and flip X coordinates
        if flip_lr:
            # Flip X-axis (index 0 in the last dimension)
            pose[:, :, 0] *= -1
            lh[:, :, 0] *= -1
            rh[:, :, 0] *= -1
            # Swap hand data
            lh, rh = rh, lh

        # Flatten for the model
        pose_flat = pose.reshape(pose.shape[0], -1) / 0.5
        lh_flat = lh.reshape(lh.shape[0], -1) * 5.0
        rh_flat = rh.reshape(rh.shape[0], -1) * 5.0

        # Normalization (Mid-shoulder centering)
        mid_x = (pose_flat[:, 11*3] + pose_flat[:, 12*3]) / 2
        mid_y = (pose_flat[:, 11*3+1] + pose_flat[:, 12*3+1]) / 2
        
        for c in range(0, 99, 3):
            pose_flat[:, c] -= mid_x
            pose_flat[:, c+1] -= mid_y

        return np.concatenate([pose_flat, lh_flat, rh_flat], axis=1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path("training_dataset") / row['relative_path']
        data = np.load(npz_path)
        
        # Randomly flip during training to handle both hands
        do_flip = False
        if self.augment and np.random.random() > 0.5:
            do_flip = True
            
        features = self.standardize_features(data, flip_lr=do_flip)

        total_f = features.shape[0]
        if total_f >= self.num_frames:
            start = np.random.randint(0, total_f - self.num_frames + 1) if self.augment else (total_f - self.num_frames) // 2
            features = features[start : start + self.num_frames]
        else:
            pad_size = self.num_frames - total_f
            features = np.concatenate([features, np.tile(features[-1:], (pad_size, 1))], axis=0)

        label_id = self.tokenizer.char2idx.get(row['char'], self.tokenizer.char2idx[self.tokenizer.unk_token])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)

def nsl_rec_collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return features, labels