import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, max_seq_len=200, augment=False):
        self.df = pd.read_csv(metadata_path, keep_default_na=False)
        self.sequences_root = Path(sequences_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path(row['relative_path'])
        if not npz_path.exists():
            npz_path = self.sequences_root.parent / row['relative_path']

        data = np.load(npz_path)
        is_cropped = bool(row['is_cropped'])
        
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) 
        lh = data['lh'].reshape(data['lh'].shape[0], -1) * 5.0 
        rh = data['rh'].reshape(data['rh'].shape[0], -1) * 5.0 
        features = np.concatenate([pose, lh, rh], axis=1)

        # --- IMPROVED SCALING LOGIC ---
        l_shoulder = features[0, 33:36] # Landmark 11
        r_shoulder = features[0, 36:39] # Landmark 12
        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)

        # fallback: 0.3 is roughly the average shoulder distance in normalized 0-1 space
        avg_shoulder_dist = 0.3 

        if not is_cropped and shoulder_dist > 0.05:
            features = features / shoulder_dist
        else:
            # If cropped, we divide by the average to keep hand size consistent
            features = features / avg_shoulder_dist
        # ------------------------------

        token_ids = self.tokenizer.tokenize(row['char'])
        features = torch.tensor(features, dtype=torch.float32)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        if hasattr(self, 'augment') and self.augment:
            features += torch.randn_like(features) * 0.002
            # Anchor Wrists
            for idx in [15, 16]:
                features[:, idx*3 : idx*3+3] = features[0, idx*3 : idx*3+3] 
        
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