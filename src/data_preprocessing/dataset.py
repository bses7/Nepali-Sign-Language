import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class NSLDataset(Dataset):
    def __init__(self, metadata_path, sequences_root, tokenizer, max_seq_len=200, augment=False):
        """
        Args:
            metadata_path: Path to the consolidated CSV
            sequences_root: Path to the folder containing .npz files
            tokenizer: The NSLTokenizer instance
            max_seq_len: Maximum number of frames to allow (padding target)
        """
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
        
        pose = data['pose'][:, :, :3].reshape(data['pose'].shape[0], -1) 

        lh = data['lh'].reshape(data['lh'].shape[0], -1) * 5.0 # Scale up by 5
        rh = data['rh'].reshape(data['rh'].shape[0], -1) * 5.0 # Scale up by 5
        
        features = np.concatenate([pose, lh, rh], axis=1)
        
        token_ids = self.tokenizer.tokenize(row['char'])
        
        features = torch.tensor(features, dtype=torch.float32)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        if hasattr(self, 'augment') and self.augment:
            noise = torch.randn_like(features) * 0.002
            features += noise
            
            features[:, 15*3 : 15*3+3] = features[0, 15*3 : 15*3+3] 
            features[:, 16*3 : 16*3+3] = features[0, 16*3 : 16*3+3]
        
        return {
            'features': features,
            'token_ids': token_ids,
            'char': row['char'],
            'frames_count': features.shape[0]
        }

def nsl_collate_fn(batch):
    """
    This function handles PADDING. 
    Since every video has a different number of frames, we pad them with zeros 
    to match the longest video in the current batch.

    Feeding videos in batches
    """

    batch.sort(key=lambda x: x['features'].shape[0], reverse=True)
    
    features = [item['features'] for item in batch]
    token_ids = [item['token_ids'] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    tokens_padded = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True)
    
    return {
        'features': features_padded,
        'token_ids': tokens_padded,
        'lengths': torch.tensor([f.shape[0] for f in features])
    }