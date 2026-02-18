import numpy as np
import pandas as pd
from pathlib import Path

def generate_reference_library(config, output_path="reference_library.npz"):
    metadata_path = Path(config['paths']['output_dir']) / "master_metadata.csv"
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found.")
        return

    df = pd.read_csv(metadata_path)
    # We use S3 as the "Gold Standard" signer
    s3_data = df[df['signer'] == 'S3'].reset_index(drop=True)
    
    library = {}
    unique_chars = s3_data['char'].unique()
    
    print(f"📂 Building Reference Library from Signer S3...")

    for char in unique_chars:
        # Get all samples for this character by S3
        samples = s3_data[s3_data['char'] == char]
        if len(samples) == 0: continue
        
        # Take the first available sample
        sample = samples.iloc[0]
        npz_path = Path("training_dataset") / sample['relative_path']
        
        if not npz_path.exists():
            print(f"⚠️ Warning: Missing file {npz_path}")
            continue

        data = np.load(npz_path)
        
        # Take the middle frame of the sequence (usually the most stable)
        mid_idx = data['lh'].shape[0] // 2
        
        library[char] = {
            'lh': data['lh'][mid_idx],
            'rh': data['rh'][mid_idx],
            'pose': data['pose'][mid_idx]
        }
    
    # Save as a single dictionary file
    np.savez(output_path, **library)
    print(f"✅ Reference library saved to {output_path} ({len(library)} characters).")