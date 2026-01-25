import torch
import numpy as np
from pathlib import Path
from src.models.motion_transformer import NSLTransformer
from src.data_preprocessing.tokenizer import NSLTokenizer
import pandas as pd

class NSLGenerator:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = NSLTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = NSLTransformer(vocab_size=checkpoint['vocab_size']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # --- NEW: Load a 'Seed Pose' from your dataset ---
        # This gives the model a realistic starting point
        self.seed_pose = self.load_default_seed()

    def load_default_seed(self):
        """Loads the first frame of a real sequence to use as a starting point."""
        try:
            # 1. Path relative to project root
            csv_path = Path("training_dataset/master_metadata.csv")
            if not csv_path.exists():
                print(f"❌ Error: CSV not found at {csv_path.absolute()}")
                return torch.zeros(225).to(self.device)

            df = pd.read_csv(csv_path)
            # Find the first 'sign' (not transition) to use as a neutral pose
            sample_row = df[df['type'] == 'sign'].iloc[0]
            
            # 2. Construct the full path to the NPZ file
            # Since relative_path in CSV is 'sequences/NSL_Vowel/...', 
            # and main.py is in root, we look inside 'training_dataset'
            npz_path = Path("training_dataset") / sample_row['relative_path']
            
            if not npz_path.exists():
                # Try absolute fallback
                print(f"⚠️ NPZ not found at {npz_path}, trying direct path...")
                npz_path = Path(sample_row['relative_path'])

            data = np.load(npz_path)
            
            # 3. Extract first frame
            pose = data['pose'][0, :, :3].flatten()
            lh = data['lh'][0].flatten()
            rh = data['rh'][0].flatten()
            combined = np.concatenate([pose, lh, rh])
            
            print(f"✅ Successfully loaded seed pose from: {npz_path.name}")
            return torch.tensor(combined, dtype=torch.float32).to(self.device)
            
        except Exception as e:
            print(f"❌ Error loading seed: {e}")
            return torch.zeros(225).to(self.device)

    def generate(self, text, max_frames=200):
        self.model.eval()
        tokens = torch.tensor(self.tokenizer.tokenize(text)).unsqueeze(0).to(self.device)
        
        # Start with seed
        generated_motion = self.seed_pose.unsqueeze(0).unsqueeze(0)
        
        # Use a smaller context window during generation
        # This prevents the 'history' from drowning out the 'text input'
        context_window = 40 

        with torch.no_grad():
            for _ in range(max_frames):
                # Only look at the last 40 frames
                input_motion = generated_motion[:, -context_window:, :]
                
                output = self.model(tokens, input_motion)
                
                # Get the last prediction
                next_frame = output[:, -1:, :]
                
                # Append
                generated_motion = torch.cat([generated_motion, next_frame], dim=1)

        return generated_motion.squeeze(0).cpu().numpy()
    
    def save_to_npz(self, keypoints, output_path):
        """Saves generated keypoints for Blender/Visualization."""
        # Total 225 = Pose(33*3=99) + LH(21*3=63) + RH(21*3=63)
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Generated motion saved to {output_path}")