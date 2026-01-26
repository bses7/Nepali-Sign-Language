import torch
import numpy as np
from pathlib import Path
from src.models.motion_transformer import NSLTransformer
from src.data_preprocessing.tokenizer import NSLTokenizer
import pandas as pd
from scipy.signal import savgol_filter


class NSLGenerator:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = NSLTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = NSLTransformer(vocab_size=checkpoint['vocab_size']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.seed_pose = self.load_default_seed()

    def smooth_motion(self, motion, window_size=15, poly_order=2):
        """
        smooths the [Frames, 225] array
        """
        if motion.shape[0] <= window_size:
            window_size = motion.shape[0] // 2 * 2 - 1 
            if window_size < 3: return motion 

        smoothed = np.zeros_like(motion)
        for i in range(motion.shape[1]):
            smoothed[:, i] = savgol_filter(motion[:, i], window_size, poly_order)
        return smoothed

    def load_default_seed(self):
        """Loads the first frame of a real sequence to use as a starting point."""
        try:
            csv_path = Path("training_dataset/master_metadata.csv")
            if not csv_path.exists():
                print(f"❌ Error: CSV not found at {csv_path.absolute()}")
                return torch.zeros(225).to(self.device)

            df = pd.read_csv(csv_path)
            sample_row = df[df['type'] == 'sign'].iloc[0]
            
            npz_path = Path("training_dataset") / sample_row['relative_path']
            
            if not npz_path.exists():
                print(f"⚠️ NPZ not found at {npz_path}, trying direct path...")
                npz_path = Path(sample_row['relative_path'])

            data = np.load(npz_path)
            
            pose = data['pose'][0, :, :3].flatten()
            lh = data['lh'][0].flatten() * 5.0  
            rh = data['rh'][0].flatten() * 5.0  
            combined = np.concatenate([pose, lh, rh])
            
            print(f"✅ Successfully loaded seed pose from: {npz_path.name}")
            return torch.tensor(combined, dtype=torch.float32).to(self.device)
            
        except Exception as e:
            print(f"❌ Error loading seed: {e}")
            return torch.zeros(225).to(self.device)

    def generate(self, text, frames_per_char=80):
        """
        Generates a word by processing each character sequentially.
        """
        self.model.eval()
        
        # Start with the initial global seed pose
        current_seed = self.seed_pose.clone().detach().unsqueeze(0).unsqueeze(0)
        full_motion = []

        print(f"🔤 Sequential Generation for: {text}")

        for char in text:
            print(f"   -> Signing: {char}")
            tokens = torch.tensor(self.tokenizer.tokenize(char), dtype=torch.long).unsqueeze(0).to(self.device)
            
            char_motion = current_seed.clone()
            
            context_window = 20

            with torch.no_grad():
                for _ in range(frames_per_char):
                    input_context = char_motion[:, -context_window:, :]
                    output = self.model(tokens, input_context)
                    
                    next_frame = output[:, -1:, :]
                    char_motion = torch.cat([char_motion, next_frame], dim=1)

            generated_frames = char_motion.squeeze(0)[1:]
            full_motion.append(generated_frames.cpu().numpy())

            current_seed = char_motion[:, -1:, :]

        final_motion = np.concatenate(full_motion, axis=0)
        
        final_motion = self.smooth_motion(final_motion, window_size=15)

        return final_motion
    
    def save_to_npz(self, keypoints, output_path):
        """Saves generated keypoints for Blender/Visualization."""
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Generated motion saved to {output_path}")