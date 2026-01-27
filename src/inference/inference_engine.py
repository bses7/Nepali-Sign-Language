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
        if motion.shape[0] <= window_size:
            window_size = motion.shape[0] // 2 * 2 - 1 
            if window_size < 3: return motion 
        smoothed = np.zeros_like(motion)
        for i in range(motion.shape[1]):
            smoothed[:, i] = savgol_filter(motion[:, i], window_size, poly_order)
        return smoothed

    def load_default_seed(self):
        try:
            csv_path = Path("training_dataset/master_metadata.csv")
            df = pd.read_csv(csv_path)
            # Find a full-body sign (not cropped) to get a good shoulder reference
            sample_row = df[(df['type'] == 'sign') & (df['is_cropped'] == False)].iloc[0]
            
            npz_path = Path("training_dataset") / sample_row['relative_path']
            data = np.load(npz_path)
            
            pose = data['pose'][0, :, :3].flatten()
            lh = data['lh'][0].flatten() * 5.0 
            rh = data['rh'][0].flatten() * 5.0 
            combined = np.concatenate([pose, lh, rh])
            
            print(f"✅ Loaded high-quality seed from: {npz_path.name}")
            return torch.tensor(combined, dtype=torch.float32).to(self.device)
        except:
            print("⚠️ Defaulting to zero seed")
            return torch.zeros(225).to(self.device)

    def generate(self, text, frames_per_char=60, hold_frames=10):
        self.model.eval()
        
        # --- CRITICAL FIX: MATCH DATASET NORMALIZATION ---
        current_seed = self.seed_pose.clone().detach().unsqueeze(0).unsqueeze(0)
        
        l_shoulder = current_seed[0, 0, 33:36].cpu().numpy()
        r_shoulder = current_seed[0, 0, 36:39].cpu().numpy()
        s_dist = np.linalg.norm(l_shoulder - r_shoulder)
        
        # Match the 0.3 fallback from dataset.py
        norm_factor = s_dist if s_dist > 0.05 else 0.3
        current_seed = current_seed / norm_factor
        # ------------------------------------------------

        full_motion = []
        for char in text:
            print(f"   -> Signing: {char}")
            token_list = self.tokenizer.tokenize(char)
            # Repeat tokens to boost attention presence
            tokens = torch.tensor(token_list * 3).unsqueeze(0).to(self.device)
            
            char_motion = current_seed.clone()
            context_window = 30

            with torch.no_grad():
                for _ in range(frames_per_char):
                    input_context = char_motion[:, -context_window:, :]
                    output = self.model(tokens, input_context)
                    next_frame = output[:, -1:, :]
                    
                    # Stabilization Mix: Prevent high-frequency jitter
                    prev_frame = char_motion[:, -1:, :]
                    stable_frame = (next_frame * 0.85) + (prev_frame * 0.15)
                    
                    char_motion = torch.cat([char_motion, stable_frame], dim=1)

            # Convert to numpy and skip the seed frame
            generated_frames_np = char_motion.squeeze(0)[1:].cpu().numpy()
            full_motion.append(generated_frames_np)
            
            # Add a small 'pause' hold to make letters distinct
            if hold_frames > 0:
                pause = np.repeat(generated_frames_np[-1:], hold_frames, axis=0)
                full_motion.append(pause)

            # Update seed for next letter
            current_seed = char_motion[:, -1:, :]

        final_motion = np.concatenate(full_motion, axis=0)
        return self.smooth_motion(final_motion, window_size=17)

    def save_to_npz(self, keypoints, output_path):
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Saved to {output_path}")