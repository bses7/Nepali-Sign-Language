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
            # Use a full-body sign for a good reference
            sample_row = df[(df['type'] == 'single') & (df['is_cropped'] == False)].iloc[0]
            npz_path = Path("training_dataset") / sample_row['relative_path']
            data = np.load(npz_path)
            
            pose = data['pose'][0, :, :3].flatten()
            lh = data['lh'][0].flatten() * 5.0 
            rh = data['rh'][0].flatten() * 5.0

            mid_x = (pose[11*3] + pose[12*3]) / 2
            mid_y = (pose[11*3+1] + pose[12*3+1]) / 2
            for c in range(0, 99, 3):
                pose[c] -= mid_x
                pose[c+1] -= mid_y

            return torch.tensor(np.concatenate([pose, lh, rh]), dtype=torch.float32).to(self.device)
        except:
            return torch.zeros(225).to(self.device)

    def generate(self, text, frames_per_char=50, hold_frames=15):
        self.model.eval()
        # Seed is already centered and scaled correctly by load_default_seed()
        current_seed = self.seed_pose.clone().detach().unsqueeze(0).unsqueeze(0)
        
        full_motion = []
        for char in text:
            token_list = self.tokenizer.tokenize(char)
            tokens = torch.tensor(token_list * 10).unsqueeze(0).to(self.device)
            char_motion = current_seed.clone()
            
            with torch.no_grad():
                for f in range(frames_per_char):
                    output = self.model(tokens, char_motion[:, -40:, :])
                    next_frame = output[:, -1:, :]

                    mix_factor = 0.95 if f < 10 else 0.80
                    
                    prev_frame = char_motion[:, -1:, :]
                    stable_frame = (next_frame * mix_factor) + (prev_frame * (1 - mix_factor))
                    
                    char_motion = torch.cat([char_motion, stable_frame], dim=1)

            generated_frames_np = char_motion.squeeze(0)[1:].cpu().numpy()
            full_motion.append(generated_frames_np)
            
            # Character Hold (The 'Still' moment)
            if hold_frames > 0:
                full_motion.append(np.repeat(generated_frames_np[-1:], hold_frames, axis=0))

            current_seed = char_motion[:, -1:, :]

        final_motion = np.concatenate(full_motion, axis=0)
        return self.smooth_motion(final_motion, window_size=15)

    def save_to_npz(self, keypoints, output_path):
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Saved to {output_path}")