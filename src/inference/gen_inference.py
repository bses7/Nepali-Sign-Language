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

    def smooth_motion(self, motion, window_size=7, poly_order=2):
        """Reduces jitter while preserving the 'snap' of the sign."""
        if motion.shape[0] <= window_size:
            return motion
        smoothed = np.zeros_like(motion)
        for i in range(motion.shape[1]):
            smoothed[:, i] = savgol_filter(motion[:, i], window_size, poly_order)
        return smoothed

    def load_default_seed(self):
        """Returns a neutral shoulder-centered T-pose or first frame of data."""
        try:
            # We want a starting pose where arms are at sides or neutral
            # Looking for a non-cropped sign to get full body context
            csv_path = Path("training_dataset/master_metadata.csv")
            df = pd.read_csv(csv_path)
            sample_row = df[df['is_cropped'] == False].iloc[0]
            npz_path = Path("training_dataset") / sample_row['relative_path']
            data = np.load(npz_path)
            
            pose = data['pose'][0, :, :3].flatten()
            lh = data['lh'][0].flatten() * 5.0 
            rh = data['rh'][0].flatten() * 5.0
            
            # Center to shoulder midpoint (just like utils.py)
            mid_x = (pose[11*3] + pose[12*3]) / 2
            mid_y = (pose[11*3+1] + pose[12*3+1]) / 2
            for c in range(0, 99, 3):
                pose[c] -= mid_x
                pose[c+1] -= mid_y

            return torch.tensor(np.concatenate([pose, lh, rh]), dtype=torch.float32).to(self.device)
        except Exception as e:
            print(f"Seed loading failed ({e}), using zero seed.")
            return torch.zeros(225).to(self.device)

    def generate(self, text, frames_per_sign=60, frames_per_trans=25):
        self.model.eval()
        generated_frames = (self.seed_pose.clone() / 0.5).unsqueeze(0).unsqueeze(0)
        
        for char in text:
            char_idx = self.tokenizer.char2idx.get(char, self.tokenizer.char2idx[self.tokenizer.unk_token])
            
            trans_tokens = torch.tensor(
                [self.tokenizer.char2idx[self.tokenizer.sos_token], 
                 self.tokenizer.char2idx[self.tokenizer.trans_mode]] + 
                [char_idx] * 5 + 
                [self.tokenizer.char2idx[self.tokenizer.eos_token]]
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                for _ in range(frames_per_trans):
                    output = self.model(trans_tokens, generated_frames[:, -30:, :])
                    next_frame = output[:, -1:, :]
                    next_frame[:, :, 99:102] = next_frame[:, :, 15*3:15*3+3]
                    next_frame[:, :, 162:165] = next_frame[:, :, 16*3:16*3+3]
                    generated_frames = torch.cat([generated_frames, next_frame], dim=1)

            sign_tokens = torch.tensor(
                [self.tokenizer.char2idx[self.tokenizer.sos_token], 
                 self.tokenizer.char2idx[self.tokenizer.sign_mode]] + 
                [char_idx] * 10 + 
                [self.tokenizer.char2idx[self.tokenizer.eos_token]]
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                for _ in range(frames_per_sign):
                    output = self.model(sign_tokens, generated_frames[:, -30:, :])
                    next_frame = output[:, -1:, :]
                    next_frame[:, :, 99:102] = next_frame[:, :, 15*3:15*3+3]
                    next_frame[:, :, 162:165] = next_frame[:, :, 16*3:16*3+3]
                    
                    stable_frame = (next_frame * 0.9) + (generated_frames[:, -1:, :] * 0.1)
                    generated_frames = torch.cat([generated_frames, stable_frame], dim=1)

        final_motion = generated_frames.squeeze(0)[1:].cpu().numpy()
        final_motion = final_motion * 0.5
        
        return self.smooth_motion(final_motion)

    def save_to_npz(self, keypoints, output_path):
        # Reshape for Unity/Blender or Visualization tools
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        
        # IMPORTANT: Remember hands were scaled by 5.0 during training
        # We must divide by 5.0 to get back to standard Mediapipe coordinates
        lh = lh / 5.0
        rh = rh / 5.0
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Motion generated and saved to {output_path}")