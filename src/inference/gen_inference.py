import torch
import numpy as np
from pathlib import Path
from src.models.motion_transformer import NSLTransformer
from src.data_preprocessing.tokenizer import NSLTokenizer
import pandas as pd

from src.generation.euro_filter import OneEuroFilter

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

    def smooth_motion(self, motion):
        """
        Applies One-Euro Filtering to the generated motion sequence.
        motion: [Frames, 225]
        """
        if motion.shape[0] < 2:
            return motion
        f = OneEuroFilter(0, motion[0], min_cutoff=0.1, beta=0.05)
        
        smoothed = np.zeros_like(motion)
        smoothed[0] = motion[0]
        
        for i in range(1, motion.shape[0]):
            smoothed[i] = f(i, motion[i])
            
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
    
    def generate(self, text, frames_per_sign=60, frames_per_trans=30, final_hold_duration=45):
        self.model.eval()
        
        # --- 1. PRIMING (Context Warm-up) ---
        # Filling the context window with 30 frames of the seed pose to prevent start-of-video snapping.
        seed_frame = (self.seed_pose.clone() / 0.5).unsqueeze(0).unsqueeze(0)
        generated_frames = seed_frame.repeat(1, 30, 1) 
        
        for i, char in enumerate(text):
            char_idx = self.tokenizer.char2idx.get(char, self.tokenizer.char2idx[self.tokenizer.unk_token])
            
            # --- 2. PHASE 1: TRANSITION ---
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
                    
                    # Tethering to prevent "dot-snapping"
                    next_frame[:, :, 99:102] = next_frame[:, :, 15*3:15*3+3]
                    next_frame[:, :, 162:165] = next_frame[:, :, 16*3:16*3+3]
                    
                    generated_frames = torch.cat([generated_frames, next_frame], dim=1)

            # --- 3. PHASE 2: SIGN HOLD ---
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

                    # Standard Momentum Blending
                    recent_avg = generated_frames[:, -3:, :].mean(dim=1, keepdim=True)
                    stable_frame = (next_frame * 0.8) + (recent_avg * 0.2)
                    generated_frames = torch.cat([generated_frames, stable_frame], dim=1)

        # --- 4. PHASE 3: EXTENDED FINAL HOLD ---
        # This executes AFTER the main word is finished. 
        # It takes the very last character and forces the model to stay in that pose.
        last_char = text[-1]
        last_char_idx = self.tokenizer.char2idx.get(last_char, self.tokenizer.char2idx[self.tokenizer.unk_token])
        
        final_hold_tokens = torch.tensor(
            [self.tokenizer.char2idx[self.tokenizer.sos_token], 
             self.tokenizer.char2idx[self.tokenizer.sign_mode]] + 
            [last_char_idx] * 15 + # Even more tokens for absolute focus
            [self.tokenizer.char2idx[self.tokenizer.eos_token]]
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(final_hold_duration):
                output = self.model(final_hold_tokens, generated_frames[:, -30:, :])
                next_frame = output[:, -1:, :]
                
                next_frame[:, :, 99:102] = next_frame[:, :, 15*3:15*3+3]
                next_frame[:, :, 162:165] = next_frame[:, :, 16*3:16*3+3]

                # High stability blending (0.5/0.5) to make the final pose rock-solid
                last_frame = generated_frames[:, -1:, :]
                very_stable_frame = (next_frame * 0.5) + (last_frame * 0.5)
                
                generated_frames = torch.cat([generated_frames, very_stable_frame], dim=1)

        # --- 5. CLEANUP ---
        # Remove the 30 priming frames from the start
        final_motion = generated_frames.squeeze(0)[30:].cpu().numpy()
        final_motion = final_motion * 0.5
        
        return self.smooth_motion(final_motion)

    def save_to_npz(self, keypoints, output_path):
        pose = keypoints[:, :99].reshape(-1, 33, 3)
        lh = keypoints[:, 99:162].reshape(-1, 21, 3)
        rh = keypoints[:, 162:].reshape(-1, 21, 3)
        
        lh = lh / 5.0
        rh = rh / 5.0
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pose=pose, lh=lh, rh=rh)
        print(f"💾 Motion generated and saved to {output_path}")