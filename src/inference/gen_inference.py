import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from src.models.motion_transformer import NSLTransformer
from src.data_preprocessing.tokenizer import NSLTokenizer

from src.data_preprocessing.text_processor import NepaliTextProcessor

class NSLGenerator:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = NSLTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        self.text_processor = NepaliTextProcessor()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        v_size = checkpoint.get('vocab_size', len(self.tokenizer.vocab))
        f_dim = checkpoint.get('feature_dim', 231) 
        
        self.model = NSLTransformer(vocab_size=v_size, feature_dim=f_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_dim = f_dim
        print(f"NSL Generator Initialized. Vocab: {v_size}, Features: {f_dim}")

    def _generate_segment(self, text, mode="sign", seed_frames=None, max_frames=None):
        """Internal helper to generate a block of motion."""
        tokens = torch.tensor(self.tokenizer.tokenize(text, mode=mode)).unsqueeze(0).to(self.device)
        
        if max_frames is None:
            max_frames = 40 if mode == "sign" else 15

        if seed_frames is None:
            generated = torch.zeros((1, 1, self.feature_dim)).to(self.device)
        else:
            generated = seed_frames 

        output_frames = []

        for _ in range(max_frames):
            with torch.no_grad():
                input_seq = generated[:, -60:, :]
                preds = self.model(tokens, input_seq)
                
                next_frame = preds[:, -1:, :]
                
                output_frames.append(next_frame)
                generated = torch.cat([generated, next_frame], dim=1)

        return torch.cat(output_frames, dim=1)

    def generate(self, user_input, smooth=True):
        clusters = self.text_processor.get_clusters(user_input)
        all_segments = []
        last_frame = None

        for cluster in clusters:
            is_syllable = len(cluster) > 1
            
            for j, char in enumerate(cluster):
                duration = 35 if is_syllable else 45
                sign_frames = self._generate_segment(char, mode="sign", seed_frames=last_frame, max_frames=duration)
                all_segments.append(sign_frames)
                last_frame = sign_frames[:, -1:, :]

                if j < len(cluster) - 1:
                    for alpha in np.linspace(1.0, 0.2, 5):
                        relaxing_frame = last_frame.clone()
                        relaxing_frame[:, :, 99:225] *= alpha
                        all_segments.append(relaxing_frame)
                        last_frame = relaxing_frame

                    next_char = cluster[j+1]
                    trans_frames = self._generate_segment(char + next_char, mode="trans", seed_frames=last_frame, max_frames=20)
                    all_segments.append(trans_frames)
                    last_frame = trans_frames[:, -1:, :]

            for _ in range(10):
                living_frame = last_frame.clone()
                tremor = torch.randn_like(living_frame[:, :, 99:]) * 0.001
                living_frame[:, :, 99:] += tremor
                all_segments.append(living_frame)
                last_frame = living_frame

            current_index = clusters.index(cluster)
            if current_index < len(clusters) - 1:
                start_char = cluster[-1]
                end_char = clusters[current_index + 1][0]
                
                trans_frames = self._generate_segment(start_char + end_char, mode="trans", seed_frames=last_frame, max_frames=18)
                all_segments.append(trans_frames)
                last_frame = trans_frames[:, -1:, :]

        full_motion = torch.cat(all_segments, dim=1).squeeze(0).cpu().numpy()
        
        if smooth:
            full_motion = gaussian_filter1d(full_motion, sigma=1.5, axis=0)

        return full_motion
    
    def save_to_npz(self, motion_data, output_path):
        """
        The method called by main.py to save the results.
        Splits the 231-dim vector back into Pose, Hands, and Meta.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        pose = motion_data[:, :99].reshape(-1, 33, 3)
        lh = motion_data[:, 99:162].reshape(-1, 21, 3)
        rh = motion_data[:, 162:225].reshape(-1, 21, 3)
        lh_meta = motion_data[:, 225:228]
        rh_meta = motion_data[:, 228:231]

        np.savez(
            output_path,
            pose=pose, 
            lh=lh, 
            rh=rh,
            lh_meta=lh_meta, 
            rh_meta=rh_meta
        )
        print(f"Generated NPZ saved to: {output_path}")