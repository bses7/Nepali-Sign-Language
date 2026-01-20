import numpy as np
from src.utils import PoseExtractor

class BaseBuilder:
    def __init__(self, config):
        self.config = config
        self.extractor = PoseExtractor(config)

    def get_signer_id(self, folder_name):
        return folder_name.split('_')[0]

    def save_npz(self, path, frames, info):
        np.savez_compressed(
            path,
            pose=np.array([f['pose'] for f in frames], dtype=np.float32),
            lh=np.array([f['lh'] for f in frames], dtype=np.float32),
            rh=np.array([f['rh'] for f in frames], dtype=np.float32),
            lh_meta=np.array([f['lh_meta'] for f in frames], dtype=np.float32),
            rh_meta=np.array([f['rh_meta'] for f in frames], dtype=np.float32),
            video_info=np.array(info, dtype=np.float32)
        )