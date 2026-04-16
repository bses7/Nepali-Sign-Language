import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

def create_skeleton_video(npz_path, output_path, width=1200, height=1200, fps=30):
    """
    Reads an NPZ file and generates an MP4 video of the Right Hand 
    using the auto-scaling logic.
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    rh = data['rh'] 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("Drawing auto-scaled skeleton frames...")

    for f in tqdm(range(len(rh))):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        frame_pts = rh[f]

        min_p = frame_pts.min(axis=0)
        max_p = frame_pts.max(axis=0)
        diff = max_p - min_p
        
        scale = (width * 0.4) / (max(diff[0], diff[1]) + 1e-6)
        center_offset = (max_p + min_p) / 2.0

        def project(point):
            x = int((point[0] - center_offset[0]) * scale + width // 2)
            y = int((point[1] - center_offset[1]) * scale + height // 2)
            return (x, y)

        for start, end in HAND_CONNECTIONS:
            p1, p2 = project(frame_pts[start]), project(frame_pts[end])
            cv2.line(img, p1, p2, (180, 180, 180), 3)

        for i in range(21):
            color = (0, 0, 255) if i == 0 else (0, 255, 0) if i in [4,8,12,16,20] else (255, 255, 0)
            cv2.circle(img, project(frame_pts[i]), 8, color, -1)

        cv2.putText(img, f"Frame: {f}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        out.write(img)

    out.release()
    print(f"Video successfully saved to: {output_path}")