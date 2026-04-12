import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def calculate_detailed_mje(pred, target):
    """Calculates MJE for Pose, Hands, and Meta separately."""
    # Ensure shape is [Frames, 77, 3]
    p = pred.view(-1, 77, 3)
    t = target.view(-1, 77, 3)
    
    dist = torch.norm(p - t, dim=-1) # [Frames, 77]
    
    return {
        'mje_pose': dist[:, :33].mean().item(),
        'mje_hands': dist[:, 33:75].mean().item(),
        'mje_meta': dist[:, 75:].mean().item(),
        'mje_total': dist.mean().item()
    }

def calculate_pck(pred, target, threshold=0.05):
    """Percentage of Correct Keypoints."""
    dist = torch.norm(pred - target, dim=-1)
    return (dist < threshold).float().mean().item() * 100

def calculate_velocity_error(pred, target):
    """Fix: Handles both [Batch, Frames, Dim] and [Frames, Dim]"""
    if pred.dim() == 2:
        v_p = pred[1:, :] - pred[:-1, :]
        v_t = target[1:, :] - target[:-1, :]
    else:
        v_p = pred[:, 1:, :] - pred[:, :-1, :]
        v_t = target[:, 1:, :] - target[:, :-1, :]
    return torch.mean(torch.abs(v_p - v_t)).item()

def calculate_dtw_distance(pred, target):
    """Dynamic Time Warping."""
    # dtw expects 2D numpy arrays [Frames, Dims]
    p = pred.view(pred.shape[0], -1).detach().cpu().numpy()
    t = target.view(target.shape[0], -1).detach().cpu().numpy()
    distance, _ = fastdtw(p, t, dist=euclidean)
    return distance

def calculate_blv_score(pred, threshold=0.1):
    """Bone Length Violation: Pred must be [Frames, 77, 3]"""
    bone_pairs = [
        (11, 13), (13, 15), (12, 14), (14, 16), # Arms
        (33, 34), (37, 38), (41, 42), (45, 46), # LH
        (54, 55), (58, 59), (62, 63), (66, 67)  # RH
    ]
    total_violations = 0
    total_checks = 0
    
    for start, end in bone_pairs:
        lengths = torch.norm(pred[:, start, :] - pred[:, end, :], dim=-1)
        ref_length = torch.median(lengths)
        if ref_length < 1e-6: continue
        deviation = torch.abs(lengths - ref_length) / ref_length
        total_violations += (deviation > threshold).sum().item()
        total_checks += len(lengths)
    return (total_violations / total_checks) * 100 if total_checks > 0 else 0

def calculate_jerk_score(pred, fps=30):
    """Smoothness: Pred must be [Frames, 77, 3]"""
    v = (pred[1:] - pred[:-1]) * fps
    a = (v[1:] - v[:-1]) * fps
    j = (a[1:] - a[:-1]) * fps
    return torch.norm(j, dim=-1).mean().item()