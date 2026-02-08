import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd # Added for consolidation check
import os 
import numpy as np

from src.data_preprocessing.tokenizer import NSLTokenizer
from src.data_preprocessing.dataset import NSLDataset, nsl_collate_fn
from src.models.motion_transformer import NSLTransformer
from src.evaluation.logger import NSLLogger

def calculate_arm_hand_alignment_loss(pred, target):
    """Forces the forearm to rotate to match the target palm direction."""
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)
    # Forearm vectors: Elbow(13/14) -> Wrist(15/16)
    p_lh_arm = p[:, :, 15, :] - p[:, :, 13, :]
    t_lh_arm = t[:, :, 15, :] - t[:, :, 13, :]
    p_rh_arm = p[:, :, 16, :] - p[:, :, 14, :]
    t_rh_arm = t[:, :, 16, :] - t[:, :, 14, :]
    cos = nn.CosineSimilarity(dim=-1)
    return (1.0 - cos(p_lh_arm, t_lh_arm).mean()) + (1.0 - cos(p_rh_arm, t_rh_arm).mean())

def calculate_static_body_loss(pred, is_cropped_batch):
    """
    Penalizes the model if the torso (shoulders/hips) moves.
    In fingerspelling, the body should be like a statue.
    """
    # Only calculate for full-body videos
    full_body_mask = ~is_cropped_batch
    if not full_body_mask.any():
        return torch.tensor(0.0).to(pred.device)

    # Landmarks 11, 12 (Shoulders) and 23, 24 (Hips)
    torso_indices = [11, 12, 23, 24]
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    
    # Calculate movement (variance) over time for these points
    # We want the variance to be zero (no movement)
    torso_pts = p[full_body_mask][:, :, torso_indices, :]
    movement = torso_pts[:, 1:, :, :] - torso_pts[:, :-1, :, :]
    
    return torch.mean(movement**2)

def calculate_acceleration_loss(pred, target):
    """Measures change in velocity to ensure smoothness."""
    if pred.shape[1] < 3: return torch.tensor(0.0).to(pred.device)
    p_vel = pred[:, 1:, :] - pred[:, :-1, :]
    t_vel = target[:, 1:, :] - target[:, :-1, :]
    p_acc = p_vel[:, 1:, :] - p_vel[:, :-1, :]
    t_acc = t_vel[:, 1:, :] - t_vel[:, :-1, :]
    return nn.MSELoss()(p_acc, t_acc)

def calculate_velocity_loss(pred, target):
    """
    Measures the difference in movement between consecutive frames.
    pred/target shape: [Batch, Frames, 225]
    """
    # Calculate difference between frame(t) and frame(t+1)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    return nn.MSELoss()(pred_vel, target_vel)

def calculate_bone_consistency_loss(pred, target):
    """
    Checks all 15 major finger bones. 
    Ensures fingers don't collapse or stretch during complex signs.
    """
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)

    # Pairs of landmarks that form bones (Wrist to Tips)
    # LH: 33 is wrist. Bones: (33,34), (34,35)... (33,38), (38,39)...
    lh_bones = [(33,34), (34,35), (37,38), (38,39), (41,42), (42,43), (45,46), (46,47), (49,50), (50,51)]
    rh_bones = [(54,55), (55,56), (58,59), (59,60), (62,63), (63,64), (66,67), (67,68), (70,71), (71,72)]
    
    loss = 0
    for i1, i2 in lh_bones + rh_bones:
        p_len = torch.norm(p[:, :, i1, :] - p[:, :, i2, :], dim=-1)
        t_len = torch.norm(t[:, :, i1, :] - t[:, :, i2, :], dim=-1)
        loss += nn.MSELoss()(p_len, t_len)
    
    return loss

def calculate_orientation_loss(pred, target):
    """Strong Cosine Similarity to prevent 180-degree hand flips."""
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)

    def get_normal(pts, w, i, p_idx):
        v1 = pts[:, :, i, :] - pts[:, :, w, :]
        v2 = pts[:, :, p_idx, :] - pts[:, :, w, :]
        return torch.cross(v1, v2, dim=-1)

    p_norm = torch.cat([get_normal(p, 33, 38, 50), get_normal(p, 54, 59, 71)], dim=-1)
    t_norm = torch.cat([get_normal(t, 33, 38, 50), get_normal(t, 54, 59, 71)], dim=-1)

    cos = nn.CosineSimilarity(dim=-1)
    return (1.0 - cos(p_norm, t_norm)).mean()


def save_visual_snapshot(model, tokenizer, epoch, device, save_dir, config):
    """Uses a real seed to generate a high-quality progress check."""
    model.eval()
    char_to_test = "A" 
    tokens = torch.tensor(tokenizer.tokenize(char_to_test)).unsqueeze(0).to(device)
    
    # Load a real seed frame
    csv_path = Path(config['paths']['output_dir']) / "master_metadata.csv"
    df = pd.read_csv(csv_path)
    sample_path = Path("training_dataset") / df[df['type'] == 'sign'].iloc[0]['relative_path']
    data = np.load(sample_path)
    seed = torch.tensor(np.concatenate([data['pose'][0, :, :3].flatten(), 
                                      data['lh'][0].flatten(), 
                                      data['rh'][0].flatten()]), dtype=torch.float32).to(device)
    
    gen_motion = seed.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(60):
            output = model(tokens, gen_motion[:, -30:, :])
            gen_motion = torch.cat([gen_motion, output[:, -1:, :]], dim=1)
            
    motion_np = gen_motion.squeeze(0).cpu().numpy()
    np.savez_compressed(save_dir / f"epoch_{epoch}_sample_{char_to_test}.npz", 
                        pose=motion_np[:, :99].reshape(-1, 33, 3),
                        lh=motion_np[:, 99:162].reshape(-1, 21, 3),
                        rh=motion_np[:, 162:].reshape(-1, 21, 3))

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = NSLLogger() 
    eval_dir = Path("experiments/eval_samples")
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(config['training']['model_save_path'])
    
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    # Load ONLY Single sign dataset
    dataset = NSLDataset(
        metadata_path=str(Path(config['paths']['output_dir']) / "master_metadata.csv"),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer, augment=True 
    )

    train_ds, val_ds = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=nsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=nsl_collate_fn)

    model = NSLTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    
    # Enable Fine-Tuning
    if save_path.exists():
        print("🔄 Loading existing model for Specialist Fine-tuning...")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = config['training'].get('early_stopping_patience', 100)

    for epoch in range(config['training']['epochs']):
        model.train()
        # FIX: Initialize all counters
        t_pos, t_vel, t_ori, t_bone, t_align, t_static = 0, 0, 0, 0, 0, 0
        
        for batch in train_loader:
            src, tgt = batch['token_ids'].to(device), batch['features'].to(device)
            is_cropped_batch = batch['is_cropped'].to(device)
            tgt_input, tgt_expected = tgt[:, :-1, :], tgt[:, 1:, :]

            # Scheduled Sampling (Improves inference robustness)
            if epoch > 10:
                tgt_input = tgt_input + torch.randn_like(tgt_input) * 0.003
            
            optimizer.zero_grad()
            output = model(src, tgt_input)

            # --- DYNAMIC VECTORIZED WEIGHTING ---
            batch_weights = torch.ones_like(output).to(device)
            
            # Create boolean masks
            is_trans_mask = torch.tensor([t == 'transition' for t in batch['types']]).to(device)
            is_sign_mask = ~is_trans_mask
            
            # Apply weights based on type (Sign vs Transition)
            # For Signs: Focus on Fingers (High Weight)
            batch_weights[is_sign_mask, :, :99] = 5.0
            batch_weights[is_sign_mask, :, 99:] = 40.0
            
            # For Transitions: Focus on Arm Movement (Higher Pose Weight)
            batch_weights[is_trans_mask, :, :99] = 15.0
            batch_weights[is_trans_mask, :, 99:] = 10.0
            
            # Mask out Pose for cropped videos (0 weight)
            batch_weights[is_cropped_batch, :, :99] = 0.0
            
            # Apply Z-axis boost
            for i in range(2, 225, 3): batch_weights[:, :, i] *= 1.5

            pos_loss = (criterion(output, tgt_expected) * batch_weights).mean()
            vel_loss = calculate_velocity_loss(output, tgt_expected)
            ori_loss = calculate_orientation_loss(output, tgt_expected)
            bone_loss = calculate_bone_consistency_loss(output, tgt_expected)
            align_loss = calculate_arm_hand_alignment_loss(output, tgt_expected)
            static_loss = calculate_static_body_loss(output, is_cropped_batch)

            loss = pos_loss + (2.0 * vel_loss) + (20.0 * ori_loss) + \
                   (10.0 * align_loss) + (5.0 * bone_loss) + (10.0 * static_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_pos += pos_loss.item(); t_vel += vel_loss.item(); t_ori += ori_loss.item()
            t_bone += bone_loss.item(); t_align += align_loss.item(); t_static += static_loss.item()

        # Validation
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for batch in val_loader:
                v_src, v_tgt = batch['token_ids'].to(device), batch['features'].to(device)
                v_out = model(v_src, v_tgt[:, :-1, :])
                val_loss_accum += criterion(v_out, v_tgt[:, 1:, :]).mean().item()

        avg_val = val_loss_accum / len(val_loader)
        logger.log_epoch(epoch+1, t_pos/len(train_loader), t_vel/len(train_loader), avg_val, optimizer.param_groups[0]['lr'])
        scheduler.step(avg_val)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]")
        print(f"Pos: {t_pos/len(train_loader):.4f} | Ori: {t_ori/len(train_loader):.4f} | Bone: {t_bone/len(train_loader):.4f} | Align: {t_align/len(train_loader):.4f} | Val: {avg_val:.4f}")

        if (epoch + 1) % 50 == 0:
            save_visual_snapshot(model, tokenizer, epoch+1, device, eval_dir, config)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'vocab_size': len(tokenizer.vocab)}, save_path)
            print("⭐ Model Saved")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("🛑 Early Stopping!")
                break