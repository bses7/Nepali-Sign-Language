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

def save_visual_snapshot(model, tokenizer, epoch, device, save_dir):
    """Generates a test sign during training to see progress."""
    model.eval()
    char_to_test = "A" # You can change this to any letter
    tokens = torch.tensor(tokenizer.tokenize(char_to_test)).unsqueeze(0).to(device)
    
    # We need a seed pose. For simplicity, we use zeros 
    # (or you could pass your real seed here)
    gen_motion = torch.zeros((1, 1, 225)).to(device)
    
    with torch.no_grad():
        for _ in range(100): # Generate 100 frames
            output = model(tokens, gen_motion[:, -60:, :])
            next_frame = output[:, -1:, :]
            gen_motion = torch.cat([gen_motion, next_frame], dim=1)
            
    # Save as NPZ
    motion_np = gen_motion.squeeze(0).cpu().numpy()
    # Unpack for the skeleton viewer
    pose = motion_np[:, :99].reshape(-1, 33, 3)
    lh = motion_np[:, 99:162].reshape(-1, 21, 3)
    rh = motion_np[:, 162:].reshape(-1, 21, 3)
    
    np.savez_compressed(save_dir / f"epoch_{epoch}_sample_{char_to_test}.npz", 
                        pose=pose, lh=lh, rh=rh)

def calculate_velocity_loss(pred, target):
    """
    Measures the difference in movement between consecutive frames.
    pred/target shape: [Batch, Frames, 225]
    """
    # Calculate difference between frame(t) and frame(t+1)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    return nn.MSELoss()(pred_vel, target_vel)

def calculate_bone_length_loss(pred, target):
    """
    Ensures that finger segments don't 'shrink' or 'stretch' during rotation.
    If the hand flips incorrectly, the bones often collapse in depth.
    """
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)
    
    # We define key 'bones' to check (Wrist to Index MCP, Wrist to Pinky MCP)
    # LH: 33->38, 33->50 | RH: 54->59, 54->71
    def get_lengths(pts, i1, i2):
        return torch.norm(pts[:, :, i1, :] - pts[:, :, i2, :], dim=-1)

    p_len = get_lengths(p, 33, 38) + get_lengths(p, 33, 50) + \
            get_lengths(p, 54, 59) + get_lengths(p, 54, 71)
            
    t_len = get_lengths(t, 33, 38) + get_lengths(t, 33, 50) + \
            get_lengths(t, 54, 59) + get_lengths(t, 54, 71)
    
    return nn.MSELoss()(p_len, t_len)

def calculate_orientation_loss(pred, target):
    """
    Uses Cosine Similarity to ensure the palm faces the same direction as target.
    This is much more effective at stopping 180-degree flips than raw MSE.
    """
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)

    def get_normal(pts, w, i, p_idx):
        v1 = pts[:, :, i, :] - pts[:, :, w, :]
        v2 = pts[:, :, p_idx, :] - pts[:, :, w, :]
        return torch.cross(v1, v2, dim=-1)

    p_lh_norm = get_normal(p, 33, 38, 50)
    t_lh_norm = get_normal(t, 33, 38, 50)
    p_rh_norm = get_normal(p, 54, 59, 71)
    t_rh_norm = get_normal(t, 54, 59, 71)

    # Cosine Similarity: 1 means same direction, -1 means opposite
    cos = nn.CosineSimilarity(dim=-1)
    # Loss = 1.0 - similarity (so 0 is perfect, 2 is opposite)
    lh_sim_loss = 1.0 - cos(p_lh_norm, t_lh_norm).mean()
    rh_sim_loss = 1.0 - cos(p_rh_norm, t_rh_norm).mean()
    
    return lh_sim_loss + rh_sim_loss

def calculate_acceleration_loss(pred, target):
    """Measures change in velocity to ensure smoothness."""
    if pred.shape[1] < 3: return torch.tensor(0.0).to(pred.device)
    p_vel = pred[:, 1:, :] - pred[:, :-1, :]
    t_vel = target[:, 1:, :] - target[:, :-1, :]
    p_acc = p_vel[:, 1:, :] - p_vel[:, :-1, :]
    t_acc = t_vel[:, 1:, :] - t_vel[:, :-1, :]
    return nn.MSELoss()(p_acc, t_acc)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = NSLLogger() 
    eval_dir = Path("experiments/eval_samples")
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(config['training']['model_save_path'])
    
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    dataset = NSLDataset(
        metadata_path=str(Path(config['paths']['output_dir']) / "master_metadata.csv"),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer, augment=True 
    )

    train_ds, val_ds = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=nsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=nsl_collate_fn)

    model = NSLTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    if save_path.exists():
        print("🔄 Resuming fine-tuning with Masked Pose Loss...")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = config['training'].get('early_stopping_patience', 100)

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        model.train()
        total_pos, total_vel, total_ori, total_bone, total_acc = 0, 0, 0, 0, 0
        
        for batch in train_loader:
            src, tgt = batch['token_ids'].to(device), batch['features'].to(device)
            is_cropped_batch = batch['is_cropped'].to(device)
            tgt_input, tgt_expected = tgt[:, :-1, :], tgt[:, 1:, :]

            if epoch > 5:
                tgt_input = tgt_input + torch.randn_like(tgt_input) * 0.004
            
            optimizer.zero_grad()
            output = model(src, tgt_input)

            # --- DYNAMIC BATCH-WISE WEIGHTING ---
            # Shape: [Batch, Sequence, Features]
            batch_weights = torch.ones_like(output).to(device)

            # Weighting: 10.0 for hands (Balanced)
            batch_weights[:, :, 99:] = 40.0
            batch_weights[is_cropped_batch, :, :99] = 0.0
            for i in range(2, 225, 3): batch_weights[:, :, i] *= 1.5

            pos_loss = (criterion(output, tgt_expected) * batch_weights).mean()
            vel_loss = calculate_velocity_loss(output, tgt_expected)
            ori_loss = calculate_orientation_loss(output, tgt_expected)
            acc_loss = calculate_acceleration_loss(output, tgt_expected)
            bone_loss = calculate_bone_length_loss(output, tgt_expected)

            loss = pos_loss + (2.0 * vel_loss) + (10.0 * ori_loss) + (2.0 * acc_loss) + (5.0 * bone_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_pos += pos_loss.item()
            total_vel += vel_loss.item()
            total_ori += ori_loss.item()
            total_bone += bone_loss.item()
            total_acc += acc_loss.item()
        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for batch in val_loader:
                v_src, v_tgt = batch['token_ids'].to(device), batch['features'].to(device)
                v_is_cropped = batch['is_cropped'].to(device)
                v_out = model(v_src, v_tgt[:, :-1, :])

                v_weights = torch.ones_like(v_out).to(device)
                v_weights[v_is_cropped, :, :99] = 0.0
                
                v_loss = (criterion(v_out, v_tgt[:, 1:, :]) * v_weights).mean()
                val_loss_accum += v_loss.item()
                

        avg_pos = total_pos / len(train_loader)
        avg_vel = total_vel / len(train_loader)
        avg_ori = total_ori / len(train_loader)
        avg_bone = total_bone / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_val = val_loss_accum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # LOGGING
        logger.log_epoch(epoch+1, avg_pos, avg_vel, avg_val, current_lr)
        scheduler.step(avg_val)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]")
        print(f"  Train -> Pos: {avg_pos:.6f} | Vel: {avg_vel:.6f} | Ori: {avg_ori:.6f} | Bone: {avg_bone:.6f}| Acc: {avg_acc:.6f}")
        print(f"  Val   -> Loss: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 50 == 0:
            print(f"📸 Saving visual snapshot...")
            save_visual_snapshot(model, tokenizer, epoch+1, device, eval_dir)

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