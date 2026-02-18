import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd 
import os 
import numpy as np

from src.data_preprocessing.tokenizer import NSLTokenizer
from src.data_preprocessing.gen_dataset import NSLDataset, nsl_collate_fn
from src.models.motion_transformer import NSLTransformer
from src.evaluation.logger import NSLLogger

# --- Loss Function Enhancements ---

def calculate_finger_direction_loss(pred, target):
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)

    segments = [
        # --- LEFT HAND (33-53) ---
        (37,38), (38,39), (39,40), 
        (41,42), (42,43), (43,44),
        (45,46), (46,47), (47,48), 
        (49,50), (50,51), (51,52), 
        (33,34), (34,35), (35,36), 
    
        # --- RIGHT HAND (54-74) ---
        (58,59), (59,60), (60,61), 
        (62,63), (63,64), (64,65),
        (66,67), (67,68), (68,69), 
        (70,71), (71,72), (72,73), 
        (54,55), (55,56), (56,57)  
    ]
    cos = nn.CosineSimilarity(dim=-1)
    total_dir_loss = 0
    for k, tip in segments:
        p_vec = p[:, :, tip, :] - p[:, :, k, :]
        t_vec = t[:, :, tip, :] - t[:, :, k, :]
        total_dir_loss += (1.0 - cos(p_vec, t_vec).mean())
    return total_dir_loss

def calculate_arm_hand_alignment_loss(pred, target, is_cropped_batch):
    """Only calculate alignment if full body is available."""
    full_body_mask = ~is_cropped_batch
    if not full_body_mask.any():
        return torch.tensor(0.0).to(pred.device)
    
    p = pred[full_body_mask].view(-1, pred.shape[1], 75, 3)
    t = target[full_body_mask].view(-1, target.shape[1], 75, 3)
    
    p_lh_arm = p[:, :, 15, :] - p[:, :, 13, :]
    t_lh_arm = t[:, :, 15, :] - t[:, :, 13, :]
    p_rh_arm = p[:, :, 16, :] - p[:, :, 14, :]
    t_rh_arm = t[:, :, 16, :] - t[:, :, 14, :]
    
    cos = nn.CosineSimilarity(dim=-1)
    return (1.0 - cos(p_lh_arm, t_lh_arm).mean()) + (1.0 - cos(p_rh_arm, t_rh_arm).mean())

def calculate_bone_consistency_loss(pred, target):
    p = pred.view(pred.shape[0], pred.shape[1], 75, 3)
    t = target.view(target.shape[0], target.shape[1], 75, 3)
    
    lh_bones = [(33,34), (34,35), (37,38), (38,39), (41,42), (42,43), (45,46), (46,47), (49,50), (50,51)]
    rh_bones = [(54,55), (55,56), (58,59), (59,60), (62,63), (63,64), (66,67), (67,68), (70,71), (71,72)]
    
    loss = 0
    for i1, i2 in lh_bones + rh_bones:
        p_len = torch.norm(p[:, :, i1, :] - p[:, :, i2, :], dim=-1)
        t_len = torch.norm(t[:, :, i1, :] - t[:, :, i2, :], dim=-1)
        loss += nn.MSELoss()(p_len, t_len)
    return loss

def calculate_velocity_loss(pred, target):
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    return nn.MSELoss()(pred_vel, target_vel)

# --- Training Logic ---

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = NSLLogger() 
    save_path = Path(config['training']['model_save_path'])
    
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    dataset = NSLDataset(
        metadata_path=str(Path(config['paths']['output_dir']) / "master_metadata.csv"),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer, augment=True 
    )

    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset)-train_len])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=nsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=nsl_collate_fn)

    model = NSLTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    
    if save_path.exists():
        print("🔄 Resuming training...")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = config['training'].get('early_stopping_patience', 100)

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_losses = {"pos": 0, "vel": 0, "bone": 0, "align": 0, "dir": 0}
        
        for batch in train_loader:
            src, tgt = batch['token_ids'].to(device), batch['features'].to(device)
            is_cropped_batch = batch['is_cropped'].to(device)
            
            # Shift for teacher forcing
            tgt_input, tgt_expected = tgt[:, :-1, :], tgt[:, 1:, :]

            # Scheduled Noise: Help model handle its own errors
            if epoch > 5:
                noise_lvl = min(0.005, 0.0001 * epoch)
                tgt_input = tgt_input + torch.randn_like(tgt_input) * noise_lvl
            
            optimizer.zero_grad()
            output = model(src, tgt_input)

            # --- DYNAMIC WEIGHTING ---
            batch_weights = torch.ones_like(output).to(device)
            
            for i, t_type in enumerate(batch['types']):
                if t_type == 'sign':
                    batch_weights[i, :, :99] = 2.0   # Pose importance
                    batch_weights[i, :, 99:] = 20.0  # Hand shape importance (High!)
                else: # Transition
                    batch_weights[i, :, :99] = 10.0  # Movement importance
                    batch_weights[i, :, 99:] = 5.0   # Hand shape less critical during move
            
            # MASK OUT POSE LOSS FOR CROPPED DATA
            batch_weights[is_cropped_batch, :, :99] = 0.0

            tips = [111, 112, 113, 123, 124, 125, 135, 136, 137, 147, 148, 149, 159, 160, 161]
            for t_idx in tips:
                batch_weights[:, :, t_idx] *= 2.0 
            
            rh_tips = [t + 63 for t in tips]
            for t_idx in rh_tips:
                batch_weights[:, :, t_idx] *= 2.0
            
            # --- LOSS CALCULATION ---
            l_pos = (criterion(output, tgt_expected) * batch_weights).mean()
            l_vel = calculate_velocity_loss(output, tgt_expected)
            l_bone = calculate_bone_consistency_loss(output, tgt_expected)
            l_align = calculate_arm_hand_alignment_loss(output, tgt_expected, is_cropped_batch)
            l_dir = calculate_finger_direction_loss(output, tgt_expected)

            # Total weighted loss
            total_loss = (
                l_pos + 
                (5.0 * l_vel) + 
                (10.0 * l_bone) + 
                (5.0 * l_align) + 
                (15.0 * l_dir) 
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["pos"] += l_pos.item()
            epoch_losses["vel"] += l_vel.item()
            epoch_losses["bone"] += l_bone.item()
            epoch_losses["align"] += l_align.item()
            epoch_losses["dir"] += l_dir.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                v_src, v_tgt = batch['token_ids'].to(device), batch['features'].to(device)
                v_out = model(v_src, v_tgt[:, :-1, :])
                val_loss += criterion(v_out, v_tgt[:, 1:, :]).mean().item()

        avg_val = val_loss / len(val_loader)
        
        num_batches = len(train_loader)
        logger.log_epoch(
            epoch + 1, 
            epoch_losses["pos"] / num_batches,
            epoch_losses["vel"] / num_batches,
            epoch_losses["bone"] / num_batches,
            avg_val, 
            optimizer.param_groups[0]['lr']
        )
        
        scheduler.step(avg_val)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]")

        print(f"Epoch {epoch+1} | Val Loss: {avg_val:.5f} | Pos: {epoch_losses['pos']/num_batches:.4f} | Dir: {epoch_losses['dir']/num_batches:.4f}")

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