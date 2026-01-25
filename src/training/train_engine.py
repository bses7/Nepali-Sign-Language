import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd # Added for consolidation check

from src.data_preprocessing.tokenizer import NSLTokenizer
from src.data_preprocessing.dataset import NSLDataset, nsl_collate_fn
from src.models.motion_transformer import NSLTransformer

def calculate_velocity_loss(pred, target):
    """
    Measures the difference in movement between consecutive frames.
    pred/target shape: [Batch, Frames, 225]
    """
    # Calculate difference between frame(t) and frame(t+1)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    
    return nn.MSELoss()(pred_vel, target_vel)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    metadata_path = Path(config['paths']['output_dir']) / "master_metadata.csv"
    # Note: Added 'augment=True' here as per previous fix
    dataset = NSLDataset(
        metadata_path=str(metadata_path),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer,
        augment=True 
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=nsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=nsl_collate_fn)

    # 2. Initialize Model
    model = NSLTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # Scheduler: Reduces learning rate when loss stops dropping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # --- EARLY STOPPING STATE ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = config['training'].get('early_stopping_patience', 30)
    
    save_path = Path(config['training']['model_save_path'])

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        model.train()
        total_pos_loss = 0
        total_vel_loss = 0
        
        for batch in train_loader:
            src = batch['token_ids'].to(device)
            tgt = batch['features'].to(device)
            
            tgt_input = tgt[:, :-1, :]
            tgt_expected = tgt[:, 1:, :]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)

            # --- WEIGHTING LOGIC ---
            weights = torch.ones(225).to(device)
            weights[99:] = 20.0 # Focus on hands

            raw_loss = criterion(output, tgt_expected) # [Batch, Seq, 225]
            weighted_loss = (raw_loss * weights).mean()
            
            vel_loss = calculate_velocity_loss(output, tgt_expected)
            loss = weighted_loss + (5.0 * vel_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # FIX: UPDATE THE COUNTERS
            total_pos_loss += weighted_loss.item()
            total_vel_loss += vel_loss.item()

        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['token_ids'].to(device)
                tgt = batch['features'].to(device)
                output = model(src, tgt[:, :-1, :])
                
                # FIX: In validation, we must reduce the 'none' loss to a mean
                v_loss = criterion(output, tgt[:, 1:, :]).mean() 
                val_loss_accum += v_loss.item()

        avg_pos = total_pos_loss / len(train_loader)
        avg_vel = total_vel_loss / len(train_loader)
        avg_val = val_loss_accum / len(val_loader)

        # Update Scheduler
        scheduler.step(avg_val)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]")
        print(f"  Train -> Pos: {avg_pos:.6f} | Vel: {avg_vel:.6f}")
        print(f"  Val   -> Loss: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- EARLY STOPPING & SAVING LOGIC ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_counter = 0 # Reset counter because we improved
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(tokenizer.vocab),
            }, save_path)
            print(f"⭐ New Best Model Saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement for {early_stop_counter}/{patience} epochs.")

        if early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered. Training stopped at epoch {epoch+1}")
            break