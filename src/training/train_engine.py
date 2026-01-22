import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.data_preprocessing.tokenizer import NSLTokenizer
from src.data_preprocessing.dataset import NSLDataset, nsl_collate_fn
from src.models.motion_transformer import NSLTransformer

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Engine started on: {device}")

    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    metadata_path = Path(config['paths']['output_dir']) / "master_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Master metadata not found at {metadata_path}. Please merge your CSVs first.")

    dataset = NSLDataset(
        metadata_path=str(metadata_path),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=nsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=nsl_collate_fn)

    # --- 2. Initialize Model ---
    model = NSLTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    best_val_loss = float('inf')
    save_path = Path(config['training']['model_save_path'])
    save_path.parent.mkdir(exist_ok=True)

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            src = batch['token_ids'].to(device)
            tgt = batch['features'].to(device)
            
            # Teacher Forcing Shift
            tgt_input = tgt[:, :-1, :]
            tgt_expected = tgt[:, 1:, :]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output, tgt_expected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['token_ids'].to(device)
                tgt = batch['features'].to(device)
                tgt_input = tgt[:, :-1, :]
                tgt_expected = tgt[:, 1:, :]
                
                output = model(src, tgt_input)
                total_val_loss += criterion(output, tgt_expected).item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(tokenizer.vocab),
                'config': config['training']
            }, save_path)
            print(f"Saved checkpoint to {save_path}")