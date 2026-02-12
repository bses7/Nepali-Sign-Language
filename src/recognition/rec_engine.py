import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from src.data_preprocessing.rec_dataset import NSLRecDataset, nsl_rec_collate_fn
from src.models.sign_classifier import NSLClassifier
from src.data_preprocessing.tokenizer import NSLTokenizer

def train_recognition(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rec_exp_dir = Path("experiments/recognition")
    rec_exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = rec_exp_dir / "best_recognizer.pth"
    log_path = rec_exp_dir / "rec_history.csv"
    
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    master_df = pd.read_csv(Path(config['paths']['output_dir']) / "master_metadata.csv")
    all_signers = master_df['signer'].unique()
    train_signers = [s for s in all_signers if s != 'S3']
    val_signers = ['S3']

    print(f"📈 Training on: {train_signers}")
    print(f"🧪 Testing on: {val_signers} (Signer-Independent Test)")

    train_ds = NSLRecDataset(
        metadata_path=str(Path(config['paths']['output_dir']) / "master_metadata.csv"),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer, signers_to_include=train_signers, augment=True
    )
    val_ds = NSLRecDataset(
        metadata_path=str(Path(config['paths']['output_dir']) / "master_metadata.csv"),
        sequences_root=config['paths']['sequences_dir'],
        tokenizer=tokenizer, signers_to_include=val_signers, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=nsl_rec_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=nsl_rec_collate_fn)

    model = NSLClassifier(num_classes=len(tokenizer.vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    
    best_acc = 0.0
    history = []

    for epoch in range(100):
        model.train()
        total_loss, correct = 0, 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for v_feats, v_labels in val_loader:
                v_feats, v_labels = v_feats.to(device), v_labels.to(device)
                v_out = model(v_feats)
                val_correct += (v_out.argmax(1) == v_labels).sum().item()

        train_acc = correct / len(train_ds)
        val_acc = val_correct / len(val_ds)
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'loss': total_loss/len(train_loader)
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': len(tokenizer.vocab),
                'signer_split': {'train': train_signers, 'test': val_signers}
            }, save_path)
            print(f"⭐ New Best Accuracy on S3: {val_acc:.2%}. Model updated.")

    print(f"✅ Recognition finished. Best model saved in {rec_exp_dir}")