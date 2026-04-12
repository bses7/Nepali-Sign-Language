import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_preprocessing.tokenizer import NSLTokenizer
from src.data_preprocessing.gen_dataset import NSLDataset, nsl_collate_fn
from src.models.motion_transformer import NSLTransformer
from src.evaluation.metrics import calculate_blv_score, calculate_detailed_mje, calculate_jerk_score, calculate_pck, calculate_dtw_distance, calculate_velocity_error
from torch.utils.data import DataLoader

def run_test_evaluation(config_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = NSLTokenizer()
    tokenizer.load_vocab("vocab.json")
    
    dataset = NSLDataset(
        metadata_path="training_dataset/master_metadata.csv",
        sequences_root="training_dataset/sequences",
        tokenizer=tokenizer, augment=False
    )
    
    _, test_ds = torch.utils.data.random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=nsl_collate_fn)

    model = NSLTransformer(vocab_size=len(tokenizer.vocab), feature_dim=231).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = []
    
    print(f"Evaluating {len(test_loader)} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src = batch['token_ids'].to(device)
            tgt = batch['features'].to(device) 
            
            generated = tgt[:, :1, :] 
            for _ in range(tgt.shape[1] - 1):
                next_frame = model.generate_step(src, generated)
                generated = torch.cat([generated, next_frame], dim=1)

            gen_flat = generated.squeeze(0) 
            tgt_flat = tgt.squeeze(0)       

            eval_gen = gen_flat.clone()
            eval_tgt = tgt_flat.clone()
            eval_gen[:, :99] *= 0.5    
            eval_gen[:, 99:225] /= 5.0 
            eval_tgt[:, :99] *= 0.5
            eval_tgt[:, 99:225] /= 5.0

            eval_gen_3d = eval_gen.view(-1, 77, 3)
            eval_tgt_3d = eval_tgt.view(-1, 77, 3)
        
            mje_details = calculate_detailed_mje(eval_gen, eval_tgt)
            
            pck_fine = calculate_pck(eval_gen_3d, eval_tgt_3d, threshold=0.02) 
            pck_loose = calculate_pck(eval_gen_3d, eval_tgt_3d, threshold=0.05) 
            
            dtw = calculate_dtw_distance(eval_gen, eval_tgt)
            vel_err = calculate_velocity_error(eval_gen, eval_tgt)

            blv = calculate_blv_score(eval_gen_3d, threshold=0.1) 
            jerk = calculate_jerk_score(eval_gen_3d, fps=30) 
            
            res = {
                'sample': i,
                'type': batch['types'][0],
                'DTW': dtw,
                'VelError': vel_err,
                'PCK@0.02': pck_fine,
                'PCK@0.05': pck_loose,
                'BLV_%': blv,
                'Jerk_Score': jerk
            }
            res.update(mje_details) 
            results.append(res)

    df_results = pd.DataFrame(results)
    summary = df_results.groupby('type').mean(numeric_only=True)
    
    print("\n📊 --- TEST EVALUATION SUMMARY ---")
    print(summary[['mje_hands', 'mje_pose', 'PCK@0.02', 'PCK@0.05', 'BLV_%', 'Jerk_Score', 'VelError']])
    
    output_dir = Path("experiments/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "test_metrics_summary.csv")

