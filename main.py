import argparse
import yaml
import pandas as pd
import sys
from pathlib import Path

from src.databuilder.build_vowel_single import build_vowel_single
from src.databuilder.build_consonant_single import build_consonant_single
from src.databuilder.build_vowel_multi import build_vowel_multi
from src.databuilder.build_consonant_multi import build_consonant_multi

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="NSL Fingerspelling Dataset Builder & Trainer")
    
    # CLI Arguments
    parser.add_argument("--stage", type=str, required=True, choices=["build", "train"], 
                        help="Stage of the pipeline: build or train")
    parser.add_argument("--data", type=str, required=True, choices=["vowel", "consonant"], 
                        help="Target data category")
    parser.add_argument("--type", type=str, required=True, choices=["single", "multi"], 
                        help="Video type: single (short clips) or multi (long sequences)")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to config file")

    args = parser.parse_args()

    config = load_config(args.config)
    
    output_base = Path(config['paths']['output_dir'])
    seq_dir = Path(config['paths']['sequences_dir'])
    seq_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "build":
        print(f"Starting Build Stage | Category: {args.data} | Type: {args.type}")
        
        metadata = []
        
        if args.data == "vowel":
            if args.type == "single":
                metadata = build_vowel_single(config)
            elif args.type == "multi":
                metadata = build_vowel_multi(config)
                
        elif args.data == "consonant":
            if args.type == "single":
                metadata = build_consonant_single(config)
            elif args.type == "multi":
                metadata = build_consonant_multi(config)

        if metadata:
            csv_name = f"{args.data}_{args.type}_metadata.csv"
            df = pd.DataFrame(metadata)
            df.to_csv(output_base / csv_name, index=False, encoding='utf-8-sig')
        else:
            print("\nBuild failed or no data was processed.")

    elif args.stage == "train":
        print("Training stage will be implemented after dataset is ready.")

if __name__ == "__main__":
    main()