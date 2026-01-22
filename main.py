import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.databuilder.single_builder import SingleBuilder
from src.databuilder.multi_builder import MultiBuilder
from src.data_preprocessing.tokenizer import NSLTokenizer

def main():
    parser = argparse.ArgumentParser(description="NSL Fingerspelling Pipeline")
    
    parser.add_argument("--stage", type=str, required=True, 
                        choices=["build", "prep", "train"], 
                        help="prep: build vocab | build: process videos")
    
    parser.add_argument("--data", type=str, required=False, choices=["vowel", "consonant"])
    parser.add_argument("--type", type=str, required=False, choices=["single", "multi"])
    parser.add_argument("--config", type=str, default="config/config.yaml")

    args = parser.parse_args()

    if args.stage in ["build", "train"]:
        if not args.data or not args.type:
            parser.error(f"--stage {args.stage} requires both --data and --type")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.stage == "prep":
        print("🛠️ Stage: PREP (Building Vocabulary)")
        tokenizer = NSLTokenizer(config)
        tokenizer.save_vocab("vocab.json")
        print("✅ Vocabulary prep finished.")

    elif args.stage == "build":
        builder = SingleBuilder(config) if args.type == "single" else MultiBuilder(config)
        
        metadata = builder.build(args.data)

        if metadata:
            csv_path = Path(config['paths']['output_dir']) / f"{args.data}_{args.type}_metadata.csv"
            pd.DataFrame(metadata).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Build complete. Metadata saved to {csv_path}")

if __name__ == "__main__":
    main()