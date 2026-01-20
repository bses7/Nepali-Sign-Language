import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.databuilder.single_builder import SingleBuilder
from src.databuilder.multi_builder import MultiBuilder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["build", "train"])
    parser.add_argument("--data", type=str, required=True, choices=["vowel", "consonant"])
    parser.add_argument("--type", type=str, required=True, choices=["single", "multi"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.stage == "build":
        builder = SingleBuilder(config) if args.type == "single" else MultiBuilder(config)
        
        metadata = builder.build(args.data)

        if metadata:
            csv_path = Path(config['paths']['output_dir']) / f"{args.data}_{args.type}_metadata.csv"
            pd.DataFrame(metadata).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Build complete. Metadata saved to {csv_path}")

if __name__ == "__main__":
    main()