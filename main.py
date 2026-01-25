import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.databuilder.single_builder import SingleBuilder
from src.databuilder.multi_builder import MultiBuilder
from src.data_preprocessing.tokenizer import NSLTokenizer
from src.training.train_engine import train_model

def main():
    parser = argparse.ArgumentParser(description="NSL Fingerspelling Pipeline")
    
    parser.add_argument("--stage", type=str, required=True, 
                        choices=["build", "prep", "train", "generate"], 
                        help="prep: build vocab | build: process videos")
    
    parser.add_argument("--data", type=str, required=False, choices=["vowel", "consonant"])
    parser.add_argument("--type", type=str, required=False, choices=["single", "multi"])
    parser.add_argument("--config", type=str, default="config/config.yaml")

    args = parser.parse_args()

    if args.stage in ["build"]:
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
    
    elif args.stage == "train":
        print(f"🎬 Initializing Training Phase...")
        output_dir = Path(config['paths']['output_dir'])
        master_csv = output_dir / "master_metadata.csv"
        
        if not master_csv.exists():
            print("📦 Master metadata not found. Consolidating all CSVs...")
            csv_files = list(output_dir.glob("*_metadata.csv"))
            if not csv_files:
                print("❌ Error: No metadata CSVs found to consolidate.")
                return
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            df.to_csv(master_csv, index=False, encoding='utf-8-sig')
            print(f"✅ Consolidated {len(df)} samples.")

        train_model(config)

    elif args.stage == "generate":
        print(f"✍️ Generating NSL for input...")
        from src.inference.inference_engine import NSLGenerator
        
        # Initialize Generator
        generator = NSLGenerator(
            model_path=config['training']['model_save_path'],
            vocab_path="vocab.json"
        )
        
        # Get word from user
        user_input = input("Enter Nepali word to fingerspell: ")
        
        # Generate
        motion = generator.generate(user_input)
        
        # Save results
        out_file = Path("experiments/generated_output.npz")
        generator.save_to_npz(motion, out_file)


if __name__ == "__main__":
    main()