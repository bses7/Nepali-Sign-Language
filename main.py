import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.databuilder.single_builder import SingleBuilder
from src.databuilder.multi_builder import MultiBuilder
from src.data_preprocessing.tokenizer import NSLTokenizer
from src.generation.gen_engine import train_model

def main():
    parser = argparse.ArgumentParser(description="NSL Fingerspelling Pipeline")
    
    parser.add_argument("--stage", type=str, required=True, 
                        # Added 'rec_inference' to choices
                        choices=["build", "prep", "train", "generate", "recognize_train", "rec_inference", "practice", "ref_gen"], 
                        help="prep: build vocab | build: process videos | recognize_train: train classifier")
    
    parser.add_argument("--data", type=str, required=False, choices=["vowel", "consonant"])
    parser.add_argument("--type", type=str, required=False, choices=["single", "multi"])
    parser.add_argument("--config", type=str, default="config/config.yaml")

    args = parser.parse_args()

    # Pre-validation for build stage
    if args.stage in ["build"]:
        if not args.data or not args.type:
            parser.error(f"--stage {args.stage} requires both --data and --type")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Helper: Ensure master metadata exists before any training
    def ensure_master_metadata():
        output_dir = Path(config['paths']['output_dir'])
        master_csv = output_dir / "master_metadata.csv"
        if not master_csv.exists():
            print("📦 Master metadata not found. Consolidating all CSVs...")
            csv_files = list(output_dir.glob("*_metadata.csv"))
            if not csv_files:
                print("❌ Error: No metadata CSVs found.")
                return None
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            df.to_csv(master_csv, index=False, encoding='utf-8-sig')
            return master_csv
        return master_csv

    if args.stage == "prep":
        print("🛠️ Stage: PREP (Building Vocabulary)")
        tokenizer = NSLTokenizer(config)
        tokenizer.save_vocab("vocab.json")

    elif args.stage == "build":
        builder = SingleBuilder(config) if args.type == "single" else MultiBuilder(config)
        metadata = builder.build(args.data)
        if metadata:
            csv_path = Path(config['paths']['output_dir']) / f"{args.data}_{args.type}_metadata.csv"
            pd.DataFrame(metadata).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Build complete. Metadata saved to {csv_path}")
    
    elif args.stage == "train":
        print(f"🎬 Initializing Generation Training Phase...")
        if ensure_master_metadata():
            train_model(config)

    elif args.stage == "recognize_train":
        print(f"👁️ Initializing Recognition Training Phase (Testing on S3)...")
        if ensure_master_metadata():
            from src.recognition.rec_engine import train_recognition
            train_recognition(config)

    elif args.stage == "rec_inference":
        print(f"👁️ Initializing Real-time Recognition Inference...")
        from src.inference.rec_inference import run_realtime
        
        # Pull paths from your config.yaml
        model_path = config['rec_training']['model_save_path']
        vocab_path = "vocab.json"
        
        if not Path(model_path).exists():
            print(f"❌ Error: Model not found at {model_path}. Train the model first using --stage recognize_train")
        else:
            run_realtime(model_path=model_path, vocab_path=vocab_path)

    elif args.stage == "ref_gen":
        print("🔧 Stage: Generating Reference Library for Feedback...")
        from src.recognition.reference_generator import generate_reference_library
        generate_reference_library(config)

    elif args.stage == "practice":
        target_char = input("Enter the Nepali character you want to practice: ")
        print(f"🎯 Practice Mode: Show the sign for '{target_char}'")
        
        from src.inference.rec_inference import run_practice_session
        run_practice_session(
            target_char=target_char, 
            model_path=config['rec_training']['model_save_path'],
            vocab_path="vocab.json",
            duration=60
        )
            
    elif args.stage == "generate":
        print(f"✍️ Generating NSL for input...")
        from src.inference.gen_inference import NSLGenerator
        generator = NSLGenerator(model_path=config['training']['model_save_path'], vocab_path="vocab.json")
        user_input = input("Enter Nepali word to fingerspell: ")
        motion = generator.generate(user_input)
        out_file = Path("experiments/generated_output.npz")
        generator.save_to_npz(motion, out_file)

if __name__ == "__main__":
    main()