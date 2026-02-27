import os
import subprocess
from pathlib import Path
from extract_batch import BatchExtractor

# ============================================
# CONFIGURATION
# ============================================
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

BASE_DIR = Path(__file__).parent.resolve()
DATA_ROOT = BASE_DIR.parent / "data"

TARGET_DIRECTORIES = [
    DATA_ROOT / "NSL_Consonant_Part_1" / "S1_NSL_Consonant_Bright",
    DATA_ROOT / "NSL_Vowel" / "S1_NSL_Vowel_Unprepared_Bright"
]

AVATAR_FILENAMES = ["avatar.glb", "avatar1.glb", "avatar2.glb", "avatar3.glb", "avatar4.glb"]

# Output Locations
JSON_DIR = BASE_DIR / "data" / "keypoints"
ANIM_DIR = BASE_DIR / "data" / "animations"
BLENDER_SCRIPT = BASE_DIR / "blender_script.py"

JSON_DIR.mkdir(parents=True, exist_ok=True)
ANIM_DIR.mkdir(parents=True, exist_ok=True)

def collect_target_videos():
    valid_extensions = (".mp4", ".MOV", ".mov", ".MP4")
    video_tasks = []
    
    for directory in TARGET_DIRECTORIES:
        if not directory.exists():
            print(f"CRITICAL: Directory not found: {directory}")
            continue
            
        print(f"Scanning folder: {directory.name}")
        for file in directory.iterdir():
            if file.suffix in valid_extensions:
                video_tasks.append((file, directory.name))
                
    return video_tasks

def run_pipeline():
    tasks = collect_target_videos()
    
    if not tasks:
        print("No videos found. Check your folder paths.")
        return

    print(f"Total unique videos to process: {len(tasks)}")

    # --- PHASE 1: KEYPOINT EXTRACTION ---
    print("\n=== PHASE 1: KEYPOINT EXTRACTION ===")
    extractor = BatchExtractor(USE_POSE=True, USE_HANDS=True)
    
    for video_path, folder_name in tasks:
        unique_name = f"{folder_name}_{video_path.stem}"
        output_json = JSON_DIR / f"{unique_name}.json"
        
        if output_json.exists():
            print(f"[-] Skipping Extraction: {output_json.name} already exists.")
            continue

        print(f"[+] Extracting: {video_path.name} from {folder_name}")
        extractor.extract(str(video_path), str(output_json))

    # --- PHASE 2: BLENDER ANIMATION ---
    print("\n=== PHASE 2: BLENDER BATCH ANIMATION (5 AVATARS) ===")
    
    json_files = list(JSON_DIR.glob("*.json"))
    
    for json_file in json_files:
        print(f"\n>> Processing Keypoints: {json_file.name}")
        
        for avatar_filename in AVATAR_FILENAMES:
            avatar_path = BASE_DIR / "data" / avatar_filename
            avatar_stem = Path(avatar_filename).stem # e.g. "avatar1"
            
            avatar_output_dir = ANIM_DIR / avatar_stem
            avatar_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_glb = avatar_output_dir / f"{json_file.stem}_animated.glb"
            
            if output_glb.exists():
                print(f"    [-] Skipping {avatar_stem}: Animation already exists.")
                continue
            
            if not avatar_path.exists():
                print(f"    [!] Error: {avatar_filename} not found in data folder.")
                continue

            print(f"    [+] Baking {avatar_stem}...")
            
            command = [
                BLENDER_PATH, "-b",
                "--factory-startup",
                "-P", str(BLENDER_SCRIPT),
                "--",
                str(json_file),
                str(avatar_path),
                str(output_glb)
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"        ✓ Created: {output_glb.name}")
            except subprocess.CalledProcessError as e:
                print(f"        × ERROR baking with {avatar_filename}:")
                print(e.stderr)

    print("\n=== ALL BATCH TASKS COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()