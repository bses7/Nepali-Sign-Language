import site
import subprocess
import os
import hashlib
import sys
from pathlib import Path
from threading import Lock
from sqlalchemy.orm import Session
from app.core.config import settings, PROJECT_ROOT
import numpy as np

from src.inference.skeleton_viz import create_skeleton_video 

blender_lock = Lock()

sys.path.append(str(PROJECT_ROOT))
from src.inference.gen_inference import NSLGenerator

generator = NSLGenerator(
    model_path=settings.GENERATOR_MODEL_PATH, 
    vocab_path=settings.VOCAB_PATH
)

import subprocess
import os
import hashlib
import sys
import site
from pathlib import Path
from threading import Lock
from sqlalchemy.orm import Session
from app.core.config import settings, PROJECT_ROOT

# The global lock to prevent multiple Blender instances from crashing the CPU
blender_lock = Lock()

# Add PROJECT_ROOT to sys.path so we can import 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.gen_inference import NSLGenerator

# Initialize Generator once
generator = NSLGenerator(
    model_path=settings.GENERATOR_MODEL_PATH, 
    vocab_path=settings.VOCAB_PATH
)

def generate_custom_animation(db: Session, user, text: str):
    # 1. Setup Directories
    generated_dir = settings.STATIC_DIR / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Hashing and Path Definitions
    # We define these at the top level so they are available everywhere in the function
    avatar_folder = user.stats.current_avatar.folder_name if user.stats.current_avatar else "avatar"
    cache_key = f"{text.strip()}_{avatar_folder}"
    file_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    output_filename = f"{file_hash}.glb"
    output_glb_path = generated_dir / output_filename
    temp_npz = generated_dir / f"{file_hash}.npz"
    public_url = f"/static/generated/{output_filename}"

    # 3. Cache Check
    if output_glb_path.exists():
        print(f"♻️ Serving cached animation for: {text}")
        return public_url

    # 4. Preparation for Generation
    # Find the specific avatar rig file
    avatar_file = settings.AVATARS_BASE_DIR / f"{avatar_folder}.glb"
    if not avatar_file.exists():
        avatar_file = settings.AVATARS_BASE_DIR / "avatar.glb"

    # 5. Run Generation with Lock
    with blender_lock:
        try:
            print(f"🛠 Generating new animation: {text} (Avatar: {avatar_folder})")
            
            # Step A: Generate the motion data (.npz)
            motion = generator.generate(text)
            generator.save_to_npz(motion, str(temp_npz))

            # Step B: Setup Environment for Blender
            env = os.environ.copy()
            env["NPZ_PATH"] = str(temp_npz.absolute())
            env["OUTPUT_GLB"] = str(output_glb_path.absolute())

            # Step C: Run Blender
            result = subprocess.run([
                settings.BLENDER_EXE,
                "--background",
                "--python", settings.BLENDER_SCRIPT,
                "--", str(avatar_file.absolute())
            ], env=env, capture_output=True, text=True)

            # Debug logs
            if result.stdout: print(result.stdout)
            if result.stderr: print(f"Blender Warnings/Errors: {result.stderr}")

            # Step D: Cleanup and Verify
            if temp_npz.exists(): 
                os.remove(temp_npz)

            if not output_glb_path.exists():
                print(f"❌ ERROR: Blender failed to create {output_glb_path}")
                return None

            return public_url

        except Exception as e:
            print(f"❌ Generation Pipeline Error: {e}")
            if temp_npz.exists(): os.remove(temp_npz)
            return None
        
def generate_skeleton_visualization(text: str):
    video_dir = settings.STATIC_DIR / "generated" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    file_hash = hashlib.md5(text.strip().encode()).hexdigest()
    output_video_path = video_dir / f"{file_hash}.mp4"
    public_url = f"/static/generated/videos/{file_hash}.mp4"

    if output_video_path.exists():
        print(f"Serving cached skeleton video for: {text}")
        return public_url

    print(f"Generating skeleton video for: {text}")
    temp_npz = video_dir / f"{file_hash}_temp.npz"
    
    try:
        motion = generator.generate(text)
        generator.save_to_npz(motion, str(temp_npz))

        create_skeleton_video(str(temp_npz), str(output_video_path))

        if temp_npz.exists():
            os.remove(temp_npz)
            
        cleanup_old_files(video_dir, max_files=50, extension="*.mp4")
        
        return public_url

    except Exception as e:
        print(f"❌ Skeleton Generation Error: {e}")
        if temp_npz.exists(): os.remove(temp_npz)
        return None

def cleanup_old_files(directory: Path, max_files: int, extension: str = "*.glb"):
    files = sorted(directory.glob(extension), key=os.path.getmtime)
    if len(files) > max_files:
        for i in range(len(files) - max_files):
            os.remove(files[i])