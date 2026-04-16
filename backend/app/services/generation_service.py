import subprocess
import os
import hashlib
import sys
from pathlib import Path
from threading import Lock
from sqlalchemy.orm import Session
from app.core.config import settings, PROJECT_ROOT

from src.inference.skeleton_viz import create_skeleton_video 

blender_lock = Lock()

sys.path.append(str(PROJECT_ROOT))
from src.inference.gen_inference import NSLGenerator

generator = NSLGenerator(
    model_path=settings.GENERATOR_MODEL_PATH, 
    vocab_path=settings.VOCAB_PATH
)

def generate_custom_animation(db: Session, user, text: str):
    generated_dir = Path("static/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    avatar_folder = user.stats.current_avatar.folder_name if user.stats.current_avatar else "avatar"
    cache_key = f"{text.strip()}_{avatar_folder}"
    file_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    output_filename = f"{file_hash}.glb"
    output_glb_path = generated_dir / output_filename
    public_url = f"/static/generated/{output_filename}"

    if output_glb_path.exists():
        print(f"Serving cached animation for: {text}")
        return public_url

    with blender_lock:
        print(f"Generating new animation for: {text} using {avatar_folder}")
        
        temp_npz = generated_dir / f"{file_hash}.npz"
        avatar_file = settings.AVATARS_BASE_DIR / f"{avatar_folder}.glb"
        if not avatar_file.exists():
            avatar_file = settings.AVATARS_BASE_DIR / "avatar.glb"

        try:
            motion = generator.generate(text)
            generator.save_to_npz(motion, str(temp_npz))

            env = os.environ.copy()
            env["NPZ_PATH"] = str(temp_npz.absolute())
            env["OUTPUT_GLB"] = str(output_glb_path.absolute())

            subprocess.run([
                settings.BLENDER_EXE,
                "--background",
                "--python", settings.BLENDER_SCRIPT,
                "--", str(avatar_file.absolute())
            ], env=env, check=True, capture_output=True)

            if temp_npz.exists(): os.remove(temp_npz)
            
            cleanup_old_files(generated_dir, max_files=100)
            
            return public_url

        except subprocess.CalledProcessError as e:
            print(f"❌ Blender Error: {e.stderr.decode()}")
            if temp_npz.exists(): os.remove(temp_npz)
            return None
        
def generate_skeleton_visualization(text: str):
    video_dir = Path("static/generated/videos")
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