from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.schemas.generation import TextGenerationRequest
from app.services import generation_service

router = APIRouter()

@router.post("/generate-sign")
async def generate_sign(
    request: TextGenerationRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Takes Nepali text and returns a URL to a 3D .glb file 
    animated specifically for the user's equipped avatar.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    glb_url = generation_service.generate_custom_animation(db, current_user, request.text)
    
    if not glb_url:
        raise HTTPException(status_code=500, detail="Animation generation failed.")

    return {
        "text": request.text,
        "avatar_used": current_user.stats.current_avatar.name if current_user.stats.current_avatar else "Default",
        "model_url": glb_url
    }

@router.post("/generate-skeleton")
async def generate_skeleton(
    request: TextGenerationRequest,
    current_user = Depends(get_current_user)
):
    """
    Takes Nepali text and returns a URL to an MP4 video 
    showing the skeleton coordinate visualization.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    video_url = generation_service.generate_skeleton_visualization(request.text)
    
    if not video_url:
        raise HTTPException(status_code=500, detail="Video generation failed.")

    return {
        "text": request.text,
        "video_url": video_url
    }