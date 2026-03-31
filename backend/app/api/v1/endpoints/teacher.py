import os
import shutil
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_verified_teacher
from app.models.user import User
from app.models.lesson import SignContribution
from app.services import user_service, notification_service

router = APIRouter()

UPLOAD_DIR = "static/uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-sign")
async def upload_sign_video(
    title: str = Form(...),
    description: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    teacher: User = Depends(get_verified_teacher)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    file_extension = video.filename.split(".")[-1]
    file_name = f"{uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    contribution = SignContribution(
        teacher_id=teacher.id,
        title=title,
        description=description,
        video_url=f"/{file_path}"
    )
    db.add(contribution)
    
    user_service.add_xp(db, teacher, 200) 
    teacher.stats.coins += 100           
    
    notification_service.create_notification(
        db, teacher.id, 
        "Contribution Received!", 
        f"Thank you! Your video for '{title}' has been submitted for review. +200 XP awarded!",
        "success"
    )

    db.commit()
    return {"message": "Video uploaded successfully", "video_path": f"/{file_path}"}