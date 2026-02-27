from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.lesson import UserProgress, Sign
from app.services import lesson_service
from app.schemas.lesson import LessonOut, LessonComplete
from app.schemas.lesson import LessonOut, LessonComplete, Avatar as AvatarSchema 
from app.models.lesson import Avatar as AvatarModel 
from app.models.gamification import UserStats as StatsModel

router = APIRouter()

@router.post("/complete")
def complete_lesson(
    data: LessonComplete,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == current_user.id,
        UserProgress.sign_id == data.sign_id
    ).first()
    
    if not progress:
        progress = UserProgress(user_id=current_user.id, sign_id=data.sign_id, is_completed=True)
        db.add(progress)
    else:
        progress.is_completed = True
        
    if current_user.stats:
        current_user.stats.xp += 10
    else:
        new_stats = StatsModel(user_id=current_user.id, xp=10, level=1)
        db.add(new_stats)
    
    db.commit()
    return {"message": "Lesson completed and XP awarded"}

@router.get("/avatars", response_model=List[AvatarSchema])
def get_available_avatars(db: Session = Depends(get_db)):
    """
    Returns the list of avatars and their folder names.
    """
    return db.query(AvatarModel).all()

@router.get("/signs", response_model=List[LessonOut])
def read_lessons(
    avatar_id: int, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """
    Returns the signs with dynamic model_urls based on the selected avatar.
    """
    return lesson_service.get_lessons_for_user(db, current_user.id, avatar_id)