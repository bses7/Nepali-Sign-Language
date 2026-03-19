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
from app.services import user_service
from app.schemas.gamification import Badge as BadgeSchema
from app.services import gamification_service
from app.schemas.lesson import SignDetail


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

    user_service.update_streak(db, current_user.stats)    
    XP_PER_LESSON = 500
    current_user.stats.xp += XP_PER_LESSON
    
    new_level = (current_user.stats.xp // 1000) + 1
    if new_level > current_user.stats.level:
        current_user.stats.level = new_level
    
    new_badges = gamification_service.check_and_award_badges(db, current_user)

    sign = db.query(Sign).filter(Sign.id == data.sign_id).first()
    
    user_service.update_daily_challenge(db, current_user.stats, sign)

    db.commit()
    return {"message": "Progress saved", "new_xp": current_user.stats.xp, "level": current_user.stats.level, "new_badges": new_badges}

@router.get("/avatars", response_model=List[AvatarSchema])
def get_available_avatars(db: Session = Depends(get_db)):
    """
    Returns the list of avatars and their folder names.
    """
    return db.query(AvatarModel).all()

@router.get("/signs", response_model=List[LessonOut])
def read_lessons(
    avatar_id: int = None, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """
    Returns signs. If avatar_id is not provided, 
    uses the user's currently equipped avatar.
    """
    return lesson_service.get_lessons_for_user(db, current_user, avatar_id)

@router.get("/signs/{sign_id}", response_model=SignDetail)
def get_lesson_detail(
    sign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get specific animation details for a letter.
    Returns 403 if the lesson is still locked for the user.
    """
    return lesson_service.get_sign_by_id(db, current_user, sign_id)