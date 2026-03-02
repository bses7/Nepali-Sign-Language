# app/services/lesson_service.py
from sqlalchemy.orm import Session
from app.models.lesson import Sign, UserProgress, DifficultyLevel, Avatar as AvatarModel, SignCategory
from app.models.user import User
from fastapi import HTTPException

def get_lessons_for_user(db: Session, user: User, avatar_id: int = None):

    if avatar_id:
        avatar = db.query(AvatarModel).filter(AvatarModel.id == avatar_id).first()
    else:
        avatar = user.stats.current_avatar
    
    avatar_folder = avatar.folder_name if avatar else "avatar"

    all_signs = db.query(Sign).order_by(Sign.difficulty, Sign.order_index).all()
    
    completed_sign_ids = [r[0] for r in db.query(UserProgress.sign_id).filter(
        UserProgress.user_id == user.id, UserProgress.is_completed == True).all()]

    easy_signs = [s for s in all_signs if s.difficulty == DifficultyLevel.EASY]
    medium_signs = [s for s in all_signs if s.difficulty == DifficultyLevel.MEDIUM]
    all_easy_done = all(s.id in completed_sign_ids for s in easy_signs) if easy_signs else True
    all_medium_done = all(s.id in completed_sign_ids for s in medium_signs) if medium_signs else True

    lessons = []
    for sign in all_signs:
        is_locked = False
        if sign.difficulty == DifficultyLevel.MEDIUM and not all(s.id in completed_sign_ids for s in [s for s in all_signs if s.difficulty == DifficultyLevel.EASY]):
            is_locked = True
        elif sign.difficulty == DifficultyLevel.HARD and not all(s.id in completed_sign_ids for s in [s for s in all_signs if s.difficulty == DifficultyLevel.MEDIUM]):
            is_locked = True
            
        if sign.category == SignCategory.VOWEL:
            filename = f"S1_NSL_Vowel_Unprepared_Bright_S1_{sign.sign_code}_animated.glb"
        else:
            filename = f"S1_NSL_Consonant_Bright_S1_{sign.sign_code}_animated.glb"
        
        model_url = f"/static/avatars/{avatar_folder}/{filename}"
            
        lessons.append({
            "id": sign.id,
            "title": sign.title,
            "nepali_char": sign.nepali_char,
            "difficulty": sign.difficulty,
            "category": sign.category,
            "is_completed": sign.id in completed_sign_ids,
            "is_locked": is_locked,
            "model_url": model_url
        })
    
    return lessons

def get_sign_by_id(db: Session, user: User, sign_id: int):
    sign = db.query(Sign).filter(Sign.id == sign_id).first()
    if not sign:
        raise HTTPException(status_code=404, detail="Sign not found")

    avatar = user.stats.current_avatar
    avatar_folder = avatar.folder_name if avatar else "avatar"
    
    type_str = "Vowel_Unprepared" if sign.category == SignCategory.VOWEL else "Consonant"
    filename = f"S1_NSL_{type_str}_Bright_S1_{sign.sign_code}_animated.glb"
    model_url = f"/static/avatars/{avatar_folder}/{filename}"

    return {
        "id": sign.id,
        "title": sign.title,
        "nepali_char": sign.nepali_char,
        "category": sign.category,
        "difficulty": sign.difficulty,
        "model_url": model_url,
        "animation_name": f"{sign.sign_code}_anim", 
        "description": sign.description 
    }

