# app/services/lesson_service.py
from sqlalchemy.orm import Session
from app.models.lesson import Sign, UserProgress, DifficultyLevel, Avatar as AvatarModel, SignCategory
from fastapi import HTTPException

def get_lessons_for_user(db: Session, user_id: int, avatar_id: int):
    avatar = db.query(AvatarModel).filter(AvatarModel.id == avatar_id).first()
    if not avatar:
        raise HTTPException(status_code=404, detail=f"Avatar with ID {avatar_id} not found")
    
    avatar_folder = avatar.folder_name

    all_signs = db.query(Sign).order_by(Sign.difficulty, Sign.order_index).all()
    
    completed_sign_ids = [r[0] for r in db.query(UserProgress.sign_id).filter(
        UserProgress.user_id == user_id, UserProgress.is_completed == True).all()]

    easy_signs = [s for s in all_signs if s.difficulty == DifficultyLevel.EASY]
    medium_signs = [s for s in all_signs if s.difficulty == DifficultyLevel.MEDIUM]
    all_easy_done = all(s.id in completed_sign_ids for s in easy_signs) if easy_signs else True
    all_medium_done = all(s.id in completed_sign_ids for s in medium_signs) if medium_signs else True

    lessons = []
    for sign in all_signs:
        is_locked = False
        if sign.difficulty == DifficultyLevel.MEDIUM and not all_easy_done:
            is_locked = True
        elif sign.difficulty == DifficultyLevel.HARD and not all_medium_done:
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