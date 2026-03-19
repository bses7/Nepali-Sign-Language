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
    
    def is_difficulty_finished(category: SignCategory, difficulty: DifficultyLevel):
        signs_in_group = [s for s in all_signs if s.category == category and s.difficulty == difficulty]
        if not signs_in_group:
            return True
        return all(s.id in completed_sign_ids for s in signs_in_group)

    completion_status = {
        SignCategory.VOWEL: {
            "easy_done": is_difficulty_finished(SignCategory.VOWEL, DifficultyLevel.EASY),
            "medium_done": is_difficulty_finished(SignCategory.VOWEL, DifficultyLevel.MEDIUM),
        },
        SignCategory.CONSONANT: {
            "easy_done": is_difficulty_finished(SignCategory.CONSONANT, DifficultyLevel.EASY),
            "medium_done": is_difficulty_finished(SignCategory.CONSONANT, DifficultyLevel.MEDIUM),
        }
    }

    lessons = []
    for sign in all_signs:
        is_locked = False
        cat_status = completion_status.get(sign.category)

        if sign.difficulty == DifficultyLevel.MEDIUM:
            if not cat_status["easy_done"]:
                is_locked = True
        elif sign.difficulty == DifficultyLevel.HARD:
            if not cat_status["medium_done"]:
                is_locked = True
            
        type_str = "Vowel_Unprepared" if sign.category == SignCategory.VOWEL else "Consonant"
        filename = f"S1_NSL_{type_str}_Bright_S1_{sign.sign_code}_animated.glb"
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

    all_signs_in_cat = db.query(Sign).filter(Sign.category == sign.category).all()
    completed_ids = [r[0] for r in db.query(UserProgress.sign_id).filter(
        UserProgress.user_id == user.id, UserProgress.is_completed == True).all()]

    def check_lock(diff_to_check):
        group = [s.id for s in all_signs_in_cat if s.difficulty == diff_to_check]
        return all(sid in completed_ids for sid in group) if group else True

    is_locked = False
    if sign.difficulty == DifficultyLevel.MEDIUM and not check_lock(DifficultyLevel.EASY):
        is_locked = True
    elif sign.difficulty == DifficultyLevel.HARD and not check_lock(DifficultyLevel.MEDIUM):
        is_locked = True

    if is_locked:
        raise HTTPException(status_code=403, detail="This lesson is locked. Complete the previous level first.")

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