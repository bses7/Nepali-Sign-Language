import random
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.lesson import Sign, SignCategory, DifficultyLevel
from app.models.user import User

def generate_quiz(db: Session, user: User, category: SignCategory, difficulty: DifficultyLevel):
    pool = db.query(Sign).filter(
        Sign.category == category,
        Sign.difficulty == difficulty
    ).all()

    if len(pool) < 4:
        raise HTTPException(status_code=400, detail="Not enough signs in this category to create a quiz.")

    correct_signs = random.sample(pool, 3)
    
    avatar = user.stats.current_avatar
    avatar_folder = avatar.folder_name if avatar else "avatar"

    questions = []
    for correct_s in correct_signs:
        distractors = random.sample([s for s in pool if s.id != correct_s.id], 3)
        
        options = [
            {"sign_id": correct_s.id, "nepali_char": correct_s.nepali_char}
        ] + [
            {"sign_id": d.id, "nepali_char": d.nepali_char} for d in distractors
        ]
        random.shuffle(options)

        type_str = "Vowel_Unprepared" if correct_s.category == SignCategory.VOWEL else "Consonant"
        model_url = f"/static/avatars/{avatar_folder}/S1_NSL_{type_str}_Bright_S1_{correct_s.sign_code}_animated.glb"

        questions.append({
            "correct_sign_id": correct_s.id,
            "model_url": model_url,
            "animation_name": f"{correct_s.sign_code}_anim",
            "options": options
        })

    return {
        "category": category,
        "difficulty": difficulty,
        "questions": questions
    }