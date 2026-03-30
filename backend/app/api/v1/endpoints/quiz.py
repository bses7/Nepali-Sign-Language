from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.gamification import QuizResult
from app.schemas.quiz import QuizSession, QuizSubmit
from app.services import quiz_service, gamification_service, notification_service
from app.services import user_service

router = APIRouter()

@router.get("/generate", response_model=QuizSession)
def get_quiz(
    category: str, 
    difficulty: str, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    return quiz_service.generate_quiz(db, current_user, category, difficulty)

@router.post("/submit")
def submit_quiz(
    data: QuizSubmit, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    result = QuizResult(
        user_id=current_user.id,
        category=data.category,
        difficulty=data.difficulty,
        score=data.score
    )
    db.add(result)

    xp_earned = data.score * 50
    coins_earned = data.score * 20
    
    user_service.add_xp(db, current_user, xp_earned)
    current_user.stats.coins += coins_earned

    bonus_msg = ""
    if data.score == 3:
        current_user.stats.coins += 50
        bonus_msg = " +50 Perfect Score Bonus!"

    notification_service.create_notification(
        db, current_user.id, "Quiz Complete!",
        f"You scored {data.score}/3. Earned {xp_earned} XP and {coins_earned} Coins{bonus_msg}",
        "success"
    )

    new_badges = gamification_service.check_and_award_badges(db, current_user)
    
    db.commit()
    return {"message": "Results saved", "xp_earned": xp_earned, "new_badges": new_badges}