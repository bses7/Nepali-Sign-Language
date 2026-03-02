from datetime import datetime
from sqlalchemy.orm import Session
from app.models.gamification import Badge, user_badges
from app.models.lesson import Sign, UserProgress, SignCategory
from app.models.user import User

def check_and_award_badges(db: Session, user: User):
    new_badges = []
    
    owned_badge_codes = [b.badge_code for b in user.badges]

    # --- LOGIC 1: VOWEL MASTER ---
    if "VOWEL_MASTER" not in owned_badge_codes:
        total_vowels = db.query(Sign).filter(Sign.category == SignCategory.VOWEL).count()
        completed_vowels = db.query(UserProgress).join(Sign).filter(
            UserProgress.user_id == user.id,
            UserProgress.is_completed == True,
            Sign.category == SignCategory.VOWEL
        ).count()
        
        if completed_vowels >= total_vowels and total_vowels > 0:
            award_badge(db, user, "VOWEL_MASTER")
            new_badges.append("VOWEL_MASTER")

    # --- LOGIC 2: EARLY BIRD (Before 7 AM) ---
    if "EARLY_BIRD" not in owned_badge_codes:
        current_hour = datetime.now().hour
        if current_hour < 7:
            award_badge(db, user, "EARLY_BIRD")
            new_badges.append("EARLY_BIRD")

    # --- LOGIC 3: CONSISTENT LEARNER (7 Day Streak) ---
    if "CONSISTENT_LEARNER" not in owned_badge_codes:
        if user.stats.streak_count >= 7:
            award_badge(db, user, "CONSISTENT_LEARNER")
            new_badges.append("CONSISTENT_LEARNER")

    return new_badges

def award_badge(db: Session, user: User, badge_code: str):
    badge = db.query(Badge).filter(Badge.badge_code == badge_code).first()
    if badge:
        user.badges.append(badge)
        db.commit()