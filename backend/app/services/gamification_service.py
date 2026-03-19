from datetime import datetime, date
from sqlalchemy.orm import Session
from app.models.gamification import Badge, user_badges
from app.models.lesson import Sign, UserProgress, SignCategory, DifficultyLevel
from app.models.user import User

def check_and_award_badges(db: Session, user: User):
    """
    The Master Badge Checker. 
    Call this whenever a user completes a lesson or performs a major action.
    """
    new_badges_earned = []
    owned_codes = [b.badge_code for b in user.badges]

    def check_category_complete(cat: SignCategory = None, diff: DifficultyLevel = None):
        query_total = db.query(Sign)
        query_done = db.query(UserProgress).join(Sign).filter(UserProgress.user_id == user.id)
        
        if cat:
            query_total = query_total.filter(Sign.category == cat)
            query_done = query_done.filter(Sign.category == cat)
        if diff:
            query_total = query_total.filter(Sign.difficulty == diff)
            query_done = query_done.filter(Sign.difficulty == diff)
            
        total = query_total.count()
        done = query_done.count()
        return done >= total and total > 0

    if "VOWEL_MASTER" not in owned_codes:
        if check_category_complete(cat=SignCategory.VOWEL):
            award_badge(db, user, "VOWEL_MASTER")
            new_badges_earned.append("VOWEL_MASTER")

    if "CONSONANT_MASTER" not in owned_codes:
        if check_category_complete(cat=SignCategory.CONSONANT):
            award_badge(db, user, "CONSONANT_MASTER")
            new_badges_earned.append("CONSONANT_MASTER")

    if "EASY_MASTER" not in owned_codes:
        if check_category_complete(diff=DifficultyLevel.EASY):
            award_badge(db, user, "EASY_MASTER")
            new_badges_earned.append("EASY_MASTER")

    if "ALPHABET_ACE" not in owned_codes:
        if check_category_complete(): 
            award_badge(db, user, "ALPHABET_ACE")
            new_badges_earned.append("ALPHABET_ACE")

    if "EARLY_BIRD" not in owned_codes:
        if datetime.now().hour < 7:
            award_badge(db, user, "EARLY_BIRD")
            new_badges_earned.append("EARLY_BIRD")

    if "NIGHT_OWL" not in owned_codes:
        hr = datetime.now().hour
        if hr >= 22 or hr <= 4:
            award_badge(db, user, "NIGHT_OWL")
            new_badges_earned.append("NIGHT_OWL")

    if "CONSISTENT_LEARNER" not in owned_codes:
        if user.stats.streak_count >= 7:
            award_badge(db, user, "CONSISTENT_LEARNER")
            new_badges_earned.append("CONSISTENT_LEARNER")

    if "LEVEL_2" not in owned_codes:
        if user.stats.level >= 2:
            award_badge(db, user, "LEVEL_2")
            new_badges_earned.append("LEVEL_2")

    if "COIN_500" not in owned_codes:
        if user.stats.coins >= 500:
            award_badge(db, user, "COIN_500")
            new_badges_earned.append("COIN_500")

    if "SOCIAL_LINK" not in owned_codes:
        if user.google_id or user.github_id:
            award_badge(db, user, "SOCIAL_LINK")
            new_badges_earned.append("SOCIAL_LINK")

    if "SPEED_DEMON" not in owned_codes:
        today_done = db.query(UserProgress).filter(
            UserProgress.user_id == user.id,
            UserProgress.is_completed == True,
            UserProgress.updated_at >= date.today() 
        ).count()
        if today_done >= 5:
            award_badge(db, user, "SPEED_DEMON")
            new_badges_earned.append("SPEED_DEMON")

    if "AVATAR_BUYER" not in owned_codes:
        if len(user.owned_avatars) > 2:
            award_badge(db, user, "AVATAR_BUYER")
            new_badges_earned.append("AVATAR_BUYER")

    return new_badges_earned

def award_badge(db: Session, user: User, badge_code: str):
    badge = db.query(Badge).filter(Badge.badge_code == badge_code).first()
    if badge and badge not in user.badges:
        user.badges.append(badge)
        db.commit()