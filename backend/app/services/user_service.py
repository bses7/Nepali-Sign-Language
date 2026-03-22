from sqlalchemy.orm import Session
from app.models.user import User
from app.models.lesson import Sign, UserProgress

from datetime import date, timedelta
from sqlalchemy.orm import Session
from app.models.gamification import UserStats

def get_dashboard_data(db: Session, user: User):
    sync_daily_stats(db, user.stats)
    
    stats = user.stats
    today = date.today()
    challenge = get_challenge_for_today()

    current_progress = stats.daily_challenge_progress or 0
    can_claim_daily = stats.last_claim_date != today
    is_finished = current_progress >= challenge["count"]
    not_claimed_challenge = stats.last_challenge_claim_date != today

    total_signs = db.query(Sign).count()
    completed_signs = db.query(UserProgress).filter(UserProgress.user_id == user.id, UserProgress.is_completed == True).count()
    percentage = (completed_signs / total_signs * 100) if total_signs > 0 else 0

    google_id = user.google_id if hasattr(user, 'google_id') else None
    github_id = user.github_id if hasattr(user, 'github_id') else None

    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": user.role,
        "phone_number": user.phone_number,
        "xp": stats.xp or 0,
        "level": stats.level or 1,
        "coins": stats.coins or 0,
        "streak_count": stats.streak_count or 0,
        "total_signs": total_signs,
        "completed_signs": completed_signs,
        "progress_percentage": round(percentage, 2),
        "equipped_avatar_id": stats.current_avatar_id,
        "equipped_avatar_folder": stats.current_avatar.folder_name if stats.current_avatar else "avatar",
        "weekly_activity": stats.weekly_activity or [],
        "can_claim_daily": can_claim_daily,
        "challenge_title": challenge["title"],
        "challenge_description": challenge["description"],
        "challenge_progress": current_progress,
        "challenge_target": challenge["count"],
        "can_claim_challenge": is_finished and not_claimed_challenge,
        "google_id": google_id,
        "github_id": github_id
    }

def sync_daily_stats(db: Session, stats: UserStats):
    """
    Handles Streaks, Challenge Resets, and Weekly Tracking.
    """
    if not stats: return
    today = date.today()
    
    if stats.last_activity_date != today:
        
        if stats.last_activity_date == today - timedelta(days=1):
            stats.streak_count += 1
        elif not stats.last_activity_date or stats.last_activity_date < today - timedelta(days=1):
            stats.streak_count = 1 
            
        stats.daily_challenge_progress = 0
        today_challenge = get_challenge_for_today()
        stats.current_challenge_id = today_challenge["id"]
        
        current_weekday = today.weekday()
        if stats.last_activity_date:
            current_monday = today - timedelta(days=today.weekday())
            last_monday = stats.last_activity_date - timedelta(days=stats.last_activity_date.weekday())
            if current_monday > last_monday:
                stats.weekly_activity = [current_weekday]
            else:
                updated_list = list(stats.weekly_activity or [])
                if current_weekday not in updated_list:
                    updated_list.append(current_weekday)
                    stats.weekly_activity = updated_list
        else:
            stats.weekly_activity = [current_weekday]

        stats.last_activity_date = today
        db.add(stats)
        db.commit()
        db.refresh(stats)


def mask_email(email: str) -> str:
    """
    Turns 'bisheshgiri@gmail.com' into 'bi***@gmail.com'
    """
    try:
        user_part, domain_part = email.split("@")
        if len(user_part) <= 2:
            return f"{user_part}***@{domain_part}"
        return f"{user_part[:2]}***@{domain_part}"
    except:
        return "***@***.com"

def get_leaderboard(db: Session, limit: int = 10):
    top_players = db.query(User).join(UserStats).order_by(UserStats.xp.desc()).limit(limit).all()
    
    leaderboard = []
    for user in top_players:
        leaderboard.append({
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": mask_email(user.email), 
            "xp": user.stats.xp,
            "level": user.stats.level,
            "streak_count": user.stats.streak_count
        })
        
    return {"top_users": leaderboard}

def claim_daily_reward(db: Session, user: User):
    today = date.today()
    
    if not user.stats:
        new_stats = UserStats(
            user_id=user.id, 
            coins=0, 
            xp=0, 
            level=1,
            current_avatar_id=None 
        )
        db.add(new_stats)
        db.flush() 
    
    if user.stats.coins is None:
        user.stats.coins = 0

    if user.stats.last_claim_date == today:
        return False, "Reward already claimed for today."
    
    reward_amount = 100
    user.stats.coins += reward_amount
    user.stats.last_claim_date = today
    
    db.commit()
    return True, f"Successfully claimed {reward_amount} coins!"


CHALLENGE_POOL = [
    {"id": 0, "title": "The Academic", "description": "Complete 5 Vowels", "target_cat": "vowel", "target_diff": None, "count": 5},
    {"id": 1, "title": "The Grinder", "description": "Complete 5 Lessons of any kind", "target_cat": None, "target_diff": None, "count": 5},
    {"id": 2, "title": "The Brave", "description": "Complete 3 Hard Difficulty Lessons", "target_cat": None, "target_diff": "hard", "count": 3},
    {"id": 3, "title": "Consonant King", "description": "Complete 5 Consonants", "target_cat": "consonant", "target_diff": None, "count": 5},
    {"id": 4, "title": "Quick Learner", "description": "Complete 3 Medium Difficulty Lessons", "target_cat": None, "target_diff": "medium", "count": 3}
]

def get_challenge_for_today():
    day_of_year = date.today().timetuple().tm_yday
    return CHALLENGE_POOL[day_of_year % len(CHALLENGE_POOL)]

def update_daily_challenge(db: Session, stats: UserStats, completed_sign):
    sync_daily_stats(db, stats)
    
    challenge = get_challenge_for_today()

    matches_cat = challenge["target_cat"] is None or completed_sign.category == challenge["target_cat"]
    matches_diff = challenge["target_diff"] is None or completed_sign.difficulty == challenge["target_diff"]

    if matches_cat and matches_diff:
        if (stats.daily_challenge_progress or 0) < challenge["count"]:
            stats.daily_challenge_progress = (stats.daily_challenge_progress or 0) + 1
    
    db.add(stats)
    db.commit()

def claim_daily_challenge(db: Session, user: User):
    today = date.today()
    stats = user.stats
    challenge = get_challenge_for_today()

    current_progress = stats.daily_challenge_progress or 0

    if current_progress < challenge["count"]:
        return False, f"Challenge not yet complete! Need {challenge['count']} steps."

    if stats.last_challenge_claim_date == today:
        return False, "Daily challenge reward already claimed today."

    stats.xp = (stats.xp or 0) + 500
    stats.coins = (stats.coins or 0) + 200
    stats.last_challenge_claim_date = today
    
    db.commit()
    return True, "Challenge complete! 500 XP and 200 Coins awarded."