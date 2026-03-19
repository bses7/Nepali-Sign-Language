from sqlalchemy.orm import Session
from app.models.user import User
from app.models.lesson import Sign, UserProgress

from datetime import date, timedelta
from sqlalchemy.orm import Session
from app.models.gamification import UserStats

def get_dashboard_data(db: Session, user: User):
    total_signs = db.query(Sign).count()
    
    completed_signs = db.query(UserProgress).filter(
        UserProgress.user_id == user.id,
        UserProgress.is_completed == True
    ).count()
    
    percentage = (completed_signs / total_signs * 100) if total_signs > 0 else 0

    # SAFE ACCESS: Use or 0/None defaults if stats or fields are missing
    stats = user.stats
    xp = stats.xp if stats and stats.xp is not None else 0
    level = stats.level if stats and stats.level is not None else 1
    coins = stats.coins if stats and stats.coins is not None else 0
    streak = stats.streak_count if stats and stats.streak_count is not None else 0
    
    last_claim = stats.last_claim_date if stats else None
    can_claim = last_claim != date.today()

    update_activity_log(db, user.stats)

    challenge = get_challenge_for_today()

    if stats.last_activity_date != date.today():
        progress = 0
    else:
        progress = stats.daily_challenge_progress or 0

    not_claimed_yet = stats.last_challenge_claim_date != date.today()
    
    can_claim_challenge = (progress >= challenge["count"]) and not_claimed_yet

    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": user.role,
        "xp": xp,
        "level": level,
        "coins": coins,
        "streak_count": streak,
        "total_signs": total_signs,
        "completed_signs": completed_signs,
        "progress_percentage": round(percentage, 2),
        "equipped_avatar_id": stats.current_avatar_id if stats else None,
        "equipped_avatar_folder": stats.current_avatar.folder_name if stats and stats.current_avatar else "avatar",
        "can_claim_daily": can_claim,
        "weekly_activity": user.stats.weekly_activity,
        "challenge_title": challenge["title"],
        "challenge_description": challenge["description"],
        "challenge_progress": progress,
        "challenge_target": challenge["count"],
        "can_claim_challenge": can_claim_challenge
    }

def update_streak(db: Session, stats: UserStats):
    today = date.today()
    
    if not stats.last_activity_date:
        stats.streak_count = 1
        stats.last_activity_date = today
    
    elif stats.last_activity_date == today - timedelta(days=1):
        stats.streak_count += 1
        stats.last_activity_date = today
        
    elif stats.last_activity_date < today - timedelta(days=1):
        stats.streak_count = 1
        stats.last_activity_date = today
    
    db.add(stats)
    db.commit()
    return stats

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
    # Fetch top users by joining with UserStats and sorting by XP
    top_players = db.query(User).join(UserStats).order_by(UserStats.xp.desc()).limit(limit).all()
    
    leaderboard = []
    for user in top_players:
        leaderboard.append({
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": mask_email(user.email), # Use the mask function here
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
    
    # 2. FIX: Ensure coins is not None before adding
    if user.stats.coins is None:
        user.stats.coins = 0

    # 3. Validation
    if user.stats.last_claim_date == today:
        return False, "Reward already claimed for today."
    
    # 4. Reward
    reward_amount = 100
    user.stats.coins += reward_amount
    user.stats.last_claim_date = today
    
    db.commit()
    return True, f"Successfully claimed {reward_amount} coins!"

def update_activity_log(db: Session, stats: UserStats):
    today = date.today()
    current_weekday = today.weekday() 
    
    if stats.weekly_activity is None:
        stats.weekly_activity = []

    if stats.last_activity_date:
        current_monday = today - timedelta(days=today.weekday())
        last_monday = stats.last_activity_date - timedelta(days=stats.last_activity_date.weekday())
        
        if current_monday > last_monday:
            stats.weekly_activity = [current_weekday]
        else:
            if current_weekday not in stats.weekly_activity:
                updated_list = list(stats.weekly_activity)
                updated_list.append(current_weekday)
                stats.weekly_activity = updated_list
    else:
        stats.weekly_activity = [current_weekday]

    stats.last_activity_date = today
    db.add(stats)
    db.commit()

CHALLENGE_POOL = [
    {"id": 0, "title": "The Academic", "description": "Complete 5 Vowels", "target_cat": "vowel", "target_diff": None, "count": 5},
    {"name": 1, "title": "The Grinder", "description": "Complete 5 Lessons of any kind", "target_cat": None, "target_diff": None, "count": 5},
    {"id": 2, "title": "The Brave", "description": "Complete 3 Hard Difficulty Lessons", "target_cat": None, "target_diff": "hard", "count": 3},
    {"id": 3, "title": "Consonant King", "description": "Complete 5 Consonants", "target_cat": "consonant", "target_diff": None, "count": 5},
    {"id": 4, "title": "Quick Learner", "description": "Complete 3 Medium Difficulty Lessons", "target_cat": None, "target_diff": "medium", "count": 3}
]

def get_challenge_for_today():
    day_of_year = date.today().timetuple().tm_yday
    return CHALLENGE_POOL[day_of_year % len(CHALLENGE_POOL)]

def update_daily_challenge(db: Session, stats: UserStats, completed_sign):
    today = date.today()
    challenge = get_challenge_for_today()

    if stats.last_activity_date != today:
        stats.daily_challenge_progress = 0
        stats.current_challenge_id = challenge["id"]

    # Ensure progress is at least 0 before incrementing
    if stats.daily_challenge_progress is None:
        stats.daily_challenge_progress = 0

    matches_cat = challenge["target_cat"] is None or completed_sign.category == challenge["target_cat"]
    matches_diff = challenge["target_diff"] is None or completed_sign.difficulty == challenge["target_diff"]

    if matches_cat and matches_diff:
        if stats.daily_challenge_progress < challenge["count"]:
            stats.daily_challenge_progress += 1
    
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

    # Award
    stats.xp = (stats.xp or 0) + 500
    stats.coins = (stats.coins or 0) + 200
    stats.last_challenge_claim_date = today
    
    db.commit()
    return True, "Challenge complete! 500 XP and 200 Coins awarded."