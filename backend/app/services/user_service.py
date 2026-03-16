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
    
    # 1. Create stats if they don't exist
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