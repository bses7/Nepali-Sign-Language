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
    
    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": user.role,
        "xp": user.stats.xp,
        "level": user.stats.level,
        "streak_count": user.stats.streak_count,
        "total_signs": total_signs,
        "completed_signs": completed_signs,
        "progress_percentage": round(percentage, 2),
        "equipped_avatar_id": user.stats.current_avatar_id,
        "equipped_avatar_folder": user.stats.current_avatar.folder_name if user.stats.current_avatar else "avatar"
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

def get_leaderboard(db: Session, limit: int = 10):
    top_stats = db.query(User).join(UserStats).order_by(UserStats.xp.desc()).limit(limit).all()
    
    leaderboard = []
    for user in top_stats:
        leaderboard.append({
            "first_name": user.first_name,
            "xp": user.stats.xp,
            "level": user.stats.level
        })
    return {"top_users": leaderboard}