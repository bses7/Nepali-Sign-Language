from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User as UserModel
from app.schemas.user import LeaderboardOut, User, UserCreate
from app.core.security import get_password_hash

from app.api.deps import get_current_user
from app.models.gamification import UserStats as StatsModel
from app.schemas.user import DashboardOut
from app.services import user_service

from app.models.lesson import Avatar as AvatarModel
from app.schemas.gamification import Badge as BadgeSchema
from app.services import gamification_service

from app.models.gamification import Badge as BadgeModel, user_badges
from app.schemas.gamification import BadgeStatusOut
from sqlalchemy import select


router = APIRouter()

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserModel).filter(UserModel.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if user_in.phone_number:
        if db.query(UserModel).filter(UserModel.phone_number == user_in.phone_number).first():
            raise HTTPException(status_code=400, detail="Phone number already registered")


    user_data = user_in.model_dump()
    password = user_data.pop("password")
    
    db_user = UserModel(
        **user_data,
        hashed_password=get_password_hash(password)
    )
    
    db.add(db_user)
    db.flush()

    free_avatars = db.query(AvatarModel).filter(AvatarModel.price == 0).all()
    for avatar in free_avatars:
        db_user.owned_avatars.append(avatar)

    default_avatar = next((a for a in free_avatars if a.folder_name == "avatar"), None)

    new_stats = StatsModel(
        user_id=db_user.id, 
        xp=0, 
        level=1, 
        coins=0,
        current_avatar_id=default_avatar.id if default_avatar else None
    )
    db.add(new_stats)

    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/me", response_model=User)
def read_user_me(current_user: UserModel = Depends(get_current_user)):
    """
    Fetch the profile of the currently logged-in user.
    """
    return current_user

@router.get("/dashboard", response_model=DashboardOut)
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all stats needed for the Student Dashboard.
    """
    user_service.update_streak(db, current_user.stats)

    gamification_service.check_and_award_badges(db, current_user)

    return user_service.get_dashboard_data(db, current_user)

@router.get("/leaderboard", response_model=LeaderboardOut)
def read_leaderboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Only logged in users see it
):
    """
    Get the top 10 players by XP.
    """
    return user_service.get_leaderboard(db)

@router.get("/badges", response_model=list[BadgeSchema])
def get_my_badges(current_user: User = Depends(get_current_user)):
    """View all earned badges."""
    return current_user.badges

@router.get("/badges/all", response_model=list[BadgeStatusOut])
def get_all_badges_with_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns EVERY badge in the app, marked with is_earned=True/False 
    for the current user. Perfect for a 'Trophy Room' view.
    """
    all_badges = db.query(BadgeModel).all()
    
    stmt = select(user_badges).where(user_badges.c.user_id == current_user.id)
    earned_rows = db.execute(stmt).fetchall()
    
    earned_map = {row.badge_id: row.earned_at for row in earned_rows}

    result = []
    for badge in all_badges:
        is_earned = badge.id in earned_map
        result.append({
            "id": badge.id,
            "name": badge.name,
            "description": badge.description,
            "icon_url": badge.icon_url,
            "badge_code": badge.badge_code,
            "is_earned": is_earned,
            "earned_at": earned_map.get(badge.id) if is_earned else None
        })
    
    return result

@router.post("/claim-daily")
def claim_reward(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Claim the daily 100 coin reward.
    """
    success, message = user_service.claim_daily_reward(db, current_user)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
        
    return {"message": message, "new_balance": current_user.stats.coins}

@router.post("/claim-challenge")
def claim_challenge(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Claim the 500 XP and 200 Coin daily challenge reward."""
    success, message = user_service.claim_daily_challenge(db, current_user)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "xp": current_user.stats.xp, "coins": current_user.stats.coins}

