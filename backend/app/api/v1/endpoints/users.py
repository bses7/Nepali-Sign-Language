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
    db.commit()
    db.refresh(db_user)
    db.flush()

    default_avatar = db.query(AvatarModel).filter(AvatarModel.folder_name == "avatar").first()
    default_avatar_id = default_avatar.id if default_avatar else None

    new_stats = StatsModel(
        user_id=db_user.id, 
        xp=0, 
        level=1, 
        coins=0,
        current_avatar_id=default_avatar_id 
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