from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User as UserModel
from app.schemas.user import User, UserCreate
from app.core.security import get_password_hash

from app.api.deps import get_current_user
from app.models.gamification import UserStats as StatsModel

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

    new_stats = StatsModel(user_id=db_user.id, xp=0, level=1)
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

