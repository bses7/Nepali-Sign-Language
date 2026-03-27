from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.services import avatar_service, gamification_service
from app.schemas.lesson import Avatar as AvatarSchema
from app.services import notification_service

router = APIRouter()

@router.get("/store", response_model=list[AvatarSchema])
def get_avatar_store(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """List all avatars with their price and ownership status."""
    return avatar_service.get_store_avatars(db, current_user.id)

@router.post("/purchase/{avatar_id}")
def buy_avatar(avatar_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # 1. Fetch the avatar object first so we have the name
    avatar = db.query(AvatarSchema).filter(AvatarSchema.id == avatar_id).first()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    message, success = avatar_service.purchase_avatar(db, current_user, avatar_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    gamification_service.check_and_award_badges(db, current_user)

    # 2. Use the avatar object name
    notification_service.create_notification(
        db, 
        current_user.id, 
        "Avatar Purchased!", 
        f"You just purchased the {avatar.name} avatar!", 
        "success"
    )
    
    return {"message": message}

@router.post("/equip/{avatar_id}")
def equip_avatar(avatar_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Set the currently active avatar for the whole app."""
    avatars = avatar_service.get_store_avatars(db, current_user.id)
    target = next((a for a in avatars if a["id"] == avatar_id), None)
    
    if not target or not target["is_owned"]:
        raise HTTPException(status_code=400, detail="You do not own this avatar")
    
    current_user.stats.current_avatar_id = avatar_id
    db.commit()
    return {"message": "Avatar equipped!"}