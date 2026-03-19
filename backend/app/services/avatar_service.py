from sqlalchemy.orm import Session
from sqlalchemy import insert, select
from app.models.lesson import Avatar, user_avatars
from app.models.user import User

def get_store_avatars(db: Session, user_id: int):
    all_avatars = db.query(Avatar).all()
    
    stmt = select(user_avatars.c.avatar_id).where(user_avatars.c.user_id == user_id)
    owned_ids = [r[0] for r in db.execute(stmt).fetchall()]

    avatars_list = []
    for a in all_avatars:
        avatars_list.append({
            "id": a.id,
            "name": a.name,
            "folder_name": a.folder_name,
            "price": a.price,
            "is_owned": (a.id in owned_ids) or (a.price == 0),
            "attributes": a.attributes or {}
        })
    return avatars_list

def purchase_avatar(db: Session, user: User, avatar_id: int):
    avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    
    if not avatar:
        return "Avatar not found", False
    
    stmt = select(user_avatars).where(
        user_avatars.c.user_id == user.id, 
        user_avatars.c.avatar_id == avatar_id
    )
    if db.execute(stmt).first() or avatar.price == 0:
        return "You already own this avatar", False

    if user.stats.coins < avatar.price:
        return "Insufficient coins", False

    user.stats.coins -= avatar.price
    db.execute(insert(user_avatars).values(user_id=user.id, avatar_id=avatar.id))
    db.commit()
    
    return "Purchase successful!", True