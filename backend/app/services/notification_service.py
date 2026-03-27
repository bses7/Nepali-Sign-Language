from sqlalchemy.orm import Session
from app.models.gamification import Notification

def create_notification(db: Session, user_id: int, title: str, message: str, type: str = "info"):
    new_notif = Notification(
        user_id=user_id,
        title=title,
        message=message,
        type=type
    )
    db.add(new_notif)
    db.commit()
    db.refresh(new_notif)
    return new_notif

def mark_as_read(db: Session, user_id: int, notification_id: int):
    notif = db.query(Notification).filter(
        Notification.id == notification_id, 
        Notification.user_id == user_id
    ).first()
    if notif:
        notif.is_read = True
        db.commit()
    return notif