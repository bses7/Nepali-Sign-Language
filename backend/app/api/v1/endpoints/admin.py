from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.api.deps import get_admin_user
from app.models.user import User as UserModel
from app.schemas.user import TeacherAdminView, UserRole
from app.core.email import send_teacher_verified_email

router = APIRouter()

@router.get("/teachers", response_model=List[TeacherAdminView])
def list_teachers_for_admin(
    db: Session = Depends(get_db),
    admin: UserModel = Depends(get_admin_user) 
):
    return db.query(UserModel).filter(UserModel.role == UserRole.TEACHER).all()

@router.post("/verify-teacher/{user_id}")
async def verify_teacher(
    user_id: int,
    db: Session = Depends(get_db),
    admin: UserModel = Depends(get_admin_user)
):
    """Verify a teacher and send them a notification email."""
    teacher = db.query(UserModel).filter(UserModel.id == user_id).first()
    
    if not teacher or teacher.role != "teacher":
        raise HTTPException(status_code=404, detail="Teacher not found")

    if teacher.is_verified_teacher:
        return {"message": "Teacher is already verified"}

    teacher.is_verified_teacher = True
    db.commit()

    try:
        await send_teacher_verified_email(teacher.email, teacher.first_name)
    except Exception as e:
        print(f"Email failed to send: {e}")

    return {"message": f"Teacher {teacher.first_name} has been verified and notified."}