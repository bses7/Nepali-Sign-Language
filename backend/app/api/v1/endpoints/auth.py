from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User as UserModel
from app.core.security import verify_password, create_access_token
from app.schemas.token import Token

from app.core.email import send_reset_password_email
from app.core.security import generate_password_reset_token, verify_password_reset_token, get_password_hash
from app.schemas.user import ForgotPasswordRequest, ResetPasswordConfirm

router = APIRouter()

@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db), 
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = db.query(UserModel).filter(UserModel.email == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token = create_access_token(subject=user.email)
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }

@router.post("/password-recovery")
async def recover_password(user_in: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == user_in.email).first()
    if not user:
        return {"message": "If the email exists, a reset link has been sent."}
    
    token = generate_password_reset_token(email=user.email)
    await send_reset_password_email(email_to=user.email, token=token)
    return {"message": "Password reset email sent."}

@router.post("/reset-password")
def reset_password(data: ResetPasswordConfirm, db: Session = Depends(get_db)):
    email = verify_password_reset_token(data.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update password
    user.hashed_password = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}