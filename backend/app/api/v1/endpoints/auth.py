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

from app.models.lesson import Avatar as AvatarModel
from app.schemas.gamification import Badge as BadgeSchema
from app.services import gamification_service
from app.models.gamification import UserStats 

from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request
from app.core.config import settings

from fastapi.responses import RedirectResponse

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
    
    user.hashed_password = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}

oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

oauth.register(
    name='github',
    client_id=settings.GITHUB_CLIENT_ID,
    client_secret=settings.GITHUB_CLIENT_SECRET,
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'}, # We need this to get their email
)

@router.get("/login/google")
async def google_login(request: Request):
    """Redirects the user to Google Login Page."""
    redirect_uri = "http://localhost:8000/api/v1/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Google sends the user here after they log in."""
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    
    if not user_info:
        raise HTTPException(status_code=400, detail="Failed to fetch user info from Google")

    user = db.query(UserModel).filter(UserModel.email == user_info['email']).first()

    if user:
        if not user.google_id:
            user.google_id = user_info['sub']
            # Update names if they were missing (common for old manual accounts)
            if not user.first_name: user.first_name = user_info.get('given_name')
            if not user.last_name: user.last_name = user_info.get('family_name')
            db.commit()
        
        if not user.stats:
            free_avatars = db.query(AvatarModel).filter(AvatarModel.price == 0).all()
            default_avatar = next((a for a in free_avatars if a.folder_name == "avatar"), None)
            new_stats = UserStats(
                user_id=user.id, xp=0, level=1, coins=0,
                current_avatar_id=default_avatar.id if default_avatar else None
            )
            db.add(new_stats)
            db.flush()
    
    else:
        user = UserModel(
            email=user_info['email'],
            first_name=user_info.get('given_name', ''),
            last_name=user_info.get('family_name', ''),
            google_id=user_info['sub'],
            is_active=True
        )
        db.add(user)
        db.flush()
        
        free_avatars = db.query(AvatarModel).filter(AvatarModel.price == 0).all()
        for avatar in free_avatars:
            user.owned_avatars.append(avatar)
        
        default_avatar = next((a for a in free_avatars if a.folder_name == "avatar"), None)
        new_stats = UserStats(
            user_id=user.id, 
            xp=0, 
            level=1, 
            coins=0,
            current_avatar_id=default_avatar.id if default_avatar else None
        )

        gamification_service.check_and_award_badges(db, user)
        
        db.add(new_stats)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(subject=user.email)

    frontend_url = "http://localhost:3000/callback"
    return RedirectResponse(url=f"{frontend_url}?token={access_token}")

@router.get("/login/github")
async def github_login(request: Request):
    """Redirects the user to GitHub Login Page."""
    redirect_uri = "http://localhost:8000/api/v1/auth/github/callback"
    return await oauth.github.authorize_redirect(request, redirect_uri)

@router.get("/github/callback")
async def github_callback(request: Request, db: Session = Depends(get_db)):
    """GitHub sends the user here after they log in."""
    token = await oauth.github.authorize_access_token(request)
    
    resp = await oauth.github.get('user', token=token)
    user_info = resp.json()
    
    email = user_info.get('email')
    if not email:
        emails_resp = await oauth.github.get('user/emails', token=token)
        emails = emails_resp.json()
        email = next(e['email'] for e in emails if e['primary'])

    user = db.query(UserModel).filter(UserModel.email == email).first()

    if user:
        if not user.github_id:
            user.github_id = str(user_info['id'])
            db.commit()
        
        if not user.stats:
            free_avatars = db.query(AvatarModel).filter(AvatarModel.price == 0).all()
            default_avatar = next((a for a in free_avatars if a.folder_name == "avatar"), None)
            new_stats = UserStats(
                user_id=user.id, xp=0, level=1, coins=0,
                current_avatar_id=default_avatar.id if default_avatar else None
            )
            db.add(new_stats)
            db.flush()
    
    else:
        full_name = user_info.get('name', 'GitHub User')
        name_parts = full_name.split(" ", 1)
        f_name = name_parts[0]
        l_name = name_parts[1] if len(name_parts) > 1 else ""

        user = UserModel(
            email=email,
            first_name=f_name,
            last_name=l_name,
            github_id=str(user_info['id']),
            is_active=True
        )
        db.add(user)
        db.flush()
        
        free_avatars = db.query(AvatarModel).filter(AvatarModel.price == 0).all()
        for avatar in free_avatars:
            user.owned_avatars.append(avatar)
        
        default_avatar = next((a for a in free_avatars if a.folder_name == "avatar"), None)
        new_stats = UserStats(
            user_id=user.id, 
            current_avatar_id=default_avatar.id if default_avatar else None,
            xp=0, 
            level=1,
            coins=0
        )

        gamification_service.check_and_award_badges(db, user)

        db.add(new_stats)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(subject=user.email)
    
    frontend_url = "http://localhost:3000/callback"
    return RedirectResponse(url=f"{frontend_url}?token={access_token}")

