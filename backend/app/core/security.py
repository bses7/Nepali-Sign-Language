from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import jwt
from app.core.config import settings

ph = PasswordHasher()

def get_password_hash(password: str) -> str:
    """
    Hashes a password using Argon2id. 
    No length limit issues like bcrypt.
    """
    return ph.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against the stored hash.
    """
    try:
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False
    
def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def generate_password_reset_token(email: str) -> str:
    delta = timedelta(hours=1) # Reset link expires in 1 hour
    now = datetime.now(timezone.utc)
    expires = now + delta
    encoded_jwt = jwt.encode(
        {"exp": expires, "nbf": now, "sub": email, "purpose": "password_reset"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt

def verify_password_reset_token(token: str) -> str | None:
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if decoded_token["purpose"] != "password_reset":
            return None
        return decoded_token["sub"]
    except jwt.JWTError:
        return None