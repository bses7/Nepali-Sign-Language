from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# Define the same Enum for validation
class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    phone_number: Optional[str] = None
    role: UserRole = UserRole.STUDENT
    is_active: Optional[bool] = True

    model_config = ConfigDict(from_attributes=True)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)

class User(UserBase):
    id: int
    is_superuser: bool
    created_at: datetime

class UserStatsSchema(BaseModel):
    xp: int
    level: int
    streak_count: int
    
    model_config = ConfigDict(from_attributes=True)

class User(UserBase):
    id: int
    is_superuser: bool
    created_at: datetime
    stats: Optional[UserStatsSchema] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)