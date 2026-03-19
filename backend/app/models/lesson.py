import enum
from sqlalchemy import JSON, Column, Integer, String, Enum, ForeignKey, Boolean, UniqueConstraint, DateTime
from sqlalchemy.orm import relationship
from app.db.session import Base
from sqlalchemy.sql import func
from sqlalchemy import Table

class DifficultyLevel(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class SignCategory(str, enum.Enum):
    VOWEL = "vowel"
    CONSONANT = "consonant"

class Sign(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)             
    nepali_char = Column(String)      
    sign_code = Column(String)         
    category = Column(Enum(SignCategory))
    difficulty = Column(Enum(DifficultyLevel))

    description = Column(String, nullable=True)
    
    order_index = Column(Integer, default=0)

    user_progress = relationship("UserProgress", back_populates="sign")

user_avatars = Table(
    "user_avatars",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("avatar_id", Integer, ForeignKey("avatars.id", ondelete="CASCADE"), primary_key=True),
)

class Avatar(Base):
    __tablename__ = "avatars"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    folder_name = Column(String)   
    price = Column(Integer, default=0) 

    attributes = Column(JSON, nullable=True)

class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"))
    is_completed = Column(Boolean, default=False)

    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    __table_args__ = (UniqueConstraint('user_id', 'sign_id', name='_user_sign_uc'),)

    sign = relationship("Sign", back_populates="user_progress")