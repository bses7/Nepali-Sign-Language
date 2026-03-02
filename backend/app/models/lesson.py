import enum
from sqlalchemy import Column, Integer, String, Enum, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.session import Base
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
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE")),
    Column("avatar_id", Integer, ForeignKey("avatars.id", ondelete="CASCADE")),
)

class Avatar(Base):
    __tablename__ = "avatars"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    folder_name = Column(String)   
    price = Column(Integer, default=0) # 0 means it's free/default  

class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"))
    is_completed = Column(Boolean, default=False)

    __table_args__ = (UniqueConstraint('user_id', 'sign_id', name='_user_sign_uc'),)

    sign = relationship("Sign", back_populates="user_progress")