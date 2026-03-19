from sqlalchemy import JSON, Column, Integer, String, ForeignKey, Table, DateTime, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base

class UserStats(Base):
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    
    xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    streak_count = Column(Integer, default=0)
    
    last_activity_date = Column(Date, nullable=True)
    coins = Column(Integer, default=0) 

    current_avatar_id = Column(Integer, ForeignKey("avatars.id"), nullable=True)

    current_avatar = relationship("Avatar")

    last_claim_date = Column(Date, nullable=True)

    weekly_activity = Column(JSON, default=list)

    daily_challenge_progress = Column(Integer, default=0)
    last_challenge_claim_date = Column(Date, nullable=True)
    
    current_challenge_id = Column(Integer, default=0)
    
    user = relationship("User", back_populates="stats")

user_badges = Table(
    "user_badges",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("badge_id", Integer, ForeignKey("badges.id", ondelete="CASCADE"), primary_key=True),
    Column("earned_at", DateTime(timezone=True), server_default=func.now())
)

class Badge(Base):
    __tablename__ = "badges"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    icon_url = Column(String) 
    badge_code = Column(String, unique=True)

    users = relationship("User", secondary=user_badges, back_populates="badges")