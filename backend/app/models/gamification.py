from sqlalchemy import Column, Integer, ForeignKey, Date, DateTime
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
    
    user = relationship("User", back_populates="stats")