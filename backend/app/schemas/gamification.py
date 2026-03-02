from pydantic import BaseModel, ConfigDict
from datetime import datetime

class BadgeBase(BaseModel):
    name: str
    description: str
    icon_url: str
    badge_code: str

class Badge(BadgeBase):
    id: int
    earned_at: datetime | None = None 

    model_config = ConfigDict(from_attributes=True)