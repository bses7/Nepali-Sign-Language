from pydantic import BaseModel, ConfigDict
from typing import Optional
from app.models.lesson import DifficultyLevel, SignCategory

class Avatar(BaseModel):
    id: int
    name: str
    folder_name: str 
    
    model_config = ConfigDict(from_attributes=True)

class LessonOut(BaseModel):
    id: int
    title: str
    nepali_char: str
    difficulty: DifficultyLevel
    category: SignCategory
    is_completed: bool
    is_locked: bool

    model_url: str 

    model_config = ConfigDict(from_attributes=True)

class LessonComplete(BaseModel):
    sign_id: int