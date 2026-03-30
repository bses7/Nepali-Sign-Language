from pydantic import BaseModel, ConfigDict
from typing import List
from app.models.lesson import DifficultyLevel, SignCategory

class QuizOption(BaseModel):
    sign_id: int
    nepali_char: str

class QuizQuestion(BaseModel):
    correct_sign_id: int
    model_url: str     
    animation_name: str
    options: List[QuizOption] 

class QuizSession(BaseModel):
    category: SignCategory
    difficulty: DifficultyLevel
    questions: List[QuizQuestion]

class QuizSubmit(BaseModel):
    category: SignCategory
    difficulty: DifficultyLevel
    score: int