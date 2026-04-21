from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import avatars, users  
from app.api.v1.endpoints import users, auth, lessons, practice, quiz, teacher, admin

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from pathlib import Path

import mimetypes

from app.api.v1.endpoints import generation


app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(lessons.router, prefix="/api/v1/lessons", tags=["lessons"])
app.include_router(avatars.router, prefix="/api/v1/avatars", tags=["avatars"])
app.include_router(practice.router, prefix="/api/v1/practice", tags=["practice"])
app.include_router(quiz.router, prefix="/api/v1/quiz", tags=["quiz"])
app.include_router(teacher.router, prefix="/api/v1/teacher", tags=["teacher"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(generation.router, prefix="/api/v1/generation", tags=["generation"])

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key="your-very-secret-key")

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

mimetypes.add_type('video/mp4', '.mp4')


base_path = Path(__file__).resolve().parent.parent 
static_path = base_path / "static"

static_path.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

