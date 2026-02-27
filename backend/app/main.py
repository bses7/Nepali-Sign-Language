from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import users  
from app.api.v1.endpoints import users, auth, lessons

from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(lessons.router, prefix="/api/v1/lessons", tags=["lessons"])

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

