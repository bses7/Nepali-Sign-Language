from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import users  
from app.api.v1.endpoints import users, auth

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

