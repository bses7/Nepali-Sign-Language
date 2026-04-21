import os
import platform
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8 # 8 days

    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str

    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: str = ""
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: str = ""
    EMAILS_FROM_NAME: str = ""

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    CONFIG_PATH: str = str(PROJECT_ROOT / "config" / "config.yaml")
    
    @property
    def ml_config(self):
        with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @property
    def RECOGNIZER_MODEL_PATH(self) -> str:
        return str(PROJECT_ROOT / self.ml_config['rec_training']['model_save_path'])

    @property
    def VOCAB_PATH(self) -> str:
        return str(PROJECT_ROOT / self.ml_config['paths']['vocab_path'])
    
    @property
    def BLENDER_EXE(self) -> str:
        """
        Dynamically detects the Blender executable path.
        Priority: 1. ENV Var, 2. OS Detection, 3. YAML config
        """
        env_path = os.getenv("BLENDER_PATH")
        if env_path:
            return env_path

        current_os = platform.system()

        if current_os == "Windows":
            return self.ml_config['blender']['executable_path']
        else:
            return "blender"

    @property
    def BLENDER_SCRIPT(self) -> str:
        return str(PROJECT_ROOT / self.ml_config['blender']['script_path'])
    
    @property
    def GENERATOR_MODEL_PATH(self) -> str:
        return str(PROJECT_ROOT / self.ml_config['training']['model_save_path'])
    
    @property
    def AVATARS_BASE_DIR(self) -> Path:
        return PROJECT_ROOT / "data" / "Avatars"
    
    @property
    def STATIC_DIR(self) -> Path:
        return PROJECT_ROOT / "backend" / "static"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()