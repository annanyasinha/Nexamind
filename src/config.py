import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env or env file in BASE_DIR
for env_name in [".env", "env"]:
    env_file = BASE_DIR / env_name
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
        break
else:
    load_dotenv(override=True)

class Settings:
    """
    Central application settings and environment configurations.
    """
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "NexaMind")
    VERSION: str = "1.0.0"
    
    # API & Host Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model Configurations
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
    LLM_MODEL_CANDIDATES: list = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-lite"
    ]
    
    # Vectorstore & Chunking Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Avatar & Branding Customizations
    USER_AVATAR: str = os.getenv("USER_AVATAR", "👤")
    AI_AVATAR: str = os.getenv("AI_AVATAR", "🤖")
    
    # Directory Paths
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    FAISS_STORE_DIR: Path = BASE_DIR / os.getenv("FAISS_STORE_DIR", "faiss_store")

settings = Settings()

