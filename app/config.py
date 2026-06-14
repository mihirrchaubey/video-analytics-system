import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration model utilizing Pydantic.
    
    Validates and loads environment variables from a local '.env' file 
    or system environment. Ensures that all necessary paths and ML model 
    identities are present before application startup.
    
    Attributes:
        video_storage_path (str): Local directory path to store uploaded MP4s.
        frame_storage_path (str): Local directory path to store extracted JPEG frames.
        chroma_db_path (str): Local directory path for the persistent Chroma vector storage.
        database_url (str): SQLAlchemy connection string (e.g., SQLite file path).
        clip_model_id (str): Hugging Face model identifier for the CLIP architecture.
        target_fps (int): Frame extraction rate. Defaults to 1 Frame Per Second.
    """
    video_storage_path: str
    frame_storage_path: str
    chroma_db_path: str
    database_url: str
    clip_model_id: str
    target_fps: int = 1

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

# Instantiate global settings
settings = Settings()

# Bootstrap Routine: Automatically create storage directories on application 
# startup to prevent missing directory errors during runtime execution.
for path in [settings.video_storage_path, settings.frame_storage_path, settings.chroma_db_path]:
    os.makedirs(path, exist_ok=True)