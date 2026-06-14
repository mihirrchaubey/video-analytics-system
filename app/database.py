from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# Initialize the SQLAlchemy engine. 
# 'check_same_thread: False' is required for SQLite compatibility when used 
# with asynchronous web frameworks like FastAPI.
engine = create_engine(
    settings.database_url, 
    connect_args={"check_same_thread": False}
)

# Configure the session factory for database transactions.
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

Base = declarative_base()

class VideoMetadata(Base):
    """
    SQLAlchemy model representing the stored video metadata.

    Attributes:
        id (int): Primary key for database indexing.
        filename (str): The unique identifier/path of the video.
        file_hash (str): SHA-256 hash of the video file for duplicate detection.
        upload_time (datetime): UTC timestamp when the video was initially processed.
    """
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_hash = Column(String, unique=True, index=True, nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db() -> None:
    """
    Initializes the database by creating all tables defined by the 
    declarative base metadata.
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Dependency generator function that yields a database session 
    and ensures it is safely closed after the request lifecycle completes.

    Yields:
        Session: An active SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()