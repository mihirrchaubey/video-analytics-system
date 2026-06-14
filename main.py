import os
import hashlib
import shutil
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, VideoMetadata
from app.schemas import VideoUploadResponse, QueryRequest, QueryResponse
from app.video_processor import process_video, query_frames

# Configure module-level logger for production observability
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Application Constants
FILE_READ_CHUNK_SIZE = 4096
VIDEO_ID_PREFIX_LENGTH = 8

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Initializes the SQLite database tables upon server startup.
    """
    logger.info("Initializing database tables...")
    init_db()
    yield
    logger.info("Application shutting down.")

# Initialize the FastAPI application
app = FastAPI(
    title="Semantic Video Analytics System",
    description="API for ingesting videos, extracting frames, and querying via CLIP embeddings.",
    version="1.0.0",
    lifespan=lifespan
)

def compute_file_hash(file_path: str) -> str:
    """
    Computes the SHA-256 cryptographic hash of a file.
    Reads the file in chunks to ensure low memory consumption for large videos.

    Args:
        file_path (str): The absolute or relative path to the physical file.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(FILE_READ_CHUNK_SIZE), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)) -> VideoUploadResponse:
    """
    Handles incoming video files, computes a SHA-256 hash to prevent duplicates,
    and initiates the frame extraction and embedding pipeline.

    Args:
        file (UploadFile): The uploaded video file stream from the client.
        db (Session): SQLAlchemy database session for metadata tracking.

    Returns:
        VideoUploadResponse: Contains unique video_id, hash, and processing status.

    Raises:
        HTTPException: If video processing fails, the file is invalid, or the system errors out.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    logger.info(f"Receiving file upload: {file.filename}")
    temp_file_path = os.path.join(settings.video_storage_path, file.filename)
    
    # Safely write the uploaded stream to local storage
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Compute hash to detect duplicates instantly (O(1) lookup avoidance)
    file_hash = compute_file_hash(temp_file_path)
    existing_video = db.query(VideoMetadata).filter(VideoMetadata.file_hash == file_hash).first()
    
    if existing_video:
        logger.info(f"Duplicate detected for hash {file_hash[:8]}... Bypassing ML processing.")
        os.remove(temp_file_path)
        return VideoUploadResponse(
            message="Duplicate video detected. Bypassing processing.", 
            video_id=existing_video.filename, 
            file_hash=file_hash, 
            status="duplicate"
        )
        
    # Generate a unique video ID based on the hash prefix
    video_id = f"vid_{file_hash[:VIDEO_ID_PREFIX_LENGTH]}"
    final_file_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    os.rename(temp_file_path, final_file_path)
    
    # Track metadata in the relational database
    new_video = VideoMetadata(filename=video_id, file_hash=file_hash)
    db.add(new_video)
    db.commit()

    # Trigger ML extraction and vector embedding pipeline
    try:
        logger.info(f"Starting ML processing for video ID: {video_id}...")
        frames_extracted = process_video(final_file_path, video_id)
        logger.info(f"Successfully processed {video_id}. Extracted {frames_extracted} frames.")
    except ValueError as ve:
        logger.error(f"Validation error processing video {video_id}: {ve}")
        db.delete(new_video)
        db.commit()
        if os.path.exists(final_file_path): 
            os.remove(final_file_path)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Critical system error processing video {video_id}: {e}")
        db.delete(new_video)
        db.commit()
        if os.path.exists(final_file_path): 
            os.remove(final_file_path)
        raise HTTPException(status_code=500, detail=str(e))

    return VideoUploadResponse(
        message=f"Video processed successfully. Extracted {frames_extracted} frames.", 
        video_id=video_id, 
        file_hash=file_hash, 
        status="processed"
    )

@app.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest) -> QueryResponse:
    """
    Receives a natural language query, validates the request, and queries 
    the vector database to retrieve the most semantically relevant frames.

    Args:
        request (QueryRequest): Payload containing text query, threshold, and video ID.

    Returns:
        QueryResponse: Ranked list of matching frames, timestamps, and confidence scores.
    """
    logger.info(f"Executing search query: '{request.query}' against video '{request.video_id}'")
    
    results = query_frames(
        query_text=request.query, 
        threshold=request.threshold, 
        video_id=request.video_id
    )
    
    if not results:
        logger.info(f"No frames met the threshold ({request.threshold}) for query: '{request.query}'.")
        return QueryResponse(
            query=request.query, 
            results=[], 
            message="No matching content found above the confidence threshold."
        )
        
    return QueryResponse(
        query=request.query, 
        results=results, 
        message=f"Matches found: {len(results)} frames retrieved."
    )

@app.get("/videos", response_model=List[str])
async def list_videos(db: Session = Depends(get_db)) -> List[str]:
    """
    Retrieves a list of all successfully processed video IDs stored in the system.

    Args:
        db (Session): SQLAlchemy database session.

    Returns:
        List[str]: A list of video identifiers currently available for semantic querying.
    """
    videos = db.query(VideoMetadata).all()
    return [v.filename for v in videos]