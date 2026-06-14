from pydantic import BaseModel, Field
from typing import List, Optional

class VideoUploadResponse(BaseModel):
    """
    Response schema for the video upload endpoint.
    
    Attributes:
        message (str): Human-readable status message.
        video_id (str): Unique identifier assigned to the uploaded video.
        file_hash (str): SHA-256 cryptographic hash of the video file.
        status (str): Processing status (e.g., 'processed' or 'duplicate').
    """
    message: str
    video_id: str
    file_hash: str
    status: str

class FrameResult(BaseModel):
    """
    Schema representing a single matched frame from the semantic search.
    
    Attributes:
        frame_path (str): File system path to the extracted JPEG image.
        timestamp (str): The calculated timestamp (in seconds) of the frame.
        similarity_score (float): Normalized cosine similarity metric.
    """
    frame_path: str
    timestamp: str
    similarity_score: float

class QueryRequest(BaseModel):
    """
    Request schema for the semantic query payload.
    
    Attributes:
        query (str): The natural language string to search for (minimum 3 characters).
        threshold (float): Minimum confidence threshold (0.0 to 1.0) required.
        video_id (Optional[str]): Target video to filter the search against.
    """
    query: str = Field(..., min_length=3)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    video_id: Optional[str] = None

class QueryResponse(BaseModel):
    """
    Response schema returning the matched frames for a given query.
    
    Attributes:
        query (str): The original text query requested by the user.
        results (List[FrameResult]): A ranked list of matching frames.
        message (str): Contextual message indicating success or empty results.
    """
    query: str
    results: List[FrameResult]
    message: str