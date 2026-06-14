import cv2
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb

from app.config import settings

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = chroma_client.get_or_create_collection(
    name="video_frames", 
    metadata={"hnsw:space": "cosine"}
)

# Configure hardware execution (CPU-optimized for portability)
device = "cpu"
processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
model = CLIPModel.from_pretrained(settings.clip_model_id).to(device)

def process_video(video_path: str, video_id: str) -> int:
    """
    Extracts frames from a video file at the configured FPS, computes CLIP embeddings,
    and persists them to the vector collection.

    Args:
        video_path (str): Filesystem path to the MP4 file.
        video_id (str): Unique identifier for metadata linkage.
        
    Returns:
        int: Number of frames successfully extracted and embedded.
        
    Raises:
        ValueError: If video is corrupted, cannot be decoded, or FPS is unreadable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV Error: Unable to open or decode video file.")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or original_fps is None:
        cap.release()
        raise ValueError("OpenCV Error: Video file has 0 or unreadable FPS (corrupted).")

    # Calculate cadence to strictly extract at target FPS
    frame_skip = int(round(original_fps / settings.target_fps)) or 1
    frame_count, extracted_count = 0, 0
    embeddings_batch, metadatas_batch, ids_batch = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream
            
        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Compute visual embeddings
            inputs = processor(images=[pil_image], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Enforce L2 normalization for accurate cosine similarity
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Save frame to disk
            frame_filename = f"{video_id}_frame_{extracted_count}.jpg"
            frame_path = os.path.join(settings.frame_storage_path, frame_filename)
            pil_image.save(frame_path)
            
            # Queue data for batch ingestion to ChromaDB
            embeddings_batch.append(image_features.cpu().numpy().tolist()[0])
            metadatas_batch.append({
                "frame_path": frame_path, 
                "timestamp": str(extracted_count), 
                "video_id": video_id
            })
            ids_batch.append(f"{video_id}_{extracted_count}")
            
            extracted_count += 1
            
        frame_count += 1

    # Execute batch persistence
    if ids_batch:
        collection.add(
            embeddings=embeddings_batch, 
            metadatas=metadatas_batch, 
            ids=ids_batch
        )
    
    cap.release()
    return extracted_count

def query_frames(query_text: str, threshold: float, video_id: str, top_k: int = 5) -> list:
    """
    Encodes a text query and performs a semantic search against the
    vector collection for the specified video.

    Args:
        query_text (str): Natural language string to search for.
        threshold (float): Minimum similarity score (0.0 to 1.0) for result inclusion.
        video_id (str): The video ID to filter the search results.
        top_k (int): Maximum number of results to return.
        
    Returns:
        list: A list of dicts containing frame_path, timestamp, and similarity_score.
    """
    # Embed the text query into the shared latent space
    with torch.no_grad():
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        
        # Ensure correct projection to 512-d space (Transformers version handling)
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "pooler_output"):
                val = text_features.pooler_output
            elif hasattr(text_features, "last_hidden_state"):
                val = text_features.last_hidden_state[:, 0, :]
            else:
                val = text_features[0]
                
            if val.shape[-1] != 512 and hasattr(model, "text_projection") and model.text_projection is not None:
                val = model.text_projection(val)
            text_features = val
            
        # Enforce L2 normalization 
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_embedding = text_features.cpu().numpy().tolist()[0]
        
    # Query ChromaDB with explicit video_id filtering
    where_filter = {"video_id": video_id} 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"]
    )
    
    if not results["ids"] or not results["ids"][0]:
        return []
        
    valid_matches = []
    # Process results and convert distance metrics back to confidence scores
    for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
        similarity_score = 1.0 - (dist / 2.0)
        
        # Apply safety thresholding
        if similarity_score >= threshold:
            valid_matches.append({
                "frame_path": meta["frame_path"],
                "timestamp": meta["timestamp"],
                "similarity_score": similarity_score
            })
            
    return valid_matches