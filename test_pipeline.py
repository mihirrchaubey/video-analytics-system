import os
import cv2
import numpy as np
import shutil
import hashlib

# Perform cleanup first, before importing app modules, to avoid file locking on Windows
def run_tests():
    # Setup test workspace cleanups
    video_storage_path = "./storage/videos"
    frame_storage_path = "./storage/frames"
    chroma_db_path = "./storage/chroma"
    database_url = "sqlite:///./storage/metadata.db"

    for path in [video_storage_path, frame_storage_path, chroma_db_path]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not remove directory {path}: {e}")
        os.makedirs(path, exist_ok=True)
        
    # Remove existing db if sqlite
    db_file = database_url.replace("sqlite:///", "")
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception as e:
            print(f"Warning: Could not remove database file {db_file}: {e}")

    # Now import app modules after cleanup is done
    from app.config import settings
    from app.database import init_db, VideoMetadata, SessionLocal
    from app.video_processor import process_video, query_frames
    
    # Initialize databases
    init_db()
    
    # Create a dummy video in storage
    test_video_path = os.path.join(settings.video_storage_path, "raw_test_video.mp4")
    create_dummy_video(test_video_path, duration_sec=5, fps=10)
    
    # Compute hash
    def compute_file_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    file_hash = compute_file_hash(test_video_path)
    video_id = f"vid_{file_hash[:8]}"
    final_video_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    os.rename(test_video_path, final_video_path)
    
    print(f"Processing video: {video_id}")
    db = SessionLocal()
    try:
        new_video = VideoMetadata(filename=video_id, file_hash=file_hash)
        db.add(new_video)
        db.commit()
        
        frames_extracted = process_video(final_video_path, video_id)
        print(f"Successfully processed video. Extracted {frames_extracted} frames.")
        
        # Verify db entry
        video_record = db.query(VideoMetadata).filter(VideoMetadata.file_hash == file_hash).first()
        assert video_record is not None, "Database record was not created"
        assert video_record.filename == video_id, "Filename mismatch in DB"
        print("Database verification passed.")
        
        # Verify frames files exist
        frame_files = os.listdir(settings.frame_storage_path)
        print(f"Saved frames in storage: {frame_files}")
        assert len(frame_files) > 0, "No frames were saved to disk"
        
        # Let's query
        print("\nQuerying semantic search for 'a red circle'...")
        results = query_frames("a red circle", threshold=0.1)
        print("Query Results:")
        for idx, res in enumerate(results):
            print(f" {idx+1}. Path: {res['frame_path']}, Timestamp: {res['timestamp']}, Score: {res['similarity_score']}")
            
        assert len(results) > 0, "Semantic search query returned 0 matches"
        
        print("\nQuerying semantic search for 'a blue square'...")
        results_blue = query_frames("a blue square", threshold=0.1)
        print("Query Results:")
        for idx, res in enumerate(results_blue):
            print(f" {idx+1}. Path: {res['frame_path']}, Timestamp: {res['timestamp']}, Score: {res['similarity_score']}")
            
        print("\nAll pipeline tests passed successfully!")
    finally:
        db.close()

def create_dummy_video(filename: str, duration_sec: int = 3, fps: int = 10):
    """Generates a dummy video with a blue background and a moving red circle."""
    print(f"Generating dummy video at: {filename}")
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame_idx in range(duration_sec * fps):
        # Create a blue background (OpenCV uses BGR)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [255, 0, 0]  # Blue background
        
        # Draw a red circle moving from left to right
        center_x = int((frame_idx / (duration_sec * fps)) * width)
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)  # Red circle
        
        out.write(frame)
        
    out.release()
    print("Dummy video generated successfully.")

if __name__ == "__main__":
    run_tests()
