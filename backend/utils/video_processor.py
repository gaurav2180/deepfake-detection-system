import cv2
import numpy as np
import torch
import os
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path: str, max_frames: int = 30, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract exactly max_frames from a video file
    
    Args:
        video_path: Path to the video file
        max_frames: Number of frames to extract (default: 30)
        target_size: Target frame size as (width, height) tuple
    
    Returns:
        numpy array of shape [max_frames, height, width, 3] with RGB frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")
    
    logger.info(f"📹 Extracting {max_frames} frames from: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    
    if total_frames == 0:
        raise ValueError("❌ Video has no frames")
    
    # Calculate frame sampling strategy
    if total_frames <= max_frames:
        # If video has fewer frames than needed, sample all and repeat last frame
        frame_indices = list(range(total_frames))
        # Pad with last frame index
        while len(frame_indices) < max_frames:
            frame_indices.append(total_frames - 1)
        logger.info(f"📝 Short video: using all {total_frames} frames + padding")
    else:
        # Sample frames evenly across the entire video
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
        logger.info(f"📝 Sampling every {step:.1f} frames")
    
    frames = []
    successful_reads = 0
    
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame to target size
            frame_resized = cv2.resize(frame, target_size)
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            successful_reads += 1
        else:
            # If frame reading fails, use the last successful frame or create black frame
            if frames:
                frames.append(frames[-1].copy())  # Repeat last frame
                logger.warning(f"⚠️  Failed to read frame {frame_idx}, using previous frame")
            else:
                # Create black frame as fallback
                black_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                frames.append(black_frame)
                logger.warning(f"⚠️  Failed to read frame {frame_idx}, using black frame")
    
    cap.release()
    
    logger.info(f"✅ Successfully extracted {successful_reads}/{len(frame_indices)} frames")
    
    # Ensure we have exactly max_frames
    while len(frames) < max_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    # Convert to numpy array and ensure correct shape
    frames_array = np.array(frames[:max_frames])  # [max_frames, height, width, 3]
    
    logger.info(f"📐 Final frame array shape: {frames_array.shape}")
    
    return frames_array

def extract_faces_from_video(video_path: str, max_frames: int = 30, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract face regions from video frames using OpenCV face detection
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of face frames to extract
        target_size: Target size for face images
    
    Returns:
        numpy array of face frames or empty array if no faces found
    """
    logger.info(f"👤 Extracting faces from: {video_path}")
    
    # Load face detection cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        logger.error("❌ Could not load face detection cascade")
        return np.array([])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return np.array([])
    
    face_frames = []
    frame_count = 0
    faces_found = 0
    
    while cap.isOpened() and len(face_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame for efficiency
        if frame_count % 3 == 0:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Take the largest face (most likely to be the main subject)
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Add padding around face
                padding = int(0.2 * max(w, h))
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                # Extract face region
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:  # Ensure face region is not empty
                    # Resize to target size
                    face_resized = cv2.resize(face_region, target_size)
                    # Convert BGR to RGB
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_frames.append(face_rgb)
                    faces_found += 1
        
        frame_count += 1
    
    cap.release()
    
    logger.info(f"👤 Found faces in {faces_found} frames out of {frame_count} processed")
    
    return np.array(face_frames) if face_frames else np.array([])

def validate_video_file(video_path: str) -> bool:
    """
    Validate if a video file is readable and has content
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if video is valid, False otherwise
    """
    try:
        if not os.path.exists(video_path):
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Check if video has frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return total_frames > 0
    
    except Exception as e:
        logger.error(f"❌ Video validation error: {e}")
        return False

def get_video_info(video_path: str) -> dict:
    """
    Get detailed information about a video file
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary containing video metadata
    """
    if not os.path.exists(video_path):
        return {"error": "File not found"}
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    
    # Extract video properties
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
    }
    
    # Calculate duration
    if info["fps"] > 0:
        info["duration_seconds"] = round(info["total_frames"] / info["fps"], 2)
    else:
        info["duration_seconds"] = 0
    
    cap.release()
    
    return info

def preprocess_video_for_model(video_path: str, max_frames: int = 30) -> torch.Tensor:
    """
    Complete preprocessing pipeline for a video file
    
    Args:
        video_path: Path to video file
        max_frames: Number of frames to extract
    
    Returns:
        Preprocessed tensor ready for model input
    """
    logger.info(f"🔄 Preprocessing video: {video_path}")
    
    # Step 1: Extract frames
    frames = extract_frames(video_path, max_frames=max_frames)
    
    # Step 2: Normalize and convert to tensor
    frames_normalized = frames.astype(np.float32) / 255.0
    
    # Step 3: Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frames_normalized = (frames_normalized - mean) / std
    
    # Step 4: Convert to PyTorch tensor
    # From [frames, height, width, channels] to [1, frames, channels, height, width]
    frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2).unsqueeze(0)
    
    logger.info(f"✅ Preprocessed tensor shape: {frames_tensor.shape}")
    
    return frames_tensor

# Test function
def test_video_processing():
    """
    Test video processing functions with a sample video
    """
    print("🧪 Testing video processing functions...")
    
    # This would test with an actual video file
    # test_video = "path/to/test/video.mp4"
    # if os.path.exists(test_video):
    #     frames = extract_frames(test_video)
    #     print(f"✅ Extracted frames shape: {frames.shape}")
    #     
    #     info = get_video_info(test_video)
    #     print(f"✅ Video info: {info}")
    
    print("✅ Video processing functions ready!")

if __name__ == "__main__":
    test_video_processing()
