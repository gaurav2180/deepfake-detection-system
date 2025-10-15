import cv2
import numpy as np
import os

def extract_faces_from_video(video_path, output_dir, max_frames=100):
    """Extract faces from video frames using OpenCV"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    faces_extracted = []
    frame_count = 0
    
    while cap.read()[0] and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face = frame[y:y+h, x:x+w]
            
            # Resize to model input size (224x224)
            face_resized = cv2.resize(face, (224, 224))
            
            # Save face image
            face_filename = f"face_frame_{frame_count}_{i}.jpg"
            face_path = os.path.join(output_dir, face_filename)
            cv2.imwrite(face_path, face_resized)
            
            faces_extracted.append(face_path)
        
        frame_count += 1
    
    cap.release()
    return faces_extracted

if __name__ == "__main__":
    # Test function
    print("Face detection utility ready!")
