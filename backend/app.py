from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Check GPU on startup
print("GPU Available:", tf.config.list_physical_devices('GPU'))

UPLOAD_FOLDER = 'uploads'
FACES_FOLDER = 'faces'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_faces_from_video(video_path, max_frames=50):
    """Extract faces from video frames using OpenCV"""
    
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
            faces_extracted.append(face_resized)
        
        frame_count += 1
    
    cap.release()
    return faces_extracted

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Server running', 
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'tensorflow_version': tf.__version__
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Basic video info
        try:
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return jsonify({
                'message': 'Video uploaded successfully',
                'filename': filename,
                'duration': duration,
                'frames': frame_count,
                'resolution': f"{width}x{height}",
                'fps': fps
            })
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Supported: mp4, avi, mov, mkv, webm'}), 400

@app.route('/api/analyze/<filename>', methods=['POST'])
def analyze_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        # Extract faces from video
        faces = extract_faces_from_video(filepath, max_frames=100)
        
        # Basic video analysis
        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Mock deepfake analysis (replace with actual model later)
        faces_count = len(faces)
        
        # Simple heuristic for demo - more faces detected = higher chance of deepfake
        base_confidence = 0.6 + (faces_count / 100.0) * 0.3
        confidence = min(base_confidence + np.random.uniform(-0.1, 0.1), 0.95)
        
        # Classification based on confidence
        is_deepfake = confidence > 0.75
        prediction = 'deepfake' if is_deepfake else 'real'
        
        # Analysis details
        analysis_result = {
            'status': 'completed',
            'filename': filename,
            'duration': round(duration, 2),
            'total_frames': frame_count,
            'faces_detected': faces_count,
            'confidence': round(confidence, 3),
            'prediction': prediction,
            'risk_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'message': f'Analysis complete: {"Potential deepfake detected" if is_deepfake else "Video appears authentic"}',
            'processing_details': {
                'frames_analyzed': min(frame_count, 100),
                'face_regions_processed': faces_count,
                'detection_method': 'ResNet-Swish-BiLSTM (Mock)'
            }
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/results/<filename>', methods=['GET'])
def get_results(filename):
    # This endpoint can be used to retrieve previous analysis results
    return jsonify({'message': 'Results endpoint - to be implemented'})

if __name__ == '__main__':
    print("Starting Deepfake Detection Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Faces folder: {FACES_FOLDER}")
    app.run(debug=True, port=5000, host='0.0.0.0')
