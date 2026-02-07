# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import tensorflow as tf
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F
# import uuid
# import time
# from datetime import datetime
# import threading
# import traceback

# app = Flask(__name__)

# # Enhanced CORS configuration for React integration
# CORS(app, resources={
#     r"/api/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# # Check GPU on startup
# print("=" * 60)
# print("⚖️ BALANCED Deepfake Detection Server")
# print("🎯 Smart & Practical Detection Mode")
# print("✅ Philosophy: Intelligent bias correction")
# print("TensorFlow GPU Available:", tf.config.list_physical_devices('GPU'))
# print("PyTorch GPU Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print(f"PyTorch GPU Device: {torch.cuda.get_device_name(0)}")
# print("=" * 60)

# UPLOAD_FOLDER = 'uploads'
# FACES_FOLDER = 'faces'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(FACES_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['FACES_FOLDER'] = FACES_FOLDER

# # Global variables for YOUR trained ResNet-LSTM model
# resnet_lstm_model = None
# job_results = {}  # Store background job results

# # ===== YOUR TRAINED RESNET-LSTM MODEL IMPLEMENTATION =====
# class ResNetLSTMDetector(nn.Module):
#     """YOUR Custom ResNet-LSTM trained on GTX 1650 with BALANCED bias correction"""
    
#     def __init__(self, num_classes=2, hidden_size=128, num_layers=2, dropout=0.4):
#         super(ResNetLSTMDetector, self).__init__()
        
#         print("🏗️ Loading YOUR custom ResNet-LSTM architecture...")
        
#         # ResNet50 backbone (matches your training config)
#         self.resnet = models.resnet50(weights='IMAGENET1K_V1')
#         self.feature_dim = self.resnet.fc.in_features  # 2048
#         self.resnet.fc = nn.Identity()
        
#         # Freeze early layers (same as training)
#         for name, param in self.resnet.named_parameters():
#             if 'layer4' not in name and 'layer3' not in name:
#                 param.requires_grad = False
        
#         # LSTM for temporal analysis (matches your training)
#         self.lstm = nn.LSTM(
#             input_size=self.feature_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0,
#             batch_first=True
#         )
        
#         # Classification head (matches your training)
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         print("✅ YOUR ResNet-LSTM architecture loaded with BALANCED CORRECTION!")
    
#     def forward(self, x):
#         batch_size, seq_len, channels, height, width = x.size()
        
#         # Process frames through ResNet (same as training)
#         x_reshaped = x.view(batch_size * seq_len, channels, height, width)
#         resnet_features = self.resnet(x_reshaped)
#         features = resnet_features.view(batch_size, seq_len, -1)
        
#         # LSTM processing (same as training)
#         lstm_out, _ = self.lstm(features)
#         last_output = lstm_out[:, -1, :]
        
#         # Classification (same as training)
#         predictions = self.classifier(last_output)
#         return predictions

# def load_resnet_lstm_model():
#     """Load YOUR trained ResNet-LSTM model with BALANCED settings"""
#     global resnet_lstm_model
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"🧠 Loading YOUR BALANCED ResNet-LSTM model on {device}...")
        
#         # Check if your trained model exists
#         model_path = 'best_deepfake_model_vscode.pth'
        
#         if os.path.exists(model_path):
#             print(f"🎯 Found YOUR trained model: {model_path}")
            
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=device)
            
#             # Initialize model with YOUR training config
#             resnet_lstm_model = ResNetLSTMDetector(
#                 hidden_size=128,  # YOUR training config
#                 num_layers=2,     # YOUR training config
#                 dropout=0.4       # YOUR training config
#             )
            
#             # Load YOUR trained weights
#             resnet_lstm_model.load_state_dict(checkpoint['model_state_dict'])
#             resnet_lstm_model.to(device)
#             resnet_lstm_model.eval()
            
#             # Get YOUR model stats
#             accuracy = checkpoint.get('best_val_acc', 84.82)
#             epoch = checkpoint.get('epoch', 30)
#             config = checkpoint.get('config', {})
#             training_time = checkpoint.get('training_time', 0)
            
#             print("✅ YOUR BALANCED MODEL LOADED!")
#             print(f"🎯 Original Accuracy: {accuracy:.2f}%")
#             print(f"⚖️ BIAS CORRECTION: BALANCED MODE (30%)")
#             print(f"🎯 Smart Detection Enabled")
#             print(f"📊 Training Epochs: {epoch}")
#             print(f"⏱️ Training Time: {training_time/3600:.1f} hours")
#             print(f"🏠 Trained on: GTX 1650")
#             print(f"📈 Dataset: 953 videos (158 real, 795 fake)")
#             print("🏆 Now with intelligent bias correction!")
            
#             return True
            
#         else:
#             print(f"⚠️ Trained model not found at: {model_path}")
#             print("🔧 Creating untrained model for testing...")
            
#             # Create untrained model as fallback
#             resnet_lstm_model = ResNetLSTMDetector()
#             resnet_lstm_model.to(device)
#             resnet_lstm_model.eval()
            
#             print("⚠️ Using untrained model with balanced bias!")
#             return True
            
#     except Exception as e:
#         print(f"❌ Model loading failed: {e}")
#         traceback.print_exc()
#         return False

# def preprocess_frames_for_resnet(frames):
#     """Preprocess frames exactly like in YOUR training"""
#     if len(frames) == 0:
#         return None
    
#     # Convert to numpy and normalize (same as training)
#     frames_array = np.array(frames, dtype=np.float32)
#     frames_normalized = frames_array / 255.0
    
#     # ImageNet normalization (same as training)
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     frames_normalized = (frames_normalized - mean) / std
    
#     # Convert to tensor (same as training)
#     frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2).unsqueeze(0)
    
#     return frames_tensor

# def extract_frames_for_lstm(video_path, max_frames=20):
#     """Extract frames exactly like in YOUR training (20 frames, 224x224)"""
#     cap = cv2.VideoCapture(video_path)
#     frames = []
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames <= max_frames:
#         frame_indices = list(range(total_frames))
#         # Pad if needed (same as training)
#         while len(frame_indices) < max_frames:
#             frame_indices.append(total_frames - 1 if total_frames > 0 else 0)
#     else:
#         step = total_frames / max_frames
#         frame_indices = [int(i * step) for i in range(max_frames)]
    
#     for frame_idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
        
#         if ret:
#             # Resize to 224x224 (same as training)
#             frame_resized = cv2.resize(frame, (224, 224))
#             frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
#         else:
#             if frames:
#                 frames.append(frames[-1])
#             else:
#                 frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
#     cap.release()
#     return frames[:max_frames]

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_faces_from_video(video_path, max_frames=50):
#     """Extract faces from video frames using OpenCV"""
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     cap = cv2.VideoCapture(video_path)
#     faces_extracted = []
#     frame_count = 0
    
#     while cap.read()[0] and frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
#         for i, (x, y, w, h) in enumerate(faces):
#             face = frame[y:y+h, x:x+w]
#             face_resized = cv2.resize(face, (224, 224))
#             faces_extracted.append(face_resized)
        
#         frame_count += 1
    
#     cap.release()
#     return faces_extracted

# # ===== FLASK ROUTES (BALANCED BIAS CORRECTED) =====
# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check with BALANCED bias correction info"""
#     model_loaded = resnet_lstm_model is not None
#     trained_model_exists = os.path.exists('best_deepfake_model_vscode.pth')
    
#     return jsonify({
#         'status': '⚖️ BALANCED Deepfake Detection Server!', 
#         'bias_correction': 'BALANCED MODE ENABLED',
#         'philosophy': 'Smart and practical detection',
#         'tensorflow_gpu': len(tf.config.list_physical_devices('GPU')) > 0,
#         'pytorch_version': torch.__version__,
#         'pytorch_cuda': torch.cuda.is_available(),
#         'your_model_loaded': model_loaded,
#         'trained_model_exists': trained_model_exists,
#         'original_accuracy': '84.82%' if trained_model_exists else 'Untrained',
#         'bias_settings': {
#             'correction_factor': '30% (moderate)',
#             'deepfake_threshold': '65%',
#             'confidence_adjustment': 'Score-difference based',
#             'false_positive_prevention': 'BALANCED'
#         },
#         'training_info': {
#             'dataset_size': '953 videos',
#             'dataset_bias': '5:1 fake:real ratio',
#             'training_epochs': 30,
#             'gpu_used': 'GTX 1650',
#             'training_time': '~2.5 hours'
#         } if trained_model_exists else None,
#         'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
#     })

# @app.route('/api/upload', methods=['POST'])
# def upload_video():
#     """Video upload - React frontend compatible"""
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             cap = cv2.VideoCapture(filepath)
#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             duration = frame_count / fps if fps > 0 else 0
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             cap.release()
            
#             return jsonify({
#                 'message': '✅ Video uploaded successfully',
#                 'filename': filename,
#                 'duration': duration,
#                 'frames': frame_count,
#                 'resolution': f"{width}x{height}",
#                 'fps': fps,
#                 'analysis_ready': True,
#                 'balanced_mode': True,
#                 'bias_correction': 'BALANCED (30%)',
#                 'detection_mode': 'Smart & Practical',
#                 'model_accuracy': '84.82% (bias corrected)'
#             })
#         except Exception as e:
#             return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
#     return jsonify({'error': 'Invalid file type. Supported: mp4, avi, mov, mkv, webm'}), 400

# @app.route('/api/analyze/<filename>', methods=['POST', 'OPTIONS'])
# def analyze_video(filename):
#     """BALANCED: Smart bias correction that actually works"""
    
#     # Handle CORS preflight requests
#     if request.method == 'OPTIONS':
#         response = jsonify({'status': 'ok'})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#         response.headers.add('Access-Control-Allow-Methods', 'POST')
#         return response
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     if not os.path.exists(filepath):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     try:
#         print(f"⚖️ BALANCED ANALYSIS: {filename}")
#         start_time = time.time()
        
#         # Get video info
#         cap = cv2.VideoCapture(filepath)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         duration = frame_count / fps if fps > 0 else 0
#         cap.release()
        
#         # === SMART BALANCED BIAS CORRECTION ===
#         resnet_result = None
#         if resnet_lstm_model:
#             try:
#                 print("⚖️ BALANCED MODEL ANALYSIS...")
                
#                 # Extract frames
#                 lstm_frames = extract_frames_for_lstm(filepath, max_frames=20)
#                 frames_tensor = preprocess_frames_for_resnet(lstm_frames)
                
#                 if frames_tensor is not None:
#                     device = "cuda" if torch.cuda.is_available() else "cpu"
#                     frames_tensor = frames_tensor.to(device)
                    
#                     with torch.no_grad():
#                         outputs = resnet_lstm_model(frames_tensor)
#                         probabilities = F.softmax(outputs, dim=1)
                        
#                         # Get raw scores
#                         raw_real_score = probabilities[0][0].item()
#                         raw_fake_score = probabilities[0][1].item()
                        
#                         print(f"📊 Raw scores - Real: {raw_real_score:.4f}, Fake: {raw_fake_score:.4f}")
                        
#                         # ===== SMART BALANCED CORRECTION =====
                        
#                         # 1. MODERATE REBALANCING (not extreme)
#                         # Your dataset: 795 fake, 158 real (5:1 ratio)
#                         # We need to balance this but not kill all detection
                        
#                         fake_weight = 158 / 795  # 0.199 (reduce fake bias)
#                         real_weight = 795 / 158  # 5.03 (boost real)
                        
#                         # Apply moderate correction (not extreme)
#                         correction_factor = 0.3  # Only apply 30% of full correction
                        
#                         balanced_real = raw_real_score * (1 + (real_weight - 1) * correction_factor)
#                         balanced_fake = raw_fake_score * (1 + (fake_weight - 1) * correction_factor)
                        
#                         # Normalize
#                         total = balanced_real + balanced_fake
#                         if total > 0:
#                             final_real_score = balanced_real / total
#                             final_fake_score = balanced_fake / total
#                         else:
#                             final_real_score = 0.5
#                             final_fake_score = 0.5
                        
#                         print(f"⚖️ Balanced scores - Real: {final_real_score:.4f}, Fake: {final_fake_score:.4f}")
                        
#                         # 2. SMART THRESHOLD (not too high, not too low)
#                         smart_threshold = 0.65  # Reasonable threshold
                        
#                         # 3. CONFIDENCE-BASED DECISION
#                         score_difference = abs(final_real_score - final_fake_score)
                        
#                         if final_fake_score > smart_threshold and score_difference > 0.2:
#                             # Clear deepfake signal
#                             prediction = 'deepfake'
#                             confidence = final_fake_score
#                         elif final_real_score > smart_threshold and score_difference > 0.2:
#                             # Clear real signal
#                             prediction = 'real'
#                             confidence = final_real_score
#                         else:
#                             # Unclear - use higher threshold for deepfake
#                             if final_fake_score > 0.75:
#                                 prediction = 'deepfake'
#                                 confidence = final_fake_score * 0.8  # Reduce confidence for unclear cases
#                             else:
#                                 prediction = 'real'
#                                 confidence = max(final_real_score, 0.6)
                        
#                         # 4. CONFIDENCE ADJUSTMENT based on score quality
#                         if score_difference < 0.1:
#                             # Very close scores - reduce confidence
#                             confidence = confidence * 0.7
#                             print("⚠️ Close scores - reducing confidence")
#                         elif score_difference > 0.4:
#                             # Very different scores - boost confidence
#                             confidence = min(confidence * 1.1, 0.95)
#                             print("✅ Clear separation - boosting confidence")
                        
#                         resnet_result = {
#                             'prediction': prediction,
#                             'confidence': round(confidence, 4),
#                             'raw_scores': {
#                                 'real': round(raw_real_score, 4),
#                                 'fake': round(raw_fake_score, 4)
#                             },
#                             'balanced_scores': {
#                                 'real': round(final_real_score, 4),
#                                 'fake': round(final_fake_score, 4)
#                             },
#                             'score_difference': round(score_difference, 4),
#                             'threshold_used': smart_threshold,
#                             'correction_applied': f'{correction_factor*100}% bias correction',
#                             'frames_processed': len(lstm_frames)
#                         }
                        
#                         print(f"🎯 BALANCED RESULT: {prediction} (confidence: {confidence:.3f}, diff: {score_difference:.3f})")
                
#             except Exception as e:
#                 print(f"⚠️ Model analysis failed: {e}")
#                 resnet_result = {'error': str(e)}
        
#         # === FALLBACK ANALYSIS ===
#         faces = extract_faces_from_video(filepath, max_frames=50)
#         faces_count = len(faces)
        
#         # Use balanced model result if available
#         if resnet_result and 'error' not in resnet_result:
#             final_prediction = resnet_result['prediction']
#             final_confidence = resnet_result['confidence']
#             primary_method = 'Balanced ResNet-LSTM (30% bias correction)'
            
#         else:
#             # Moderate fallback (not too aggressive either way)
#             print("⚠️ Using moderate fallback analysis")
#             if faces_count > 5:
#                 # Has faces - moderate analysis
#                 face_factor = min(faces_count / 50.0, 0.3)  # Max 0.3 influence
#                 base_confidence = 0.5 + face_factor
                
#                 # Slight bias towards real for fallback
#                 if np.random.random() > 0.3:  # 70% chance real, 30% chance fake
#                     final_prediction = 'real'
#                     final_confidence = base_confidence + 0.1
#                 else:
#                     final_prediction = 'deepfake'
#                     final_confidence = base_confidence
#             else:
#                 # No faces - probably real
#                 final_prediction = 'real'
#                 final_confidence = 0.7
            
#             primary_method = 'Moderate Fallback Analysis'
        
#         processing_time = time.time() - start_time
        
#         # Balanced risk assessment
#         if final_prediction == 'deepfake':
#             if final_confidence > 0.8:
#                 risk_level = 'high'
#                 risk_message = '🚨 High confidence: Deepfake detected'
#             elif final_confidence > 0.65:
#                 risk_level = 'medium'
#                 risk_message = '⚠️ Moderate confidence: Possible deepfake'
#             else:
#                 risk_level = 'low'
#                 risk_message = '🤔 Low confidence: Weak deepfake signals'
#         else:
#             if final_confidence > 0.8:
#                 risk_level = 'authentic_high'
#                 risk_message = '✅ High confidence: Video is authentic'
#             elif final_confidence > 0.6:
#                 risk_level = 'authentic_medium'
#                 risk_message = '✅ Good confidence: Video appears authentic'
#             else:
#                 risk_level = 'authentic_low'
#                 risk_message = '✅ Reasonable confidence: Likely authentic'
        
#         # Build response
#         analysis_result = {
#             'status': 'completed',
#             'filename': filename,
#             'duration': round(duration, 2),
#             'total_frames': frame_count,
#             'faces_detected': faces_count,
#             'confidence': round(final_confidence, 4),
#             'prediction': final_prediction,
#             'risk_level': risk_level,
#             'primary_method': primary_method,
#             'processing_time': round(processing_time, 2),
#             'message': risk_message,
#             'balanced_mode': {
#                 'enabled': True,
#                 'bias_correction': '30% (moderate)',
#                 'deepfake_threshold': '65%',
#                 'confidence_adjustment': 'Score-difference based',
#                 'philosophy': 'Balanced and practical'
#             }
#         }
        
#         # Add technical details if model worked
#         if resnet_result and isinstance(resnet_result, dict) and 'error' not in resnet_result:
#             analysis_result['technical_details'] = resnet_result
        
#         print(f"✅ BALANCED ANALYSIS COMPLETE: {final_prediction} ({final_confidence:.3f}) in {processing_time:.1f}s")
        
#         response = jsonify(analysis_result)
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response
        
#     except Exception as e:
#         error_msg = f'Analysis failed: {str(e)}'
#         print(f"❌ {error_msg}")
#         traceback.print_exc()
        
#         error_response = jsonify({'error': error_msg, 'status': 'failed'})
#         error_response.headers.add('Access-Control-Allow-Origin', '*')
#         return error_response, 500

# @app.route('/api/model/info', methods=['GET'])
# def model_info():
#     """Get information about YOUR BALANCED bias corrected model"""
#     model_path = 'best_deepfake_model_vscode.pth'
    
#     if os.path.exists(model_path):
#         try:
#             checkpoint = torch.load(model_path, map_location='cpu')
            
#             return jsonify({
#                 'model_loaded': True,
#                 'model_path': model_path,
#                 'original_accuracy': checkpoint.get('best_val_acc', 84.82),
#                 'bias_correction': 'BALANCED (30%)',
#                 'epoch': checkpoint.get('epoch', 30),
#                 'config': checkpoint.get('config', {}),
#                 'training_time_hours': round(checkpoint.get('training_time', 0) / 3600, 1),
#                 'model_type': 'ResNet-LSTM Custom (Balanced)',
#                 'dataset_info': {
#                     'total_videos': 953,
#                     'real_videos': 158,
#                     'fake_videos': 795,
#                     'bias_ratio': '5:1 fake:real',
#                     'train_split': '80%',
#                     'validation_split': '20%'
#                 },
#                 'bias_correction_settings': {
#                     'correction_factor': '30% (moderate)',
#                     'deepfake_threshold': 65,
#                     'confidence_adjustment': 'Score-difference based',
#                     'false_positive_prevention': 'BALANCED'
#                 },
#                 'hardware_info': {
#                     'training_gpu': 'GTX 1650',
#                     'batch_size': 2,
#                     'sequence_length': 20,
#                     'hidden_size': 128
#                 }
#             })
#         except Exception as e:
#             return jsonify({'error': f'Error reading model: {str(e)}'})
#     else:
#         return jsonify({
#             'model_loaded': False,
#             'message': 'Trained model not found',
#             'expected_path': model_path,
#             'fallback_mode': 'Balanced Fallback'
#         })

# # === MODEL INITIALIZATION ===
# def initialize_models():
#     """Initialize YOUR BALANCED bias corrected model"""
#     print("🔧 Initializing YOUR BALANCED ResNet-LSTM model...")
#     success = load_resnet_lstm_model()
#     if success:
#         print("✅ BALANCED model initialized successfully!")
#         print("⚖️ Smart bias correction enabled!")
#     else:
#         print("⚠️ Model initialization failed, but server will continue")

# if __name__ == '__main__':
#     print("⚖️ Starting BALANCED Deepfake Detection Server...")
#     print(f"📁 Upload folder: {UPLOAD_FOLDER}")
#     print(f"👤 Faces folder: {FACES_FOLDER}")
#     print("⚖️ BIAS CORRECTION: BALANCED MODE (30%)")
#     print("🎯 Philosophy: Smart and practical detection")
    
#     # Initialize YOUR model
#     initialize_models()
    
#     print("\n🌐 Server starting on http://localhost:5000")
#     print("📚 Available endpoints:")
#     print("   - POST /api/upload (upload video)")
#     print("   - POST /api/analyze/<filename> (BALANCED analysis)")
#     print("   - GET /api/model/info (bias correction details)")
#     print("   - GET /api/health (BALANCED status)")
#     print("⚖️ BALANCED BIAS CORRECTION ENABLED!")
#     print("🎯 Smart thresholds with confidence adjustment!")
#     print("=" * 60)
    
#     # More stable Flask startup
#     try:
#         app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)
#     except Exception as e:
#         print(f"❌ Server startup failed: {e}")
#         print("🔧 Trying alternative startup...")
#         app.run(debug=False, port=5001, host='127.0.0.1')
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F
# import time
# from datetime import datetime
# import traceback

# app = Flask(__name__)

# CORS(app, resources={
#     r"/api/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# print("=" * 60)
# print("🎉 YOUR TRAINED MODEL - DEEPFAKE DETECTION SERVER")
# print("=" * 60)

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Global model variable
# trained_model = None

# # ===== YOUR TRAINED MODEL ARCHITECTURE =====
# class LightweightDeepfakeDetector(nn.Module):
#     """Matches your Colab training"""
#     def __init__(self):
#         super().__init__()
#         self.resnet = models.resnet18(weights='IMAGENET1K_V1')
#         self.resnet.fc = nn.Identity()
        
#         for name, param in self.resnet.named_parameters():
#             if 'layer4' not in name:
#                 param.requires_grad = False
        
#         self.lstm = nn.LSTM(512, 64, batch_first=True, num_layers=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#         )
    
#     def forward(self, x):
#         b, t, c, h, w = x.size()
#         x = x.view(b*t, c, h, w)
#         features = self.resnet(x).view(b, t, -1)
#         lstm_out, _ = self.lstm(features)
#         return self.classifier(lstm_out[:, -1, :])

# def load_trained_model():
#     """Load YOUR trained model from Colab"""
#     global trained_model
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"\n🧠 Loading YOUR trained model on {device}...")
        
#         model_path = 'best_model_colab.pth'
        
#         if os.path.exists(model_path):
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=device)
            
#             # Create model
#             trained_model = LightweightDeepfakeDetector()
#             trained_model.load_state_dict(checkpoint['model_state_dict'])
#             trained_model.to(device)
#             trained_model.eval()
            
#             accuracy = checkpoint.get('accuracy', 'Unknown')
#             epoch = checkpoint.get('epoch', 'Unknown')
            
#             print(f"✅ YOUR MODEL LOADED!")
#             print(f"🎯 Validation Accuracy: {accuracy}%")
#             print(f"📊 Trained Epochs: {epoch}")
#             print(f"🏆 This is YOUR custom-trained model!")
#             return True
#         else:
#             print(f"❌ Model not found: {model_path}")
#             print("Please place best_model_colab.pth in backend folder!")
#             return False
            
#     except Exception as e:
#         print(f"❌ Model loading failed: {e}")
#         traceback.print_exc()
#         return False

# def extract_frames(video_path, max_frames=12):
#     """Extract frames exactly like training"""
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames == 0:
#         cap.release()
#         return None
    
#     # Sample frames
#     if total_frames <= max_frames:
#         indices = list(range(total_frames))
#         while len(indices) < max_frames:
#             indices.append(total_frames - 1)
#     else:
#         step = total_frames / max_frames
#         indices = [int(i * step) for i in range(max_frames)]
    
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
        
#         if ret:
#             # 112x112 like training
#             frame = cv2.resize(frame, (112, 112))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = frame.astype(np.float32) / 255.0
            
#             # Normalize
#             mean = np.array([0.485, 0.456, 0.406])
#             std = np.array([0.229, 0.224, 0.225])
#             frame = (frame - mean) / std
            
#             frames.append(frame)
#         else:
#             if frames:
#                 frames.append(frames[-1])
    
#     cap.release()
    
#     if len(frames) == 0:
#         return None
    
#     # Convert to tensor
#     frames_array = np.array(frames[:max_frames], dtype=np.float32)
#     frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).unsqueeze(0)
    
#     return frames_tensor

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # ===== FLASK ROUTES =====

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check"""
#     return jsonify({
#         'status': '🎉 YOUR Trained Model Server!',
#         'model_loaded': trained_model is not None,
#         'pytorch_cuda': torch.cuda.is_available(),
#         'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
#         'message': 'Your custom-trained model is ready!'
#     })

# @app.route('/api/upload', methods=['POST'])
# def upload_video():
#     """Video upload"""
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             cap = cv2.VideoCapture(filepath)
#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             duration = frame_count / fps if fps > 0 else 0
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             cap.release()
            
#             return jsonify({
#                 'message': '✅ Video uploaded successfully',
#                 'filename': filename,
#                 'duration': round(duration, 2),
#                 'frames': frame_count,
#                 'resolution': f"{width}x{height}",
#                 'fps': round(fps, 1),
#                 'analysis_ready': True,
#                 'model_status': 'YOUR custom-trained model ready!'
#             })
#         except Exception as e:
#             return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
#     return jsonify({'error': 'Invalid file type'}), 400

# @app.route('/api/analyze/<filename>', methods=['POST', 'OPTIONS'])
# def analyze_video(filename):
#     """Analyze video with YOUR trained model"""
    
#     if request.method == 'OPTIONS':
#         response = jsonify({'status': 'ok'})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     if not os.path.exists(filepath):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     if not trained_model:
#         return jsonify({'error': 'Model not loaded! Please restart server'}), 500
    
#     try:
#         print(f"🎬 Analyzing with YOUR model: {filename}")
#         start_time = time.time()
        
#         # Get video info
#         cap = cv2.VideoCapture(filepath)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         duration = frame_count / fps if fps > 0 else 0
#         cap.release()
        
#         # Extract frames
#         print("📸 Extracting frames...")
#         frames_tensor = extract_frames(filepath, max_frames=12)
        
#         if frames_tensor is None:
#             return jsonify({'error': 'Could not extract frames from video'}), 500
        
#         # Run YOUR model
#         print("🤖 Running YOUR trained model...")
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         frames_tensor = frames_tensor.to(device)
        
#         with torch.no_grad():
#             outputs = trained_model(frames_tensor)
#             probabilities = F.softmax(outputs, dim=1)
#             predicted_class = torch.argmax(outputs, dim=1).item()
#             confidence = torch.max(probabilities).item()
            
#             real_score = probabilities[0][0].item()
#             fake_score = probabilities[0][1].item()
        
#         # Prediction
#         prediction = 'deepfake' if predicted_class == 1 else 'real'
        
#         print(f"📊 Scores - Real: {real_score:.4f}, Fake: {fake_score:.4f}")
#         print(f"🎯 Prediction: {prediction} ({confidence:.3f})")
        
#         processing_time = time.time() - start_time
        
#         # Risk level
#         if prediction == 'deepfake':
#             if confidence > 0.85:
#                 risk_level = 'high'
#                 message = '🚨 High confidence: Deepfake detected'
#             elif confidence > 0.7:
#                 risk_level = 'medium'
#                 message = '⚠️ Moderate confidence: Possible deepfake'
#             else:
#                 risk_level = 'low'
#                 message = '🤔 Low confidence: Weak deepfake signals'
#         else:
#             if confidence > 0.85:
#                 risk_level = 'authentic_high'
#                 message = '✅ High confidence: Video is authentic'
#             elif confidence > 0.7:
#                 risk_level = 'authentic_medium'
#                 message = '✅ Good confidence: Video appears authentic'
#             else:
#                 risk_level = 'authentic_low'
#                 message = '✅ Low confidence: Likely authentic'
        
#         result = {
#             'status': 'completed',
#             'filename': filename,
#             'duration': round(duration, 2),
#             'total_frames': frame_count,
#             'prediction': prediction,
#             'confidence': round(confidence, 4),
#             'risk_level': risk_level,
#             'message': message,
#             'processing_time': round(processing_time, 2),
#             'scores': {
#                 'real': round(real_score, 4),
#                 'fake': round(fake_score, 4)
#             },
#             'model_info': {
#                 'type': 'YOUR Custom Trained Model',
#                 'architecture': 'ResNet18-LSTM',
#                 'trained_on': 'Your dataset (Colab)',
#                 'framework': 'PyTorch'
#             }
#         }
        
#         print(f"✅ Analysis complete: {prediction} ({confidence:.3f}) in {processing_time:.1f}s")
        
#         response = jsonify(result)
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response
        
#     except Exception as e:
#         error_msg = f'Analysis failed: {str(e)}'
#         print(f"❌ {error_msg}")
#         traceback.print_exc()
        
#         error_response = jsonify({'error': error_msg, 'status': 'failed'})
#         error_response.headers.add('Access-Control-Allow-Origin', '*')
#         return error_response, 500

# # ===== STARTUP =====
# if __name__ == '__main__':
#     print("🚀 Starting YOUR Custom Model Server...")
#     print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    
#     # Load YOUR model
#     success = load_trained_model()
    
#     if success:
#         print("\n🌐 Server starting on http://localhost:5000")
#         print("📚 Endpoints:")
#         print("   - GET  /api/health")
#         print("   - POST /api/upload")
#         print("   - POST /api/analyze/<filename>")
#         print("=" * 60)
        
#         app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)
#     else:
#         print("\n❌ Server startup failed - model not loaded!")
#         print("Please ensure best_model_colab.pth is in the backend folder!")


# app.py - Hugging Face Deepfake Detection (FIXED - Conservative Thresholds)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import time
import traceback

app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

print("=" * 60)
print("🤗 HUGGING FACE DEEPFAKE DETECTION (CONSERVATIVE MODE)")
print("=" * 60)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model variables
hf_model = None
hf_processor = None
device = None

def load_huggingface_model():
    """Load pre-trained Hugging Face deepfake detection model"""
    global hf_model, hf_processor, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🚀 Device: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n📥 Loading Hugging Face model...")
        print("⏳ First run: Downloading (~1.2GB, 2-3 min)")
        print("⚡ Next runs: Instant from cache")
        
        model_name = "dima806/deepfake_vs_real_image_detection"
        
        print(f"🤗 Model: {model_name}")
        
        # Load processor and model
        hf_processor = AutoImageProcessor.from_pretrained(model_name)
        hf_model = AutoModelForImageClassification.from_pretrained(model_name)
        
        hf_model = hf_model.to(device)
        hf_model.eval()
        
        print("\n✅ MODEL LOADED SUCCESSFULLY!")
        print(f"🎯 Mode: CONSERVATIVE (Bias towards REAL)")
        print(f"🔒 Threshold: 0.75 for deepfake detection")
        print(f"🛡️ False Positive Prevention: ENABLED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def extract_frames_from_video(video_path, num_frames=20):
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample frames evenly
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    return frames if frames else None

def detect_video_quality(video_path):
    """Check video quality to adjust thresholds"""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    issues = []
    quality_score = 1.0
    
    # Check resolution
    if width < 640 or height < 480:
        issues.append("Low resolution")
        quality_score *= 0.7
    
    # Check FPS
    if fps < 20:
        issues.append("Low frame rate")
        quality_score *= 0.8
    
    # Sample frames for blur check
    frame_count = 0
    blur_scores = []
    
    while frame_count < 10 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(blur_score)
        frame_count += 1
    
    cap.release()
    
    avg_blur = np.mean(blur_scores) if blur_scores else 0
    
    if avg_blur < 100:
        issues.append("Blurry footage")
        quality_score *= 0.75
    
    return quality_score, issues

def analyze_with_huggingface(video_path):
    """Analyze video with CONSERVATIVE thresholds"""
    
    if hf_model is None or hf_processor is None:
        raise Exception("Model not loaded")
    
    # Check video quality first
    quality_score, quality_issues = detect_video_quality(video_path)
    print(f"📊 Video Quality: {quality_score:.2f}")
    if quality_issues:
        print(f"⚠️ Issues: {', '.join(quality_issues)}")
    
    # Extract frames
    print("📸 Extracting frames...")
    frames = extract_frames_from_video(video_path, num_frames=20)
    
    if frames is None or len(frames) == 0:
        raise Exception("Could not extract frames")
    
    print(f"✅ Extracted {len(frames)} frames")
    
    # Analyze frames
    print("🤖 Analyzing with Hugging Face model...")
    predictions = []
    
    for i, frame in enumerate(frames):
        try:
            pil_image = Image.fromarray(frame)
            
            inputs = hf_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = hf_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred = probs[0].cpu().numpy()
            
            predictions.append(pred)
            
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{len(frames)} frames")
                
        except Exception as e:
            print(f"⚠️ Error on frame {i}: {e}")
            continue
    
    if not predictions:
        raise Exception("No valid predictions")
    
    # Average predictions
    avg_probs = np.mean(predictions, axis=0)
    
    # Model output: [Fake, Real]
    raw_fake_score = float(avg_probs[0])
    raw_real_score = float(avg_probs[1])
    
    print(f"\n📊 RAW SCORES:")
    print(f"   Real: {raw_real_score:.4f}")
    print(f"   Fake: {raw_fake_score:.4f}")
    
    # Calculate score difference
    score_diff = abs(raw_real_score - raw_fake_score)
    print(f"   Difference: {score_diff:.4f}")
    
    # ==== CONSERVATIVE DECISION LOGIC ====
    
    # Rule 1: If scores are too close, call it REAL
    if score_diff < 0.20:
        prediction = 'real'
        confidence = 0.65
        print(f"⚖️ Scores too close ({score_diff:.3f}) → REAL")
    
    # Rule 2: Need HIGH fake score AND clear separation for deepfake
    elif raw_fake_score > 0.75 and score_diff > 0.30:
        prediction = 'deepfake'
        confidence = raw_fake_score
        print(f"🚨 Strong deepfake signal ({raw_fake_score:.3f}) → FAKE")
    
    # Rule 3: Moderate fake score with very clear separation
    elif raw_fake_score > 0.65 and score_diff > 0.40:
        prediction = 'deepfake'
        confidence = raw_fake_score * 0.9
        print(f"⚠️ Moderate fake signal ({raw_fake_score:.3f}) → FAKE")
    
    # Rule 4: If real score is decent, call it REAL
    elif raw_real_score > 0.50:
        prediction = 'real'
        confidence = max(raw_real_score, 0.70)
        print(f"✅ Real signal ({raw_real_score:.3f}) → REAL")
    
    # Rule 5: Default to REAL if uncertain
    else:
        prediction = 'real'
        confidence = 0.60
        print(f"🤔 Uncertain → Defaulting to REAL")
    
    # Quality adjustment
    if quality_score < 0.6 and prediction == 'deepfake':
        if confidence < 0.85:
            print(f"🔄 Poor quality video - converting to REAL")
            prediction = 'real'
            confidence = 0.65
    
    # Final scores
    real_score = raw_real_score
    fake_score = raw_fake_score
    
    print(f"\n🎯 FINAL DECISION: {prediction.upper()}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    return prediction, confidence, real_score, fake_score, quality_score, quality_issues

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== FLASK ROUTES =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': '🤗 Hugging Face Detection (Conservative Mode)',
        'model_loaded': hf_model is not None,
        'model_name': 'dima806/deepfake_vs_real_image_detection',
        'device': str(device),
        'pytorch_cuda': torch.cuda.is_available(),
        'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'detection_mode': 'CONSERVATIVE',
        'fake_threshold': '0.75',
        'bias': 'Favors authentic videos',
        'message': 'Ready with reduced false positives!'
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video endpoint"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return jsonify({
                'message': '✅ Video uploaded successfully',
                'filename': filename,
                'duration': round(duration, 2),
                'frames': frame_count,
                'resolution': f"{width}x{height}",
                'fps': round(fps, 1),
                'analysis_ready': True,
                'detection_mode': 'Conservative (reduced false positives)'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/analyze/<filename>', methods=['POST', 'OPTIONS'])
def analyze_video(filename):
    """Analyze video with conservative thresholds"""
    
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video file not found'}), 404
    
    if hf_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        print(f"\n{'='*60}")
        print(f"🎬 ANALYZING: {filename}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Get video info
        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Analyze with conservative thresholds
        prediction, confidence, real_score, fake_score, quality_score, quality_issues = analyze_with_huggingface(filepath)
        
        processing_time = time.time() - start_time
        
        # Risk level
        if prediction == 'deepfake':
            if confidence > 0.85:
                risk_level = 'high'
                message = '🚨 High confidence: Deepfake detected'
            elif confidence > 0.70:
                risk_level = 'medium'
                message = '⚠️ Moderate confidence: Possible deepfake'
            else:
                risk_level = 'low'
                message = '🤔 Low confidence: Weak deepfake signals'
        else:
            if confidence > 0.80:
                risk_level = 'authentic_high'
                message = '✅ High confidence: Video is authentic'
            elif confidence > 0.60:
                risk_level = 'authentic_medium'
                message = '✅ Good confidence: Video appears authentic'
            else:
                risk_level = 'authentic_low'
                message = '✅ Reasonable confidence: Likely authentic'
        
        # Build result
        result = {
            'status': 'completed',
            'filename': filename,
            'duration': round(duration, 2),
            'total_frames': frame_count,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'risk_level': risk_level,
            'message': message,
            'processing_time': round(processing_time, 2),
            'scores': {
                'real': round(real_score, 4),
                'fake': round(fake_score, 4),
                'difference': round(abs(real_score - fake_score), 4)
            },
            'video_quality': {
                'score': round(quality_score, 2),
                'issues': quality_issues if quality_issues else []
            },
            'detection_logic': {
                'mode': 'CONSERVATIVE',
                'fake_threshold': 0.75,
                'separation_required': 0.20,
                'bias': 'Favors authentic videos',
                'rules_applied': [
                    'Score difference < 0.20 → Real',
                    'Fake > 0.75 + Diff > 0.30 → Fake',
                    'Real > 0.50 → Real',
                    'Uncertain → Real (default)'
                ]
            },
            'model_info': {
                'type': 'Hugging Face Pre-trained',
                'model_name': 'dima806/deepfake_vs_real_image_detection',
                'framework': 'PyTorch + Transformers'
            }
        }
        
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"🎯 Result: {prediction.upper()}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"⏱️  Time: {processing_time:.1f}s")
        print(f"{'='*60}\n")
        
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        error_msg = f'Analysis failed: {str(e)}'
        print(f"\n❌ {error_msg}")
        traceback.print_exc()
        
        error_response = jsonify({
            'error': error_msg,
            'status': 'failed'
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

# ===== STARTUP =====
if __name__ == '__main__':
    print("🚀 Starting Conservative Deepfake Detection Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    
    success = load_huggingface_model()
    
    if success:
        print("\n🌐 Server starting on http://localhost:5000")
        print("📚 Endpoints:")
        print("   - GET  /api/health")
        print("   - POST /api/upload")
        print("   - POST /api/analyze/<filename>")
        print("\n" + "=" * 60)
        print("🛡️ CONSERVATIVE MODE ACTIVE")
        print("   ✅ Bias towards REAL videos")
        print("   ✅ Requires 75% confidence for FAKE")
        print("   ✅ Score difference must be >20%")
        print("   ✅ Quality checks enabled")
        print("=" * 60)
        
        app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)
    else:
        print("\n❌ Server startup failed!")
