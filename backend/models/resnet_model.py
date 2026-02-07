import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os

class ResNetLSTMDetector(nn.Module):
    """
    ResNet-LSTM Deepfake Detection Model
    
    Architecture:
    1. ResNet50 for spatial feature extraction from video frames
    2. LSTM for temporal sequence analysis across frames
    3. Classification head for real vs fake prediction
    """
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, dropout=0.4):
        super(ResNetLSTMDetector, self).__init__()
        
        print("🏗️  Initializing ResNet-LSTM Detector...")
        
        # Phase 1: ResNet50 Backbone for Spatial Features
        self.resnet = models.resnet50(pretrained=True)
        print("✅ Loaded pretrained ResNet50")
        
        # Get ResNet feature dimension (2048 for ResNet50)
        self.feature_dim = self.resnet.fc.in_features
        
        # Remove final classification layer - we only want features
        self.resnet.fc = nn.Identity()
        
        # Freeze early ResNet layers for faster training (optional)
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        
        # Phase 2: LSTM for Temporal Analysis
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,    # 2048 from ResNet50
            hidden_size=hidden_size,        # 256 hidden units
            num_layers=num_layers,          # 2 LSTM layers
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,               # (batch, seq, feature) format
            bidirectional=False             # Unidirectional LSTM
        )
        
        # Phase 3: Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)  # 2 classes: real(0) vs fake(1)
        )
        
        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        print(f"✅ LSTM layers: {num_layers}, Hidden size: {hidden_size}")
        print(f"✅ Classification head: {num_classes} classes")
        print("🎯 ResNet-LSTM model initialized successfully!")
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channels, height, width]
               Example: [1, 30, 3, 224, 224] - 1 video with 30 frames
        
        Returns:
            predictions: Tensor of shape [batch_size, num_classes]
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Debug print
        # print(f"📥 Input shape: {x.shape}")
        
        # Reshape to process all frames through ResNet
        # Combine batch and sequence dimensions: [batch*seq, channels, height, width]
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Phase 1: Extract spatial features using ResNet
        with torch.set_grad_enabled(self.training):
            resnet_features = self.resnet(x_reshaped)  # [batch*seq, 2048]
        
        # Reshape back to sequence format: [batch, seq, features]
        features = resnet_features.view(batch_size, seq_len, -1)
        
        # Phase 2: Process temporal sequence with LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(features)
        
        # Use the last LSTM output for classification
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]  # Take last timestep: [batch_size, hidden_size]
        
        # Phase 3: Final classification
        predictions = self.classifier(last_output)
        
        return predictions
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            probabilities: Softmax probabilities for each class
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities

def load_model(model_path=None, device=None):
    """
    Load ResNet-LSTM model with optional pretrained weights
    
    Args:
        model_path: Path to saved model weights (optional)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded ResNet-LSTM model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 Loading model on device: {device}")
    
    # Initialize model
    model = ResNetLSTMDetector(
        num_classes=2,
        hidden_size=256,
        num_layers=2,
        dropout=0.4
    )
    
    # Load pretrained weights if available
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded trained weights from: {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("🔄 Using pretrained ResNet + untrained LSTM")
    else:
        print("🔄 Using pretrained ResNet50 + untrained LSTM layers")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_frames(frames):
    """
    Preprocess video frames for model input
    
    Args:
        frames: numpy array of shape [num_frames, height, width, channels]
                Values should be in range [0, 255]
    
    Returns:
        tensor: PyTorch tensor of shape [1, num_frames, 3, 224, 224]
                Normalized and ready for model input
    """
    if len(frames) == 0:
        print("⚠️  No frames provided, returning zero tensor")
        return torch.zeros((1, 1, 3, 224, 224))
    
    # Convert to float and normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # ImageNet normalization (required for pretrained ResNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Apply normalization
    frames_normalized = (frames - mean) / std
    
    # Convert to PyTorch tensor and rearrange dimensions
    # From [frames, height, width, channels] to [frames, channels, height, width]
    frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2)
    
    # Add batch dimension: [1, frames, channels, height, width]
    frames_tensor = frames_tensor.unsqueeze(0)
    
    print(f"📐 Preprocessed tensor shape: {frames_tensor.shape}")
    
    return frames_tensor

def get_model_info():
    """
    Get information about the model architecture
    
    Returns:
        dict: Model information
    """
    return {
        "architecture": "ResNet50 + LSTM",
        "spatial_extractor": "ResNet50 (pretrained on ImageNet)",
        "temporal_analyzer": "LSTM (2 layers, 256 hidden units)",
        "input_format": "[batch, 30_frames, 3_channels, 224_height, 224_width]",
        "output_classes": ["real", "fake"],
        "expected_accuracy": "90-95% (with proper training)"
    }

# Test function for debugging
def test_model():
    """
    Test the model with dummy data
    """
    print("🧪 Testing ResNet-LSTM model...")
    
    # Create dummy video data: 1 video, 30 frames, 3 channels, 224x224
    dummy_input = torch.randn(1, 30, 3, 224, 224)
    
    # Initialize model
    model = ResNetLSTMDetector()
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
    
    print(f"✅ Model output shape: {output.shape}")
    print(f"✅ Probabilities: {probabilities}")
    print("🎉 Model test successful!")

if __name__ == "__main__":
    test_model()
