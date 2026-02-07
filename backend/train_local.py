# train_local.py - VS Code Local Training
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datetime import datetime

# Print VS Code friendly output
print("🎬 Local ResNet-LSTM Training - VS Code Edition")
print("=" * 60)

class ResNetLSTMDetector(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, num_layers=2, dropout=0.4):
        super(ResNetLSTMDetector, self).__init__()
        
        print("🏗️ Building ResNet-LSTM for GTX 1650...")
        
        # ResNet50 backbone
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Freeze early layers for GTX 1650
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Model ready! Trainable parameters: {trainable_params:,}")
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Extract features
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        resnet_features = self.resnet(x_reshaped)
        features = resnet_features.view(batch_size, seq_len, -1)
        
        # LSTM temporal analysis
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        
        # Classification
        predictions = self.classifier(last_output)
        return predictions

class LocalVideoDataset(Dataset):
    def __init__(self, data_dir, split='train', sequence_length=20):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.samples = []
        
        # Load video paths
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov']:
                for video_path in real_dir.glob(ext):
                    self.samples.append((str(video_path), 0))
        
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov']:
                for video_path in fake_dir.glob(ext):
                    self.samples.append((str(video_path), 1))
        
        # Train/val split
        random.shuffle(self.samples)
        split_idx = int(0.8 * len(self.samples))
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"📊 {split.upper()}: {len(self.samples)} videos")
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = len(self.samples) - real_count
        print(f"   Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            frames = self.extract_frames(video_path)
            frames = self.preprocess_frames(frames)
            return frames, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"⚠️ Error with {video_path}: {e}")
            dummy_frames = torch.zeros(self.sequence_length, 3, 224, 224, dtype=torch.float32)
            return dummy_frames, torch.tensor(label, dtype=torch.long)
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"No frames: {video_path}")
        
        # Sample frames evenly
        if total_frames <= self.sequence_length:
            frame_indices = list(range(total_frames))
            while len(frame_indices) < self.sequence_length:
                frame_indices.append(total_frames - 1)
        else:
            step = total_frames / self.sequence_length
            frame_indices = [int(i * step) for i in range(self.sequence_length)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames[:self.sequence_length], dtype=np.float32)
    
    def preprocess_frames(self, frames):
        frames = frames / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames = (frames - mean) / std
        
        return torch.from_numpy(frames).permute(0, 3, 1, 2)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"📚 Epoch {epoch}")
    
    for videos, labels in progress_bar:
        videos = videos.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # VS Code friendly progress display
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100*correct/total:.1f}%'
        })
    
    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="🔍 Validating"):
            videos = videos.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

def main():
    # System check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Training Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected! Training will be slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Configuration
    config = {
        'data_path': r'C:\Users\gaura\Downloads\training_data',  # ⚠️ UPDATE THIS PATH!
        'batch_size': 2,
        'sequence_length': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'weight_decay': 1e-4
    }
    
    print(f"\n⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Dataset validation
    data_path = Path(config['data_path'])
    if not data_path.exists():
        print(f"\n❌ Dataset path not found: {config['data_path']}")
        print("Please update the 'data_path' in the config above")
        return
    
    real_count = len(list((data_path / 'real').glob('*.mp4'))) if (data_path / 'real').exists() else 0
    fake_count = len(list((data_path / 'fake').glob('*.mp4'))) if (data_path / 'fake').exists() else 0
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Real videos: {real_count}")
    print(f"   Fake videos: {fake_count}")
    print(f"   Total: {real_count + fake_count}")
    
    if real_count + fake_count < 10:
        print("❌ Not enough videos! Check your dataset path.")
        return
    
    # Create datasets and loaders
    print(f"\n📁 Creating datasets...")
    train_dataset = LocalVideoDataset(config['data_path'], split='train', sequence_length=config['sequence_length'])
    val_dataset = LocalVideoDataset(config['data_path'], split='val', sequence_length=config['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # Model setup
    model = ResNetLSTMDetector(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training variables
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🚀 STARTING TRAINING!")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'training_time': time.time() - start_time
            }, 'best_deepfake_model_vscode.pth')
            
            print(f"✅ NEW BEST MODEL! Accuracy: {val_acc:.2f}%")
        
        # Log results
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        
        # VS Code friendly output
        print(f"\n📊 EPOCH {epoch+1}/{config['num_epochs']} COMPLETE")
        print(f"   ⏱️  Epoch time: {epoch_time/60:.1f} minutes")
        print(f"   📈 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
        print(f"   🎯 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
        print(f"   🏆 Best: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"   ⌚ Total elapsed: {elapsed_total/3600:.2f} hours")
        print("-" * 50)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"⏱️ Total Time: {total_time/3600:.2f} hours")
    print(f"💾 Model saved: best_deepfake_model_vscode.pth")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save training plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best: {best_val_acc:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results_vscode.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎨 Training plots saved: training_results_vscode.png")

if __name__ == '__main__':
    main()
