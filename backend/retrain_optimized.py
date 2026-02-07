# retrain_optimized.py - OPTIMIZED training for REAL results
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datetime import datetime
from collections import Counter

print("=" * 70)
print("🔥 OPTIMIZED ResNet-LSTM RETRAINING")
print("🎯 Goal: Break the plateau and get REAL learning")
print("💪 Strategy: Balanced dataset + Higher LR + Smart training")
print("=" * 70)

class ResNetLSTMDetector(nn.Module):
    """Optimized ResNet-LSTM for GTX 1650"""
    
    def __init__(self, num_classes=2, hidden_size=128, num_layers=2, dropout=0.4):
        super(ResNetLSTMDetector, self).__init__()
        
        print("🏗️ Building OPTIMIZED ResNet-LSTM...")
        
        # ResNet50 backbone
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Freeze early layers for faster training
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Simplified classifier for better learning
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        print("✅ Optimized model built!")
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        resnet_features = self.resnet(x_reshaped)
        features = resnet_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        
        predictions = self.classifier(last_output)
        return predictions

class BalancedVideoDataset(Dataset):
    """BALANCED dataset - fixes the 5:1 ratio problem"""
    
    def __init__(self, data_dir, split='train', sequence_length=16, balance=True):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # Collect all videos
        real_samples = []
        fake_samples = []
        
        # Real videos
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov']:
                for video_path in real_dir.glob(ext):
                    real_samples.append((str(video_path), 0))
        
        # Fake videos
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov']:
                for video_path in fake_dir.glob(ext):
                    fake_samples.append((str(video_path), 1))
        
        print(f"\n📊 Original dataset:")
        print(f"   Real: {len(real_samples)}")
        print(f"   Fake: {len(fake_samples)}")
        
        # === BALANCE THE DATASET ===
        if balance:
            # Oversample minority class (real videos)
            min_count = min(len(real_samples), len(fake_samples))
            max_count = max(len(real_samples), len(fake_samples))
            
            target_count = max_count  # Balance to majority class size
            
            if len(real_samples) < len(fake_samples):
                # Duplicate real samples
                multiplier = target_count // len(real_samples) + 1
                real_samples = real_samples * multiplier
                real_samples = real_samples[:target_count]
                print(f"🔄 Oversampled real videos to {len(real_samples)}")
            else:
                # Duplicate fake samples
                multiplier = target_count // len(fake_samples) + 1
                fake_samples = fake_samples * multiplier
                fake_samples = fake_samples[:target_count]
                print(f"🔄 Oversampled fake videos to {len(fake_samples)}")
        
        # Combine and shuffle
        self.samples = real_samples + fake_samples
        random.shuffle(self.samples)
        
        # Train/val split
        split_idx = int(0.8 * len(self.samples))
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        # Count distribution
        label_counts = Counter([label for _, label in self.samples])
        
        print(f"\n✅ {split.upper()} dataset:")
        print(f"   Total: {len(self.samples)} videos")
        print(f"   Real: {label_counts[0]} ({label_counts[0]/len(self.samples)*100:.1f}%)")
        print(f"   Fake: {label_counts[1]} ({label_counts[1]/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            frames = self.extract_frames(video_path)
            frames = self.preprocess_frames(frames)
            return frames, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Return dummy on error
            dummy_frames = torch.zeros(self.sequence_length, 3, 224, 224, dtype=torch.float32)
            return dummy_frames, torch.tensor(label, dtype=torch.long)
    
    def extract_frames(self, video_path):
        """Extract 16 frames (reduced from 20 for speed)"""
        cap = cv2.VideoCapture(video_path)
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
        """Preprocess with data augmentation"""
        frames = frames / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames = (frames - mean) / std
        
        # Random augmentation (training only)
        if random.random() > 0.5:
            # Random brightness
            brightness_factor = random.uniform(0.8, 1.2)
            frames = frames * brightness_factor
        
        return torch.from_numpy(frames).permute(0, 3, 1, 2)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """Train one epoch with progress tracking"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"🔥 Epoch {epoch}")
    
    for batch_idx, (videos, labels) in enumerate(progress_bar):
        videos = videos.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler and hasattr(scheduler, 'step') and batch_idx % 10 == 0:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100*correct/total:.1f}%'
        })
    
    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    """Validate with detailed metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0, 0]
    class_total = [0, 0]
    
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
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Calculate accuracies
    overall_acc = 100 * correct / total
    real_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    fake_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    print(f"   📊 Per-class accuracy:")
    print(f"      Real: {real_acc:.1f}% ({class_correct[0]}/{class_total[0]})")
    print(f"      Fake: {fake_acc:.1f}% ({class_correct[1]}/{class_total[1]})")
    
    return val_loss / len(val_loader), overall_acc, real_acc, fake_acc

def main():
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Training Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    
    # === OPTIMIZED CONFIGURATION ===
    config = {
        'data_path': r'C:\Users\gaura\Downloads\training_data',  # UPDATE THIS!
        'batch_size': 4,  # Increased for better gradient estimates
        'sequence_length': 16,  # Reduced from 20 for speed
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 0.003,  # 3x HIGHER than before!
        'num_epochs': 15,  # Fewer but more effective epochs
        'weight_decay': 5e-5,  # Lower weight decay
        'balance_dataset': True,  # KEY FIX!
        'use_scheduler': True
    }
    
    print(f"\n⚙️ OPTIMIZED Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Verify dataset exists
    data_path = Path(config['data_path'])
    if not data_path.exists():
        print(f"\n❌ Dataset path not found: {config['data_path']}")
        print("Please update the 'data_path' in the config!")
        return
    
    # Create BALANCED datasets
    print(f"\n📁 Creating BALANCED datasets...")
    train_dataset = BalancedVideoDataset(
        config['data_path'], 
        split='train', 
        sequence_length=config['sequence_length'],
        balance=config['balance_dataset']
    )
    
    val_dataset = BalancedVideoDataset(
        config['data_path'], 
        split='val', 
        sequence_length=config['sequence_length'],
        balance=config['balance_dataset']
    )
    
    # Data loaders with more workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print(f"\n🏗️ Building model...")
    model = ResNetLSTMDetector(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # AGGRESSIVE SCHEDULER - key for breaking plateau!
    if config['use_scheduler']:
        total_steps = len(train_loader) * config['num_epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'] * 2,  # Peak at 2x base LR
            total_steps=total_steps,
            pct_start=0.3,  # Warmup for 30% of training
            anneal_strategy='cos'
        )
        print("✅ Using OneCycleLR scheduler (aggressive)")
    else:
        scheduler = None
    
    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    train_accs = []
    val_accs = []
    real_accs = []
    fake_accs = []
    
    print(f"\n🚀 STARTING OPTIMIZED TRAINING!")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    # TRAINING LOOP
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📚 EPOCH {epoch+1}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch+1, scheduler
        )
        
        # Validate
        val_loss, val_acc, real_acc, fake_acc = validate(model, val_loader, criterion, device)
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'training_time': time.time() - start_time,
                'real_acc': real_acc,
                'fake_acc': fake_acc
            }, 'best_deepfake_model_optimized.pth')
            
            print(f"\n🎉 NEW BEST MODEL! Accuracy: {val_acc:.2f}% (+{improvement:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        
        # Epoch summary
        print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
        print(f"   ⏱️  Time: {epoch_time/60:.1f} min")
        print(f"   📈 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
        print(f"   🎯 Val:   Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
        print(f"   🏆 Best:  {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"   📚 LR:    {current_lr:.6f}")
        print(f"   ⌚ Total: {elapsed_total/3600:.2f} hours")
        
        # Check for improvement
        if epoch > 5 and val_acc <= best_val_acc - 5:
            print(f"   ⚠️  Warning: Accuracy dropped significantly!")
        elif epoch > 0 and abs(val_acc - val_accs[-2]) < 0.5:
            print(f"   ⚠️  Warning: Little improvement this epoch")
    
    total_time = time.time() - start_time
    
    # TRAINING COMPLETE
    print(f"\n{'='*70}")
    print(f"🎉 OPTIMIZED TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"🏆 Best Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
    print(f"💾 Model saved: best_deepfake_model_optimized.pth")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Plot comprehensive results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses, 'b-', label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training vs Validation accuracy
    axes[0, 1].plot(train_accs, 'b-', label='Train Accuracy')
    axes[0, 1].plot(val_accs, 'g-', label='Val Accuracy')
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.1f}%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Per-class accuracy
    axes[1, 0].plot(real_accs, 'b-', marker='o', label='Real Video Accuracy')
    axes[1, 0].plot(fake_accs, 'r-', marker='s', label='Fake Video Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Per-Class Accuracy (CRITICAL)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate schedule
    axes[1, 1].text(0.5, 0.5, f'FINAL RESULTS:\n\n'
                              f'Best Validation: {best_val_acc:.2f}%\n'
                              f'Real Accuracy: {real_accs[best_epoch-1]:.2f}%\n'
                              f'Fake Accuracy: {fake_accs[best_epoch-1]:.2f}%\n'
                              f'Training Time: {total_time/3600:.2f}h\n\n'
                              f'Dataset: BALANCED\n'
                              f'Epochs: {config["num_epochs"]}\n'
                              f'Learning Rate: {config["learning_rate"]}',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎨 Training plots saved: training_results_optimized.png")
    print(f"\n🏆 YOUR OPTIMIZED MODEL IS READY!")
    print(f"📊 Real videos accuracy: {real_accs[best_epoch-1]:.2f}%")
    print(f"🎭 Fake videos accuracy: {fake_accs[best_epoch-1]:.2f}%")

if __name__ == '__main__':
    main()
