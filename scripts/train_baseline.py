# Training script for baseline model
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import JerseyNumberDataset, collate_fn
from src.models.baseline import BaselineCNN


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for frames, labels, _ in pbar:
        # Move to device
        if isinstance(frames, list):
            frames = [f.to(device) for f in frames]
        else:
            frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for frames, labels, _ in pbar:
            # Move to device
            if isinstance(frames, list):
                frames = [f.to(device) for f in frames]
            else:
                frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--data-root", type=str, default="data/SoccerNet/jersey-2023",
                        help="Root directory of dataset")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Maximum frames per tracklet")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet34", "resnet50", "efficientnet_b0"],
                        help="CNN backbone")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers (use 0 on Windows)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (default: None)")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision for faster training")
    
    args = parser.parse_args()
    
    # Setup - Force CUDA initialization
    torch.cuda.init()  # Initialize CUDA
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"Using device: cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu")
        print("Warning: CUDA not available, training will be slow!")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = JerseyNumberDataset(
        data_root=args.data_root,
        split="train",
        max_frames=args.max_frames,
        sample_strategy="uniform"
    )
    
    test_dataset = JerseyNumberDataset(
        data_root=args.data_root,
        split="test",
        max_frames=args.max_frames,
        sample_strategy="uniform"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    print(f"\nCreating model with {args.backbone} backbone...")
    model = BaselineCNN(
        num_classes=101,  # 0-99 for jersey numbers, 100 for "not visible" (-1)
        backbone=args.backbone,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    results = []
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0)
            if Path(output_dir / "results.json").exists():
                with open(output_dir / "results.json", 'r') as f:
                    results = json.load(f)
            print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
        else:
            print(f"\nWarning: Checkpoint not found at {args.resume}, starting from scratch")
    
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        scheduler.step(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'best_acc': best_acc,
            }, output_dir / "best_model.pth")
            print(f"Saved new best model (acc: {best_acc:.2f}%)")
        
        # Save last checkpoint (for resuming)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'best_acc': best_acc,
        }, output_dir / "last_checkpoint.pth")
        
        # Save results
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
