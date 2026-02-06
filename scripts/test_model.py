"""
Inference script for testing trained model on tracklets
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import JerseyNumberDataset
from src.models.baseline import BaselineCNN


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = BaselineCNN(
        num_classes=101,
        backbone="resnet50",  # Adjust if you used a different backbone
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
    
    return model


def predict_tracklet(model, frames, device):
    """
    Predict jersey number for a tracklet.
    
    Args:
        model: Trained model
        frames: Tensor of shape (num_frames, C, H, W)
        device: Device to run on
    
    Returns:
        predicted_number: Predicted jersey number
        confidence: Confidence score
        all_probs: Probability distribution
    """
    with torch.no_grad():
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = probs.max(dim=1)
        
        prediction = prediction.item()
        confidence = confidence.item()
        
        # Map class 100 back to -1 (not visible)
        if prediction == 100:
            prediction = -1
        
        return prediction, confidence, probs.cpu().numpy()[0]


def visualize_tracklet(frames_tensor, prediction, confidence, ground_truth=None, save_path=None):
    """Visualize a few frames from the tracklet with prediction."""
    # Denormalize frames
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_frames = min(5, frames_tensor.shape[0])
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
    
    if num_frames == 1:
        axes = [axes]
    
    # Select frames evenly across the sequence
    indices = np.linspace(0, frames_tensor.shape[0] - 1, num_frames, dtype=int)
    
    for i, idx in enumerate(indices):
        frame = frames_tensor[idx].permute(1, 2, 0).numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f'Frame {idx}')
    
    # Add prediction info
    title = f'Prediction: {prediction} (Confidence: {confidence*100:.1f}%)'
    if ground_truth is not None:
        title += f'\nGround Truth: {ground_truth}'
        if prediction == ground_truth:
            title += ' ✓'
        else:
            title += ' ✗'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_dataset_samples(model, dataset, device, num_samples=10, save_dir=None):
    """Test model on random samples from dataset."""
    print(f"\nTesting on {num_samples} random samples...")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    correct = 0
    results = []
    
    for i, idx in enumerate(indices):
        frames, label, player_id = dataset[idx]
        
        # Predict
        prediction, confidence, probs = predict_tracklet(model, frames, device)
        
        # Map label 100 back to -1
        if label == 100:
            label = -1
        
        is_correct = (prediction == label)
        correct += int(is_correct)
        
        result = {
            'player_id': player_id,
            'prediction': int(prediction),
            'ground_truth': int(label),
            'confidence': float(confidence),
            'correct': is_correct
        }
        results.append(result)
        
        print(f"\nSample {i+1}/{num_samples} - Player {player_id}")
        print(f"  Prediction: {prediction}")
        print(f"  Ground Truth: {label}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        # Visualize
        if save_dir:
            save_path = save_dir / f"sample_{i+1}_{player_id}.png"
            visualize_tracklet(frames, prediction, confidence, label, save_path)
    
    accuracy = 100 * correct / num_samples
    print(f"\nAccuracy on {num_samples} samples: {accuracy:.2f}%")
    
    return results


def test_single_tracklet(model, data_root, split, player_id, device):
    """Test model on a specific tracklet."""
    dataset = JerseyNumberDataset(
        data_root=data_root,
        split=split,
        max_frames=None,  # Use all frames
        sample_strategy="uniform"
    )
    
    # Find the player
    try:
        idx = dataset.player_ids.index(player_id)
    except ValueError:
        print(f"Player {player_id} not found in {split} split")
        return
    
    frames, label, player_id = dataset[idx]
    
    # Predict
    prediction, confidence, probs = predict_tracklet(model, frames, device)
    
    # Map label back
    if label == 100:
        label = -1
    
    print(f"\nPlayer ID: {player_id}")
    print(f"Number of frames: {frames.shape[0]}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Ground Truth: {label}")
    print(f"Result: {'✓ Correct' if prediction == label else '✗ Incorrect'}")
    
    # Show top-5 predictions
    top5_indices = np.argsort(probs)[-5:][::-1]
    print("\nTop 5 predictions:")
    for rank, idx in enumerate(top5_indices, 1):
        jersey_num = idx if idx < 100 else -1
        print(f"  {rank}. Jersey #{jersey_num}: {probs[idx]*100:.2f}%")
    
    # Visualize
    visualize_tracklet(frames, prediction, confidence, label)


def main():
    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="data/SoccerNet/jersey-2023",
                        help="Root directory of dataset")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Which split to test on")
    parser.add_argument("--player-id", type=str, default=None,
                        help="Specific player ID to test (optional)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of random samples to test")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    if args.player_id:
        # Test specific tracklet
        test_single_tracklet(model, args.data_root, args.split, args.player_id, device)
    else:
        # Test random samples
        dataset = JerseyNumberDataset(
            data_root=args.data_root,
            split=args.split,
            max_frames=5,  # Use same as training
            sample_strategy="uniform"
        )
        
        results = test_dataset_samples(
            model, dataset, device, 
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
        
        # Save results
        if args.save_dir:
            save_path = Path(args.save_dir) / "test_results.json"
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {save_path}")


if __name__ == "__main__":
    main()
