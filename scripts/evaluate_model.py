"""
Evaluate a trained model on a full split and report accuracy.
"""
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import JerseyNumberDataset, collate_fn
from src.models.baseline import BaselineCNN


def load_model(checkpoint_path: str, backbone: str, device: torch.device) -> BaselineCNN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = BaselineCNN(
        num_classes=101,
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(model: BaselineCNN, dataloader: DataLoader, device: torch.device, aggregate: str) -> float:
    """Evaluate model accuracy on a dataloader."""
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for frames, labels, _ in pbar:
            if isinstance(frames, list):
                frames = [f.to(device) for f in frames]
            else:
                frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames, aggregate=aggregate)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if total > 0:
                pbar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})

    return 100 * correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained jersey number model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/SoccerNet/jersey-2023",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet34", "resnet50", "efficientnet_b0"],
        help="Backbone used during training",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Max frames per tracklet (match training)",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "max", "voting"],
        help="Aggregation method for frame logits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader workers (use 0 on Windows)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.backbone, device)

    dataset = JerseyNumberDataset(
        data_root=args.data_root,
        split=args.split,
        max_frames=args.max_frames,
        sample_strategy="uniform",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    accuracy = evaluate(model, dataloader, device, args.aggregate)
    print(f"\nAccuracy on {args.split} split: {accuracy:.2f}%")
    print(f"Total samples: {len(dataset)}")


if __name__ == "__main__":
    main()
