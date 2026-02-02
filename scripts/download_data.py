# Script to download the SoccerNet Jersey Number Recognition dataset
from pathlib import Path
import argparse
from typing import Optional


def download_dataset(local_directory: str, splits: Optional[list] = None):
    """
    Download the SoccerNet Jersey Number Recognition dataset.
    
    Args:
        local_directory: Where to save the dataset
        splits: List of splits to download ('train', 'test', 'challenge')
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader as SNdl
    except ImportError:
        print("Error: SoccerNet package not installed.")
        print("Please install it with: pip install SoccerNet")
        return
    
    if splits is None:
        splits = ["train", "test", "challenge"]
    
    print(f"Downloading dataset to: {local_directory}")
    print(f"Splits: {splits}")
    
    # Create downloader
    downloader = SNdl(LocalDirectory=local_directory)
    
    # Download the data
    downloader.downloadDataTask(task="jersey-2023", split=splits)
    
    print("\nDownload complete!")
    print(f"\nDataset structure:")
    
    data_path = Path(local_directory)
    for split in splits:
        split_path = data_path / split
        if split_path.exists():
            image_dir = split_path / "images"
            if image_dir.exists():
                n_tracklets = len(list(image_dir.iterdir()))
                print(f"  {split}/: {n_tracklets} tracklets")


def main():
    parser = argparse.ArgumentParser(
        description="Download SoccerNet Jersey Number Recognition dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/SoccerNet",
        help="Directory to save the dataset (default: data/SoccerNet)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "challenge"],
        choices=["train", "test", "challenge"],
        help="Which splits to download (default: all)"
    )
    
    args = parser.parse_args()
    
    download_dataset(args.data_root, args.splits)


if __name__ == "__main__":
    main()
