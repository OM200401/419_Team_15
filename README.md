# COSC 419/519 Team 15 - Jersey Number Recognition

Deep Learning Project for COSC 419/519 - Winter 2026

## Project Overview

This project tackles the SoccerNet Jersey Number Recognition Challenge, where we identify jersey numbers from short video tracklets of soccer players. The challenge involves processing low-resolution, motion-blurred thumbnails where jersey numbers may only be visible in a small subset of frames.

### Task Description

Given short video tracklets of soccer players (typically a few hundred frames long), our goal is to identify the jersey number of each player. Players with non-visible jersey numbers are annotated with the value -1. The main challenges include:

- Low-resolution thumbnails
- High motion blur
- Jersey numbers visible in only a small subset of frames
- Variable lighting and camera angles

### Dataset

The SoccerNet Jersey Number dataset consists of:
- 2,853 player tracklets for training/testing
- 1,211 separate player tracklets for the challenge set (hidden annotations)
- Data derived from SoccerNet tracking videos
- JSON ground truth files mapping player IDs to jersey numbers

## Team Members

- Bridgette Hunt
- Kelvin Chen
- Om Mistry
- Karim Jassani
- Milan Bertolutti

## Project Links

- Challenge Page: https://www.soccer-net.org/tasks/jersey-number-recognition
- EvalAI Challenge: https://eval.ai/web/challenges/challenge-page/1952/overview
- Development Kit: https://github.com/SoccerNet/sn-jersey
- Research Paper: https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf
- Alternative Approach: https://arxiv.org/abs/2309.06285

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OM200401/419_Team_15.git
cd 419_Team_15
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="data/SoccerNet")
mySNdl.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
```

## Quick Commands (Current)

Download dataset with the script:

```bash
python scripts/download_data.py --data-root data/SoccerNet
```

Train baseline model:

```bash
python scripts/train_baseline.py \
  --data-root data/SoccerNet \
  --backbone resnet50 \
  --batch-size 8 \
  --epochs 50 \
  --max-frames 10 \
  --output-dir outputs/baseline
```

Test random samples:

```bash
python scripts/test_model.py  --checkpoint outputs/best_model.pth --data-root data/SoccerNet/jersey-2023 --split test --num-samples 10
```

Test a single tracklet:

```bash
python scripts/test_model.py \
  --checkpoint outputs/best_model.pth \
  --data-root data/SoccerNet/jersey-2023 \
  --split test \
  --player-id <PLAYER_ID>
```

Evaluate full-split accuracy:

```bash
python scripts/evaluate_model.py \
  --checkpoint outputs/best_model.pth \
  --data-root data/SoccerNet/jersey-2023 \
  --split test \
  --max-frames 10 \
  --aggregate mean
```

## Dataset Format

### Directory Structure
```
dataset/
├── train/
│   ├── groundtruth.json
│   └── images/
│       ├── player_id_1/
│       │   ├── frame_001.jpg
│       │   ├── frame_002.jpg
│       │   └── ...
│       └── player_id_2/
│           └── ...
├── test/
│   ├── groundtruth.json
│   └── images/
└── challenge/
    └── images/
```

### Ground Truth Format
JSON file mapping player IDs (strings) to jersey numbers (integers):
```json
{
  "player_id_1": 10,
  "player_id_2": 7,
  "player_id_3": -1
}
```
Note: -1 indicates the jersey number is not visible.

## Submission Format

For EvalAI submissions, create a JSON file with the same format as the ground truth:
```json
{
  "player_id": jersey_number
}
```

## Neural Network Implementation

We are implementing a multi-phase approach based on the research paper "Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos" by Balaji et al. (University of Waterloo - SARG UWaterloo team, 73.77% accuracy).

### Architecture Overview

#### Phase 1: Baseline CNN with Frame Averaging
Current Implementation: src/models/baseline.py

- Backbone: ResNet-50 (pre-trained on ImageNet)
- Approach: Process each frame independently through CNN, then aggregate predictions
- Aggregation Methods:
  - Mean pooling across frame logits
  - Max pooling for highest confidence frames
  - Soft voting with averaged probabilities
- Expected Accuracy: 45-55%

Key Components:
```python
class BaselineCNN(nn.Module):
    - backbone: ResNet-50/34 or EfficientNet-B0
    - classifier: FC layers (2048 -> 512 -> 100)
    - forward(): Processes variable-length frame sequences
```

#### Phase 2: Keyframe Selection Module
Status: Planned

The paper's key insight is that jersey numbers are only visible in a small subset of frames. We will implement:

- Blur Detection: Laplacian variance to identify sharp frames
- Scale Scoring: Prefer frames where player occupies more pixels
- Orientation Detection: Identify frames showing player's back
- Top-K Selection: Select k best frames per tracklet (k=5-10)

Expected Improvement: +15-20% accuracy -> ~65-70%

#### Phase 3: Temporal Modeling
Status: Planned

Add temporal context to capture motion patterns:

Option A - LSTM/GRU:
- Process selected keyframes in sequence
- Bidirectional LSTM to capture temporal dependencies
- Hidden state: 512 dimensions

Option B - 3D CNN:
- SlowFast networks or X3D
- Direct spatio-temporal feature extraction
- More computationally intensive

Option C - Temporal Attention:
- Self-attention over frame features
- Learn which frames are most informative
- Transformer-based aggregation

Expected Improvement: +5-8% accuracy -> ~70-75%

#### Phase 4: Multi-Task Learning
Status: Planned

Separate prediction heads for each digit:

- Tens Digit Head: Predicts 0-9 (for numbers 10-99)
- Units Digit Head: Predicts 0-9
- Combined Loss: Weighted sum of both digit losses
- Handles: Single-digit numbers (0-9) and two-digit (10-99)

Benefits:
- Better gradient flow for each digit
- Handles digit confusion separately
- Improves accuracy on similar-looking numbers (e.g., 18 vs 13)

Expected Improvement: +2-3% accuracy -> ~73-75%

### Training Strategy

1. Data Augmentation:
   - Random rotation (plus/minus 15 degrees)
   - Color jitter (brightness, contrast, saturation)
   - Random erasing (simulate occlusions)
   - Gaussian blur (simulate motion blur)

2. Optimization:
   - Adam optimizer with learning rate 1e-4
   - ReduceLROnPlateau scheduler (patience=5)
   - Batch size: 8 tracklets
   - Frame sampling: 10 frames per tracklet (uniform spacing)

3. Loss Function:
   - Cross-entropy for classification
   - Class weighting for imbalanced jersey numbers
   - Special handling for -1 (not visible) class

### Quick Start Training

Download dataset:
```bash
python scripts/download_data.py --data-root data/SoccerNet
```

Train baseline model:
```bash
python scripts/train_baseline.py \
    --data-root data/SoccerNet \
    --backbone resnet50 \
    --batch-size 8 \
    --epochs 50 \
    --max-frames 10 \
    --output-dir outputs/baseline
```

### Expected Results by Phase

| Phase | Approach | Target Accuracy |
|-------|----------|----------------|
| 0 | Random Baseline | ~3.9% |
| 1 | CNN + Frame Averaging | 45-55% |
| 2 | + Keyframe Selection | 65-70% |
| 3 | + Temporal Modeling | 70-75% |
| 4 | + Multi-Task Learning | 73-75% |

### Project Structure

```
├── src/
│   ├── data/
│   │   └── dataset.py              # JerseyNumberDataset loader
│   └── models/
│       └── baseline.py             # Baseline CNN implementation
├── scripts/
│   ├── download_data.py            # Dataset download utility
│   ├── train_baseline.py           # Training script
│   ├── test_model.py               # Inference/testing script
│   └── evaluate_model.py           # Full-split accuracy eval
├── notebooks/                      # Jupyter notebooks for exploration
├── outputs/                        # Model checkpoints and results
└── data/                          # Dataset (gitignored)
```

## Reference Leaderboard (2023 Challenge)

| Team | Accuracy |
|------|----------|
| ZZPM | 92.85% |
| UniBw Munich - VIS | 90.95% |
| zzzzz | 88.08% |
| MT-IOT | 81.7% |
| Baseline (Random) | 3.93% |

## Citation

If you use this code or the SoccerNet dataset, please cite:

```bibtex
@inproceedings{Somers2024SoccerNetGameState,
  title = {{SoccerNet} Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
  author = {Somers, Vladimir and Joos, Victor and Giancola, Silvio and Cioppa, Anthony and Ghasemzadeh, Seyed Abolfazl and Magera, Floriane and Standaert, Baptiste and Mansourian, Amir Mohammad and Zhou, Xin and Kasaei, Shohreh and Ghanem, Bernard and Alahi, Alexandre and Van Droogenbroeck, Marc and De Vleeschouwer, Christophe},
  booktitle = {CVPR Workshops},
  month = {June},
  year = {2024}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.
