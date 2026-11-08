import torch
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Project & Dataset Paths
# -----------------------------------------------------------------------------
# Project repository root (on Kaggle: /kaggle/working/yolo-from-scratch; locally: current dir)
ROOT = Path("/kaggle/working/yolo-from-scratch") if Path("/kaggle/working").exists() else Path(".")

# Pascal VOC 2007 dataset directory on Kaggle
DATA_DIR = Path("/kaggle/input/datasets/stpeteishii/pascal-voc-2007-dataset/VOCdevkit2007/VOC2007")

# Checkpoint directory and file destination (persists on Kaggle under /kaggle/working/)
CHECKPOINT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
CHECKPOINT_FILE = CHECKPOINT_DIR / "yolov1_best.pt"

# -----------------------------------------------------------------------------
# 2. Hardware & Multi-GPU Compute
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# -----------------------------------------------------------------------------
# 3. Model Architecture Settings
# -----------------------------------------------------------------------------
IMAGE_SIZE = 448
S = 7   # Grid split size (7x7)
B = 2   # Number of bounding boxes predicted per grid cell
C = 20  # Number of Pascal VOC classes

# -----------------------------------------------------------------------------
# 4. Loss Weights
# -----------------------------------------------------------------------------
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# -----------------------------------------------------------------------------
# 5. Training Hyperparameters & Discriminative Learning Rates
# -----------------------------------------------------------------------------
# Discriminative learning rates: 10x smaller for pretrained backbone, higher for new head
BACKBONE_LR = 2e-5   # 2e-5: preserves and gently adapts ImageNet features
HEAD_LR = 2e-4       # 2e-4: trains detection head aggressively from scratch

WEIGHT_DECAY = 1e-4
EPOCHS = 20

# Scale batch size and workers for dual GPUs (16 per T4 = 32 total)
BATCH_SIZE = 32 if NUM_GPUS > 1 else 16
NUM_WORKERS = 4 if NUM_GPUS > 1 else 2
PIN_MEMORY = True

# -----------------------------------------------------------------------------
# 6. Inference Thresholds
# -----------------------------------------------------------------------------
CONF_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.5
