"""Single source of truth for paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data (git-ignored; see data/README.md)
DATA_DIR = ROOT / "data"
UCF_DIR = DATA_DIR / "ucf_crime"
HF_DATASET_URL = "https://huggingface.co/datasets/jinmang2/ucf_crime_tencrop_i3d_seg32/resolve/tushar-n"

# Outputs
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# UCF-Crime feature layout: each video is split into 32 equal temporal segments,
# and each segment is summarised by an I3D feature vector.
N_SEGMENTS = 32
FRAMES_PER_CLIP = 16  # I3D consumes 16-frame clips; ground truth is per frame

SEED = 0
