"""Single source of truth for paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data (git-ignored; see data/README.md)
DATA_DIR = ROOT / "data"
UCF_DIR = DATA_DIR / "ucf_crime"
HF_DATASET_URL = "https://huggingface.co/datasets/jinmang2/ucf_crime_tencrop_i3d_seg32/resolve/tushar-n"
TRAIN_ZIP = UCF_DIR / "train.zip"
TEST_ZIP = UCF_DIR / "test.zip"
GT_JSON = UCF_DIR / "ground_truth.json"

# Outputs
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# UCF-Crime feature layout (verified on load):
#   train: (10 crops, 32 segments, 2048) - each video pre-pooled into 32 equal temporal segments
#   test:  (T clips, 10 crops, 2048)     - one feature per 16-frame clip; len(ground truth) == 16 * T
N_SEGMENTS = 32
FRAMES_PER_CLIP = 16
N_CROPS = 10          # ten-crop augmentation at extraction time; we average the crops
FEATURE_DIM = 2048    # I3D-ResNet50 (RGB only - no optical-flow stream in this release)

# Expected counts of the official UCF-Crime split - asserted on load so silent data loss is impossible.
N_TRAIN = {"normal": 800, "anomalous": 810}
N_TEST = {"normal": 150, "anomalous": 140}

# Real-time detection + tracking (surveillance.detect)
DETECTOR_MODEL = "yolov8n.pt"   # nano: the only YOLOv8 size near real time on a laptop CPU (see reports/detector_fps.json)
DETECTOR_IMGSZ = 320            # 320: 23.8 FPS end-to-end on M1 CPU (real time) vs 6.5 at 640, at the cost of
                                # ~4% fewer people detected and more track ID switches (reports/detector_tradeoff.json)
DETECTOR_DEVICE = "cpu"         # on this M1, MPS hung twice and was 7x slower than CPU when it ran
PERSON_CLASS = 0                # COCO class id for "person"
LOITER_SECONDS = 20.0           # one tracked person visible longer than this -> "loitering" event
CROWD_MIN_PEOPLE = 8            # at least this many people at once...
CROWD_MIN_SECONDS = 2.0         # ...for at least this long -> "crowd" event
OUTPUTS_DIR = ROOT / "outputs"
SAMPLE_VIDEO = DATA_DIR / "samples" / "vtest.avi"  # OpenCV sample (Apache-2.0), see data/README.md

VAL_FRACTION = 0.1    # held out from TRAIN for model selection (never touch test frames to pick epochs)
SEED = 0
