# FabricIQ - Configuration

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fabric_defect_vgg16_final.keras"
)

DATA_DIR = os.path.join(BASE_DIR, "data")

HISTORY_PATH = os.path.join(
    DATA_DIR,
    "inspection_history.csv"
)

# Model configuration
IMG_SIZE = 224

MODEL_NAME = "VGG16 Transfer Learning"

MODEL_ACCURACY = 90.98

# Fabric defect categories
CLASS_NAMES = [
    "Broken stitch",
    "defect free",
    "hole",
    "horizontal",
    "lines",
    "Needle mark",
    "Pinched fabric",
    "stain",
    "Vertical"
]

# Create data directory automatically
os.makedirs(DATA_DIR, exist_ok=True)