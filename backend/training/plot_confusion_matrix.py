from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .dataset import SportsVisionDataset
from .model import SportsVisionModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "dataset" / "features"
SPLIT_FILE = PROJECT_ROOT / "dataset" / "splits.pt"
MODEL_FILE = PROJECT_ROOT / "dataset" / "sportsvision_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Settings
# --------------------------------------------------

BATCH_SIZE = 8

CLASS_NAMES = [
    "basketball",
    "biking",
    "diving",
    "golf_swing",
    "horse_riding",
    "soccer_juggling",
    "swing",
    "tennis_swing",
    "trampoline_jumping",
    "volleyball_spiking",
    "walking"
]


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

dataset = SportsVisionDataset(FEATURES_DIR)

splits = torch.load(
    SPLIT_FILE,
    map_location="cpu"
)

test_dataset = Subset(
    dataset,
    splits["test"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = SportsVisionModel(
    input_size=2048,
    hidden_size=256,
    num_layers=2,
    num_heads=8,
    num_classes=11,
    dropout=0.3
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Get Predictions
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 10)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "SportsVision - Confusion Matrix"
)

plt.tight_layout()


# --------------------------------------------------
# Save
# --------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved at:")
print(output_file)