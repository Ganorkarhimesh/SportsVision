from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from .dataset import SportsVisionDataset
from .model import SportsVisionModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "dataset" / "features"
SPLIT_FILE = PROJECT_ROOT / "dataset" / "splits.pt"
MODEL_FILE = PROJECT_ROOT / "dataset" / "models" / "run_1" / "best_model.pth"


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

dataset = SportsVisionDataset(FEATURES_DIR)

splits = torch.load(SPLIT_FILE, map_location="cpu")

test_indices = splits["test"]

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


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
    torch.load(MODEL_FILE, map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Loss Function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_predictions = []
all_labels = []

total_loss = 0.0
total_samples = 0


with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# --------------------------------------------------
# Calculate Metrics
# --------------------------------------------------

test_loss = total_loss / total_samples

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("SPORTSVISION TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# --------------------------------------------------
# Per-Class Accuracy
# --------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    total = cm[i].sum()

    correct = cm[i, i]

    if total > 0:
        accuracy = correct / total * 100
    else:
        accuracy = 0

    print(
        f"{class_name:<25} : "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


print("\nEvaluation completed.")