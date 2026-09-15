from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from .dataset import SportsVisionDataset
from .model import SportsVisionModel


# ==================================================
# Reproducibility
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Random seed:", SEED)


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "dataset" / "features"
SPLIT_FILE = PROJECT_ROOT / "dataset" / "splits.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "dataset" / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ==================================================
# Training Settings
# ==================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# Create Unique Run Folder
# ==================================================

run_number = 1

while (MODELS_DIR / f"run_{run_number}").exists():
    run_number += 1

RUN_DIR = MODELS_DIR / f"run_{run_number}"
RUN_DIR.mkdir(parents=True)

RUN_RESULTS_DIR = RESULTS_DIR / f"run_{run_number}"
RUN_RESULTS_DIR.mkdir(parents=True)

BEST_MODEL_FILE = RUN_DIR / "best_model.pth"
FINAL_MODEL_FILE = RUN_DIR / "final_model.pth"

print("Run:", run_number)
print("Model folder:", RUN_DIR)
print("Results folder:", RUN_RESULTS_DIR)


# ==================================================
# Dataset
# ==================================================

dataset = SportsVisionDataset(FEATURES_DIR)

splits = torch.load(
    SPLIT_FILE,
    map_location="cpu"
)

train_dataset = Subset(
    dataset,
    splits["train"]
)

val_dataset = Subset(
    dataset,
    splits["val"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==================================================
# Model
# ==================================================

model = SportsVisionModel(
    input_size=2048,
    hidden_size=256,
    num_layers=2,
    num_heads=8,
    num_classes=11,
    dropout=0.3
)

model = model.to(device)


# ==================================================
# Loss and Optimizer
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# Training History
# ==================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0
best_epoch = 0


# ==================================================
# Training Loop
# ==================================================

for epoch in range(EPOCHS):

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()

    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_train += labels.size(0)

    train_loss = (
        total_train_loss /
        total_train
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            total_val_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_val += (
                (predictions == labels)
                .sum()
                .item()
            )

            total_val += labels.size(0)


    val_loss = (
        total_val_loss /
        total_val
    )

    val_accuracy = (
        correct_val /
        total_val
    )


    # ----------------------------------------------
    # Store History
    # ----------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # ----------------------------------------------
    # Save Best Model
    # ----------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            BEST_MODEL_FILE
        )


    # ----------------------------------------------
    # Print Epoch Results
    # ----------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy * 100:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy * 100:.2f}%"
    )


# ==================================================
# Save Final Model
# ==================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_FILE
)


# ==================================================
# Save Training History
# ==================================================

history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies,
    "best_val_accuracy": best_val_accuracy,
    "best_epoch": best_epoch
}

HISTORY_FILE = RUN_RESULTS_DIR / "history.pt"

torch.save(
    history,
    HISTORY_FILE
)


# ==================================================
# Plot Loss
# ==================================================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    f"SportsVision Loss - Run {run_number}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

LOSS_FILE = RUN_RESULTS_DIR / "loss_curve.png"

plt.savefig(
    LOSS_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# Plot Accuracy
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    f"SportsVision Accuracy - Run {run_number}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

ACCURACY_FILE = RUN_RESULTS_DIR / "accuracy_curve.png"

plt.savefig(
    ACCURACY_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# Final Output
# ==================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

print()
print("Best model:")
print(BEST_MODEL_FILE)

print()
print("Final model:")
print(FINAL_MODEL_FILE)

print()
print("Training history:")
print(HISTORY_FILE)

print()
print("Graphs:")
print(LOSS_FILE)
print(ACCURACY_FILE)