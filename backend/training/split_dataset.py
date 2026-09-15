from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from training.dataset import SportsVisionDataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURES_DIR = Path("../dataset/features")
SPLIT_FILE = Path("../dataset/splits.pt")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = SportsVisionDataset(FEATURES_DIR)

print("\nTotal samples:", len(dataset))


# --------------------------------------------------
# Get labels
# --------------------------------------------------

labels = []

for _, label in dataset.samples:
    labels.append(label)

labels = torch.tensor(labels)

indices = torch.arange(len(dataset))


# --------------------------------------------------
# Train / Temporary split
# 70% Train
# 30% Temporary
# --------------------------------------------------

train_indices, temp_indices = train_test_split(
    indices.numpy(),
    test_size=0.30,
    random_state=42,
    stratify=labels.numpy()
)


# --------------------------------------------------
# Validation / Test split
# 15% Validation
# 15% Test
# --------------------------------------------------

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=labels.numpy()[temp_indices]
)


# Convert to tensors

train_indices = torch.tensor(train_indices)
val_indices = torch.tensor(val_indices)
test_indices = torch.tensor(test_indices)


# --------------------------------------------------
# Save splits
# --------------------------------------------------

splits = {
    "train": train_indices,
    "val": val_indices,
    "test": test_indices
}

SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

torch.save(splits, SPLIT_FILE)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nDataset split:")
print("-" * 40)

print("Training samples   :", len(train_indices))
print("Validation samples :", len(val_indices))
print("Testing samples    :", len(test_indices))

print(
    "Total              :",
    len(train_indices) +
    len(val_indices) +
    len(test_indices)
)


# --------------------------------------------------
# Display class distribution
# --------------------------------------------------

print("\nClass names:")

for index, class_name in enumerate(dataset.class_names):

    train_count = (labels[train_indices] == index).sum().item()
    val_count = (labels[val_indices] == index).sum().item()
    test_count = (labels[test_indices] == index).sum().item()

    print(
        f"{class_name:20s} "
        f"Train: {train_count:3d} | "
        f"Val: {val_count:3d} | "
        f"Test: {test_count:3d}"
    )


print("\nSplit file saved at:")
print(SPLIT_FILE.resolve())