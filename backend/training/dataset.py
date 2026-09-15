from pathlib import Path

import torch
from torch.utils.data import Dataset


class SportsVisionDataset(Dataset):
    """
    PyTorch Dataset for SportsVision.

    Each .pt file contains:
        32 frames × 2048 features

    Output:
        features -> [32, 2048]
        label    -> integer from 0 to 10
    """

    def __init__(self, features_dir):
        self.features_dir = Path(features_dir)

        # Class names used in UCF11
        self.class_names = sorted([
            folder.name
            for folder in self.features_dir.iterdir()
            if folder.is_dir()
        ])

        # Convert class name to integer label
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        # Find all feature files
        self.samples = []

        for class_name in self.class_names:

            class_dir = self.features_dir / class_name

            for feature_file in class_dir.rglob("*.pt"):

                self.samples.append(
                    (
                        feature_file,
                        self.class_to_idx[class_name]
                    )
                )

        print(f"Classes: {self.class_names}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        feature_file, label = self.samples[index]

        # Load feature tensor
        features = torch.load(
            feature_file,
            map_location="cpu"
        )

        # Make sure features are float32
        features = features.float()

        return features, torch.tensor(label, dtype=torch.long)