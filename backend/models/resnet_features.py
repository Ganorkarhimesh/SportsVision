import torch
import torch.nn as nn
import numpy as np

from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet-50 based spatial feature extractor.

    Role in SportsVision:
        YOLOv8
            ↓
        Athlete ROI
            ↓
        ResNet-50
            ↓
        2048-dimensional feature vector
            ↓
        Bi-LSTM
            ↓
        Self-Attention
            ↓
        Action Classification

    ResNet-50 is NOT used to predict the final sports action.
    It extracts spatial information from each athlete image/frame.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet-50 weights
        weights = ResNet50_Weights.DEFAULT

        model = resnet50(weights=weights)

        # Remove the final ImageNet classification layer.
        # The remaining network produces a 2048-dimensional
        # feature representation.
        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

        # Preprocessing required by the pretrained weights.
        self.transform = weights.transforms()

        # Feature extraction only
        self.eval()

        # Freeze ResNet-50 parameters
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def extract(self, image):
        """
        Extract a 2048-dimensional feature vector
        from an athlete ROI.

        Parameters
        ----------
        image : numpy.ndarray
            Athlete ROI obtained from YOLOv8.
            OpenCV/YOLO image is normally in BGR format.

        Returns
        -------
        torch.Tensor
            2048-dimensional feature vector.
        """

        # If YOLO did not detect an athlete
        if image is None:
            return None

        # Make sure the input is a NumPy array
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image)}"
            )

        # Convert BGR → RGB
        image = image[:, :, ::-1].copy()

        # Convert NumPy array → PIL Image.
        #
        # torchvision's weights.transforms() expects
        # a PIL image or tensor, not a NumPy array.
        image = Image.fromarray(image)

        # Apply ResNet preprocessing:
        # - resize
        # - center crop
        # - convert to tensor
        # - normalize
        tensor = self.transform(image)

        # Add batch dimension
        # [3, 224, 224] → [1, 3, 224, 224]
        tensor = tensor.unsqueeze(0)

        # Move input to same device as ResNet
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        # Extract CNN features
        # [1, 2048, 1, 1]
        features = self.features(tensor)

        # Flatten spatial dimensions
        # [1, 2048, 1, 1] → [1, 2048]
        features = features.flatten(1)

        # Remove batch dimension
        # [1, 2048] → [2048]
        return features.squeeze(0)