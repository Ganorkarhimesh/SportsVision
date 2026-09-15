from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from ultralytics import YOLO

from ..models.resnet_features import ResNet50FeatureExtractor
from ..training.model import SportsVisionModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "yolov8n.pt"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "models"
    / "run_1"
    / "best_model.pth"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_FRAMES = 32
FEATURE_SIZE = 2048
NUM_CLASSES = 11

# Test accuracy of the current run_1 model
MODEL_TEST_ACCURACY = 83.33

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

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
    "walking",
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Device:", DEVICE)

print("Loading YOLO...")
yolo_model = YOLO(str(YOLO_MODEL_FILE))

print("Loading ResNet-50...")
resnet_model = ResNet50FeatureExtractor()
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

print("Loading SportsVision model...")

sports_model = SportsVisionModel(
    input_size=2048,
    hidden_size=256,
    num_layers=2,
    num_heads=8,
    num_classes=11,
    dropout=0.3,
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

sports_model.load_state_dict(checkpoint)

sports_model = sports_model.to(DEVICE)
sports_model.eval()

print("All models loaded successfully.")


# ============================================================
# SAMPLE 32 FRAMES
# ============================================================

def sample_video_frames(video_path, num_frames=32):
    """
    Extract exactly num_frames frames from a video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video frame count."
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if success:
            frames.append(frame)

    cap.release()

    return frames


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_person(frame):
    """
    Detect the largest person in the frame.

    Returns:
        cropped person ROI
        or None if no person is detected
    """

    results = yolo_model(
        frame,
        verbose=False
    )

    best_box = None
    best_area = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            # COCO class 0 = person
            if class_id != 0:
                continue

            confidence = float(box.conf[0])

            if confidence < 0.30:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))

            x2 = min(
                frame.shape[1],
                int(x2)
            )

            y2 = min(
                frame.shape[0],
                int(y2)
            )

            width = x2 - x1
            height = y2 - y1

            area = width * height

            if area > best_area:
                best_area = area
                best_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return roi


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_video_features(video_path):
    """
    Convert a video into a tensor of shape:

        [1, 32, 2048]
    """

    print("\nSampling video frames...")

    frames = sample_video_frames(
        video_path,
        NUM_FRAMES
    )

    print(
        "Frames obtained:",
        len(frames)
    )

    feature_sequence = []

    for frame_number, frame in enumerate(frames):

        roi = detect_person(frame)

        if roi is None:

            print(
                f"Frame {frame_number + 1}: "
                f"No person detected"
            )

            feature_sequence.append(
                torch.zeros(FEATURE_SIZE)
            )

            continue

        try:

            feature = resnet_model.extract(roi)

            if feature is None:
                feature = torch.zeros(
                    FEATURE_SIZE
                )

            feature_sequence.append(
                feature.cpu()
            )

            print(
                f"Frame {frame_number + 1}: "
                f"Feature extracted"
            )

        except Exception as error:

            print(
                f"Frame {frame_number + 1}: "
                f"Feature extraction failed"
            )

            print(
                "Error:",
                error
            )

            feature_sequence.append(
                torch.zeros(FEATURE_SIZE)
            )

    # --------------------------------------------------------
    # Make sure sequence contains exactly 32 frames
    # --------------------------------------------------------

    if len(feature_sequence) < NUM_FRAMES:

        missing_frames = (
            NUM_FRAMES
            - len(feature_sequence)
        )

        for _ in range(missing_frames):

            feature_sequence.append(
                torch.zeros(FEATURE_SIZE)
            )

    elif len(feature_sequence) > NUM_FRAMES:

        feature_sequence = (
            feature_sequence[:NUM_FRAMES]
        )

    feature_tensor = torch.stack(
        feature_sequence
    )

    print(
        "Feature tensor shape:",
        feature_tensor.shape
    )

    # Add batch dimension
    feature_tensor = feature_tensor.unsqueeze(0)

    return feature_tensor


# ============================================================
# PREDICTION
# ============================================================

def predict_video(video_path):
    """
    Predict the action performed in a video.

    Returns:
        Dictionary containing:

        - prediction
        - probability
        - top_predictions
        - test_accuracy
    """

    video_path = Path(video_path)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print("\n" + "=" * 60)
    print("SPORTSVISION VIDEO PREDICTION")
    print("=" * 60)

    print(
        "Video:",
        video_path
    )

    # --------------------------------------------------------
    # Extract 32 × 2048 features
    # --------------------------------------------------------

    features = extract_video_features(
        video_path
    )

    features = features.to(DEVICE)

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = sports_model(
            features
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # Get Top-3 predictions
    # --------------------------------------------------------

    top_probabilities, top_indices = torch.topk(
        probabilities[0],
        k=3
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_index = index.item()

        class_name = CLASS_NAMES[
            class_index
        ]

        probability_value = (
            probability.item() * 100
        )

        top_predictions.append(
            {
                "class": class_name,
                "probability": round(
                    probability_value,
                    2
                )
            }
        )

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    predicted_label = (
        top_predictions[0]["class"]
    )

    prediction_probability = (
        top_predictions[0]["probability"]
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(
        f"Action              : "
        f"{predicted_label}"
    )

    print(
        f"Prediction probability: "
        f"{prediction_probability:.2f}%"
    )

    print(
        f"Model test accuracy : "
        f"{MODEL_TEST_ACCURACY:.2f}%"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Show Top-3 predictions
    # --------------------------------------------------------

    print("\nTop 3 predictions:")

    for rank, item in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{rank}. "
            f"{item['class']:22s} "
            f"{item['probability']:.2f}%"
        )

    # --------------------------------------------------------
    # Show all class probabilities
    # --------------------------------------------------------

    print("\nAll class probabilities:")

    probability_values = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    sorted_indices = np.argsort(
        probability_values
    )[::-1]

    for index in sorted_indices:

        print(
            f"{CLASS_NAMES[index]:22s} "
            f"{probability_values[index] * 100:.2f}%"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "prediction": predicted_label,

        "probability": prediction_probability,

        "top_predictions": top_predictions,

        "test_accuracy": MODEL_TEST_ACCURACY,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THIS VIDEO PATH
    # --------------------------------------------------------

    VIDEO_PATH = (
        PROJECT_ROOT
        / "dataset"
        / "UCF11"
        / "tennis_swing"
        / "v_tennis_01"
        / "v_tennis_01_01.mpg"
    )

    result = predict_video(
        VIDEO_PATH
    )

    print("\nReturned result:")
    print(result)