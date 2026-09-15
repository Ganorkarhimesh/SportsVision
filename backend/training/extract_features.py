from pathlib import Path

import torch

from models.yolov8_detector import YOLOv8Detector
from models.resnet_features import ResNet50FeatureExtractor
from utils.video_processing import sample_video_frames


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "UCF11"

FEATURES_DIR = PROJECT_ROOT / "dataset" / "features"

YOLO_MODEL_PATH = PROJECT_ROOT / "backend" / "yolov8n.pt"


# ============================================================
# SETTINGS
# ============================================================

NUM_FRAMES = 32

FEATURE_SIZE = 2048

VIDEO_EXTENSIONS = [
    ".avi",
    ".mp4",
    ".mov",
    ".wmv",
    ".mpeg",
    ".mpg"
]


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_features():

    print("=" * 60)
    print("SportsVision Feature Extraction")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_DIR}"
        )

    print("\nDataset directory:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_names = sorted(
        [
            folder.name
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    if len(class_names) == 0:
        raise RuntimeError(
            f"No class folders found inside:\n{DATASET_DIR}"
        )

    print("\nClasses found:")

    for number, class_name in enumerate(
        class_names,
        start=1
    ):
        print(f"  {number}. {class_name}")

    print(f"\nTotal classes: {len(class_names)}")

    # --------------------------------------------------------
    # Check YOLO model
    # --------------------------------------------------------

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nYOLO model not found:\n"
            f"{YOLO_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU not available. "
            "Feature extraction will run on CPU."
        )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("\nLoading YOLOv8...")

    detector = YOLOv8Detector(
        YOLO_MODEL_PATH
    )

    print("YOLOv8 loaded successfully.")

    # --------------------------------------------------------
    # Load ResNet
    # --------------------------------------------------------

    print("\nLoading ResNet-50...")

    feature_extractor = (
        ResNet50FeatureExtractor()
        .to(device)
    )

    print("ResNet-50 loaded successfully.")

    # --------------------------------------------------------
    # Create feature directory
    # --------------------------------------------------------

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_number, class_name in enumerate(
        class_names,
        start=1
    ):

        class_dir = DATASET_DIR / class_name

        # ----------------------------------------------------
        # Find all videos recursively
        # ----------------------------------------------------

        video_files = []

        for video_file in class_dir.rglob("*"):

            if (
                video_file.is_file()
                and video_file.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                video_files.append(video_file)

        video_files = sorted(video_files)

        # ----------------------------------------------------
        # Class information
        # ----------------------------------------------------

        print("\n" + "=" * 60)

        print(
            f"Class {class_number}/{len(class_names)}: "
            f"{class_name}"
        )

        print(
            f"Videos found: {len(video_files)}"
        )

        print("=" * 60)

        if len(video_files) == 0:

            print(
                f"No video files found in "
                f"{class_dir}"
            )

            continue

        # ----------------------------------------------------
        # Process videos
        # ----------------------------------------------------

        for video_number, video_path in enumerate(
            video_files,
            start=1
        ):

            # ------------------------------------------------
            # Preserve folder structure
            # ------------------------------------------------

            relative_video_path = (
                video_path.relative_to(class_dir)
            )

            relative_video_folder = (
                relative_video_path.parent
            )

            output_class_dir = (
                FEATURES_DIR
                / class_name
                / relative_video_folder
            )

            output_class_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                output_class_dir
                / f"{video_path.stem}.pt"
            )

            # ------------------------------------------------
            # Skip existing files
            # ------------------------------------------------

            if output_path.exists():

                print(
                    f"[{video_number}/"
                    f"{len(video_files)}] "
                    f"Already processed: "
                    f"{video_path.name}"
                )

                total_skipped += 1

                continue

            # ------------------------------------------------
            # Processing message
            # ------------------------------------------------

            print(
                f"\n[{video_number}/"
                f"{len(video_files)}] "
                f"Processing: "
                f"{class_name}/"
                f"{relative_video_path}"
            )

            try:

                # ============================================
                # SAMPLE VIDEO FRAMES
                # ============================================

                frames, fps = sample_video_frames(
                    str(video_path),
                    num_frames=NUM_FRAMES
                )

                print(
                    f"  Frames sampled: "
                    f"{len(frames)}"
                )

                # ------------------------------------------------
                # No frames
                # ------------------------------------------------

                if len(frames) == 0:

                    print(
                        "  Skipped: "
                        "No frames extracted."
                    )

                    total_skipped += 1

                    continue

                # ============================================
                # FEATURE SEQUENCE
                # ============================================

                feature_sequence = []

                detected_frames = 0

                # ------------------------------------------------
                # Process sampled frames
                # ------------------------------------------------

                for frame_number, frame in enumerate(
                    frames,
                    start=1
                ):

                    # ========================================
                    # YOLO PERSON DETECTION
                    # ========================================

                    roi, bbox, confidence = (
                        detector.detect_athlete(frame)
                    )

                    # ========================================
                    # ATHLETE DETECTED
                    # ========================================

                    if roi is not None:

                        features = (
                            feature_extractor.extract(roi)
                        )

                        if features is not None:

                            features = (
                                features.cpu().flatten()
                            )

                            if (
                                features.shape[0]
                                == FEATURE_SIZE
                            ):

                                feature_sequence.append(
                                    features
                                )

                                detected_frames += 1

                            else:

                                print(
                                    f"  Warning: "
                                    f"Unexpected feature size "
                                    f"at frame "
                                    f"{frame_number}"
                                )

                                feature_sequence.append(
                                    torch.zeros(
                                        FEATURE_SIZE
                                    )
                                )

                        else:

                            feature_sequence.append(
                                torch.zeros(
                                    FEATURE_SIZE
                                )
                            )

                    # ========================================
                    # ATHLETE NOT DETECTED
                    # ========================================

                    else:

                        feature_sequence.append(
                            torch.zeros(
                                FEATURE_SIZE
                            )
                        )

                # ============================================
                # FIX SEQUENCE LENGTH
                # ============================================

                current_length = len(feature_sequence)

                # ------------------------------------------------
                # If fewer than 32 features
                # ------------------------------------------------

                if current_length < NUM_FRAMES:

                    missing_frames = (
                        NUM_FRAMES - current_length
                    )

                    print(
                        f"  Padding "
                        f"{missing_frames} missing "
                        f"feature vectors."
                    )

                    for _ in range(missing_frames):

                        feature_sequence.append(
                            torch.zeros(
                                FEATURE_SIZE
                            )
                        )

                # ------------------------------------------------
                # If more than 32 features
                # ------------------------------------------------

                elif current_length > NUM_FRAMES:

                    print(
                        f"  Trimming "
                        f"{current_length - NUM_FRAMES} "
                        f"extra feature vectors."
                    )

                    feature_sequence = (
                        feature_sequence[:NUM_FRAMES]
                    )

                # ============================================
                # FINAL TENSOR
                # ============================================

                feature_tensor = torch.stack(
                    feature_sequence
                )

                # ============================================
                # FINAL SHAPE CHECK
                # ============================================

                expected_shape = (
                    NUM_FRAMES,
                    FEATURE_SIZE
                )

                if tuple(feature_tensor.shape) != expected_shape:

                    raise RuntimeError(
                        f"Final feature shape is "
                        f"{tuple(feature_tensor.shape)} "
                        f"instead of "
                        f"{expected_shape}"
                    )

                # ============================================
                # SAVE FEATURES
                # ============================================

                torch.save(
                    feature_tensor,
                    output_path
                )

                print(
                    f"  Saved: "
                    f"{output_path}"
                )

                print(
                    f"  Shape: "
                    f"{tuple(feature_tensor.shape)}"
                )

                print(
                    f"  Athlete detected: "
                    f"{detected_frames}/"
                    f"{len(frames)} frames"
                )

                total_processed += 1

            # ====================================================
            # ERROR HANDLING
            # ====================================================

            except Exception as error:

                print(
                    f"  ERROR processing "
                    f"{video_path.name}:"
                )

                print(
                    f"  {error}"
                )

                total_errors += 1

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "Feature extraction completed."
    )

    print(
        f"Total videos processed: "
        f"{total_processed}"
    )

    print(
        f"Total videos skipped/already processed: "
        f"{total_skipped}"
    )

    print(
        f"Total errors: "
        f"{total_errors}"
    )

    print(
        f"Features saved in:\n"
        f"{FEATURES_DIR}"
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    extract_features()