# backend/inference.py

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.yolov8_detector import YOLOv8Detector
from models.resnet_features import ResNet50FeatureExtractor
from models.temporal_model import ActionRecognitionModel

from utils.video_processing import sample_video_frames
from utils.visualization import draw_detection


class SportsVisionInference:
    """
    Complete inference pipeline for SportsVision.

    Pipeline:

        Input Video
             ↓
        Frame Sampling
             ↓
           YOLOv8
             ↓
        Athlete ROI
             ↓
          ResNet-50
             ↓
        2048-D Features
             ↓
          Bi-LSTM
             ↓
       Self-Attention
             ↓
        Action Class
    """

    def __init__(
        self,
        yolo_model_path,
        action_model_path,
        num_classes=6
    ):
        """
        Initialize all required models.

        Parameters
        ----------
        yolo_model_path : str
            YOLOv8 weights.

        action_model_path : str
            Trained Bi-LSTM model weights.

        num_classes : int
            Number of sports action classes.
        """

        # Use GPU if CUDA is available.
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        # -------------------------------------------------
        # YOLOv8
        # -------------------------------------------------

        self.detector = YOLOv8Detector(
            yolo_model_path
        )

        # -------------------------------------------------
        # ResNet-50
        # -------------------------------------------------

        self.feature_extractor = (
            ResNet50FeatureExtractor()
            .to(self.device)
        )

        # -------------------------------------------------
        # Bi-LSTM + Attention
        # -------------------------------------------------

        self.action_model = ActionRecognitionModel(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)

        # Load trained action recognition weights.
        self._load_action_model(
            action_model_path
        )

        # Action labels.
        #
        # IMPORTANT:
        # These must exactly match the classes
        # used while training the model.
        self.action_classes = [
            "Cricket Batting",
            "Cricket Bowling",
            "Football Kicking",
            "Tennis Swing",
            "Basketball Shooting",
            "Basketball Dunking"
        ]

    def _load_action_model(
        self,
        model_path
    ):
        """
        Load trained Bi-LSTM model weights.
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Action model not found: {model_path}"
            )

        # Load checkpoint.
        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Some training scripts save:
        #
        # {
        #     "model_state_dict": ...
        # }
        #
        # while others directly save:
        #
        # model.state_dict()
        #
        # Support both formats.
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        self.action_model.load_state_dict(
            state_dict
        )

        # Put model into evaluation mode.
        self.action_model.eval()

    def predict(
        self,
        video_path,
        output_video_path=None
    ):
        """
        Predict the action present in a sports video.

        Parameters
        ----------
        video_path : str
            Input video path.

        output_video_path : str, optional
            Path where annotated video should be saved.

        Returns
        -------
        dict
            Prediction information.
        """

        start_time = time.time()

        # -------------------------------------------------
        # STEP 1: Sample video frames
        # -------------------------------------------------

        frames, fps = sample_video_frames(
            video_path,
            num_frames=32
        )

        if len(frames) == 0:
            raise ValueError(
                "No frames could be extracted."
            )

        # Store ResNet features from every
        # successfully detected athlete.
        feature_sequence = []

        # Store detection information.
        detection_data = []

        # -------------------------------------------------
        # STEP 2: YOLOv8 + ResNet-50
        # -------------------------------------------------

        for frame in frames:

            # Detect athlete.
            roi, bbox, confidence = (
                self.detector.detect_athlete(
                    frame
                )
            )

            detection_data.append({
                "bbox": bbox,
                "confidence": confidence
            })

            # If no athlete was detected,
            # skip this frame.
            if roi is None:
                continue

            # Extract 2048-D spatial features.
            features = (
                self.feature_extractor.extract(
                    roi
                )
            )

            if features is not None:
                feature_sequence.append(
                    features
                )

        # We cannot make an action prediction
        # if no athlete was detected.
        if len(feature_sequence) == 0:
            raise ValueError(
                "No athlete was detected in the video."
            )

        # -------------------------------------------------
        # STEP 3: Create temporal sequence
        # -------------------------------------------------

        # Stack individual feature vectors.
        #
        # Example:
        #
        # 32 frames
        # each → 2048 features
        #
        # Result:
        # [32, 2048]
        feature_sequence = torch.stack(
            feature_sequence
        )

        # Add batch dimension.
        #
        # [32, 2048]
        #       ↓
        # [1, 32, 2048]
        #
        # This is the format expected by
        # our Bi-LSTM.
        feature_sequence = (
            feature_sequence
            .unsqueeze(0)
            .to(self.device)
        )

        # -------------------------------------------------
        # STEP 4: Bi-LSTM + Attention
        # -------------------------------------------------

        with torch.no_grad():

            logits, attention_weights = (
                self.action_model(
                    feature_sequence
                )
            )

            # Convert logits to probabilities.
            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # Find class with highest probability.
            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][
                    predicted_class
                ].item()
            )

        action = self.action_classes[
            predicted_class
        ]

        # -------------------------------------------------
        # STEP 5: Calculate average YOLO confidence
        # -------------------------------------------------

        valid_confidences = [
            item["confidence"]
            for item in detection_data
            if item["bbox"] is not None
        ]

        average_yolo_confidence = (
            sum(valid_confidences)
            / len(valid_confidences)
            if valid_confidences
            else 0.0
        )

        # -------------------------------------------------
        # STEP 6: Create annotated video
        # -------------------------------------------------

        if output_video_path is not None:

            self.create_processed_video(
                video_path,
                output_video_path,
                action
            )

        processing_time = (
            time.time() - start_time
        )

        return {
            "action": action,
            "confidence": round(
                confidence,
                4
            ),
            "yolo_confidence": round(
                average_yolo_confidence,
                4
            ),
            "frames": len(frames),
            "processing_time": round(
                processing_time,
                2
            ),
            "detected_athlete": (
                len(valid_confidences) > 0
            )
        }

    def create_processed_video(
        self,
        video_path,
        output_path,
        action
    ):
        """
        Create a video with athlete bounding boxes
        and predicted action displayed on frames.

        Note:
        This function performs YOLO inference again
        on the video frames for visualization.
        """

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        # MP4 video writer.
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        while True:

            success, frame = (
                capture.read()
            )

            if not success:
                break

            # Detect athlete again.
            roi, bbox, confidence = (
                self.detector.detect_athlete(
                    frame
                )
            )

            # Draw detection and predicted action.
            if bbox is not None:
                frame = draw_detection(
                    frame,
                    bbox,
                    confidence,
                    action
                )

            writer.write(frame)

        capture.release()
        writer.release()