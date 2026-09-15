from pathlib import Path

from ultralytics import YOLO


class YOLOv8Detector:
    """
    Handles athlete detection using YOLOv8.

    YOLOv8 is NOT used here to classify the sports action.
    Its job is to:
        1. Detect the athlete.
        2. Find the athlete's bounding box.
        3. Crop the athlete region (ROI).

    The ROI is then passed to ResNet-50.
    """

    def __init__(self, model_path: str):
        # Convert the model path into a Path object
        model_path = Path(model_path)

        # Make sure the YOLO model actually exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        # Load the trained YOLOv8 model
        self.model = YOLO(str(model_path))

    def detect_athlete(self, frame):
        """
        Detect the primary athlete in a video frame.

        Parameters:
            frame:
                OpenCV image/frame.

        Returns:
            roi:
                Cropped athlete image.

            bbox:
                Athlete bounding box:
                [x1, y1, x2, y2]

            confidence:
                YOLO detection confidence.
        """

        # Run YOLO inference on the current frame
        results = self.model(
            frame,
            verbose=False
        )

        best_detection = None

        # YOLO can return multiple detections.
        # We search for objects belonging to the
        # "person" class.
        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # COCO class ID 0 represents "person"
                class_id = int(box.cls[0])

                if class_id != 0:
                    continue

                # Confidence of this detection
                confidence = float(box.conf[0])

                # Keep the person with the highest confidence
                if (
                    best_detection is None
                    or confidence >
                    best_detection["confidence"]
                ):
                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    best_detection = {
                        "bbox": coordinates,
                        "confidence": confidence
                    }

        # No athlete/person was detected
        if best_detection is None:
            return None, None, 0.0

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Extract bounding-box coordinates
        x1, y1, x2, y2 = (
            best_detection["bbox"]
        )

        # Make sure coordinates stay inside
        # the image boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        # Crop the athlete from the frame.
        # This is our Region of Interest (ROI).
        roi = frame[
            y1:y2,
            x1:x2
        ]

        # Make sure the crop is valid
        if roi.size == 0:
            return None, None, 0.0

        return (
            roi,
            [x1, y1, x2, y2],
            best_detection["confidence"]
        )