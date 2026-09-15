import cv2
import numpy as np


def sample_video_frames(
    video_path,
    num_frames=32
):
    """
    Uniformly sample frames from a video.

    Why sampling?

    A video may contain hundreds or thousands
    of frames.

    Processing every frame would be expensive.

    Instead, we select a fixed number of frames.

    Example:

        300-frame video
               ↓
        select 32 frames
               ↓
        YOLOv8
               ↓
        ResNet-50
               ↓
        Bi-LSTM

    Parameters
    ----------
    video_path : str
        Path to video file.

    num_frames : int
        Number of frames to sample.

    Returns
    -------
    frames : list
        List of sampled video frames.

    fps : float
        Frames per second of original video.
    """

    # Open the video.
    capture = cv2.VideoCapture(
        video_path
    )

    # Check whether OpenCV successfully
    # opened the video.
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    # Get total number of frames.
    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Get video FPS.
    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    # Make sure FPS is valid.
    if fps <= 0:
        fps = 30.0

    # If the video contains fewer frames
    # than requested, use all available frames.
    actual_num_frames = min(
        num_frames,
        total_frames
    )

    if actual_num_frames <= 0:
        capture.release()

        raise ValueError(
            "Video does not contain valid frames."
        )

    # Generate evenly spaced frame indices.
    #
    # Example:
    #
    # total frames = 300
    # num_frames = 32
    #
    # np.linspace selects 32 positions
    # throughout the video.
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        actual_num_frames,
        dtype=int
    )

    frames = []

    # Read each selected frame.
    for frame_index in frame_indices:

        # Move video pointer to the required frame.
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = capture.read()

        # Add only successfully read frames.
        if success:
            frames.append(frame)

    # Release video resources.
    capture.release()

    return frames, fps


def get_video_info(video_path):
    """
    Get basic information about a video.

    Returns:
        width
        height
        fps
        total frames
        duration
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

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Calculate video duration.
    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    capture.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
        "duration": duration
    }