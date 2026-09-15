import cv2

def draw_detection(
    frame,
    bbox,
    confidence,
    action=None
):
    """
    Draw athlete detection on a video frame.

    Parameters
    ----------
    frame : numpy.ndarray
        Original video frame.

    bbox : list
        Bounding box:
        [x1, y1, x2, y2]

    confidence : float
        YOLO detection confidence.

    action : str, optional
        Predicted action.

    Returns
    -------
    frame : numpy.ndarray
        Frame with annotations.
    """

    # If no athlete was detected,
    # simply return the original frame.
    if bbox is None:
        return frame

    # Extract bounding box coordinates.
    x1, y1, x2, y2 = bbox

    # Convert coordinates to integers.
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Draw bounding box around athlete.
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),

        # BGR color used by OpenCV.
        (139, 92, 246),

        2
    )

    # Display YOLO confidence.
    label = (
        f"Athlete: "
        f"{confidence:.2f}"
    )

    # If action prediction is available,
    # add it to the label.
    if action is not None:
        label += f" | {action}"

    # Draw label background.
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 30)),
        (x1 + 300, y1),
        (139, 92, 246),
        -1
    )

    # Draw text.
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return frame