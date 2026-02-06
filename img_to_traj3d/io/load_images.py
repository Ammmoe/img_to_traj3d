import os
import cv2


def list_image_files(image_folder, extensions=(".png", ".jpg", ".jpeg")):
    """
    Returns sorted list of image filenames in the folder.
    Sorting is important for correct time sequence.

    Parameters
    ----------
    image_folder : str
    extensions : tuple

    Returns
    -------
    list[str]
    """
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(extensions)
    ]

    files.sort()
    return files


def load_image(image_path):
    """
    Loads a single image using OpenCV.

    Returns
    -------
    frame : np.ndarray
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return frame


def load_image_sequence(image_folder, extensions=(".png", ".jpg", ".jpeg")):
    """
    Returns sorted full paths for all images.

    Returns
    -------
    list[str]
    """
    filenames = list_image_files(image_folder, extensions=extensions)
    return [os.path.join(image_folder, f) for f in filenames]
