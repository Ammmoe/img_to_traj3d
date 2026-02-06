import cv2
from .bbox_utils import bbox_to_int, bbox_center
import pandas as pd


def create_kcf_tracker():
    """
    OpenCV version-safe KCF tracker creation.
    """
    # if hasattr(cv2, "TrackerKCF_create"):
    #     return cv2.TrackerKCF_create()

    # if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
    #     return cv2.legacy.TrackerKCF_create()

    # raise RuntimeError("KCF tracker is not available in your OpenCV build.")
    
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available")



class KCFTracker:
    def __init__(self):
        self.tracker = create_kcf_tracker()
        self.initialized = False
        self.bbox = None

    def initialize(self, frame, bbox):
        bbox = bbox_to_int(bbox)

        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        if w <= 0 or h <= 0:
            raise RuntimeError(f"Invalid bbox size: width or height <= 0")
        if x < 0 or y < 0 or x + w > w_frame or y + h > h_frame:
            raise RuntimeError(f"Bbox {bbox} is out of frame bounds {frame.shape}")

        try:
            self.tracker.init(frame, bbox)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize KCF tracker: {e}")

        self.initialized = True
        self.bbox = bbox

    def update(self, frame):
        """
        Returns
        -------
        bbox : tuple or None
        center : tuple or None
        success : bool
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        try:
            success, bbox = self.tracker.update(frame)
        except Exception:
            # fallback if update returns only bbox or throws
            bbox = self.tracker.update(frame)
            success = bbox is not None

        if not success:
            return None, None, False

        bbox = bbox_to_int(bbox)
        center = bbox_center(bbox)
        self.bbox = bbox
        return bbox, center, True
    
# In img_to_traj3d/tracking/kcf_tracker.py
def load_tracked_results(csv_path, N=None):
    """
    Load tracked points CSV file.

    Parameters:
        csv_path (str): Path to tracked results CSV.
        N (int, optional): Number of points to load, loads all if None.

    Returns:
        frames (np.array): Frame numbers.
        images (np.array): Image file names.
        pixels (np.array): Pixel coordinates (u,v).
        bbox_centers (np.array): Bounding box centers (optional).
    """
    df = pd.read_csv(csv_path)
    df = df[df['success'] == 1]
    if N is not None:
        df = df.iloc[:N]

    frames = df['frame'].values
    images = df['image'].values
    pixels = df[['u', 'v']].values

    # Optionally calculate bbox centers if needed
    # centers = df[['x', 'y', 'w', 'h']].copy()
    # centers['cx'] = centers['x'] + centers['w'] / 2
    # centers['cy'] = centers['y'] + centers['h'] / 2
    # bbox_centers = centers[['cx', 'cy']].values

    return frames, images, pixels


