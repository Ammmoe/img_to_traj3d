import numpy as np
import pandas as pd


def load_camera_poses_from_csv(filepath):
    df = pd.read_csv(filepath)

    # Extract camera positions as Nx3 array
    translations = df[['x_friendly', 'y_friendly', 'z_friendly']].values

    # Convert to list of (3,1) column vectors
    translations_list = [translations[i].reshape(3,1) for i in range(len(translations))]

    return translations_list


def load_camera_poses(N, camera_pose_csv_path):
    K, R_single = load_camera_config()  # now only returns two variables
    R_list = [R_single] * N  # replicate R for all frames
    
    T_list = load_camera_poses_from_csv(camera_pose_csv_path)
    if len(T_list) < N:
        raise ValueError("Not enough camera poses in CSV for N frames")
    
    return K, R_list[:N], T_list[:N]


def euler_to_rotation_matrix(rx_deg, ry_deg, rz_deg, order='zyx'):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    if order == 'zyx':
        R = Rz @ Ry @ Rx
    elif order == 'xyz':
        R = Rx @ Ry @ Rz
    else:
        raise ValueError("Rotation order not supported")

    return R

def get_intrinsic_matrix(focal_length_mm, sensor_width_mm, sensor_height_mm, img_width_px, img_height_px):
    fx = focal_length_mm * (img_width_px / sensor_width_mm)
    fy = focal_length_mm * (img_height_px / sensor_height_mm)
    cx = img_width_px / 2
    cy = img_height_px / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def load_camera_config():
    # From your example config:
    RX = 0.0
    RY = 270.0
    RZ = 0.0

    img_width = 640
    img_height = 360

    sensor_width = 32.0
    sensor_height = 18.0

    focal_length = 35.0  # mm

    K = get_intrinsic_matrix(focal_length, sensor_width, sensor_height, img_width, img_height)
    R = euler_to_rotation_matrix(RX, RY, RZ, order='zyx')

    # For now, return just one R and K
    return K, R
