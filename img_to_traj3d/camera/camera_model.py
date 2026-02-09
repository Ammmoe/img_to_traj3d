import numpy as np
import pandas as pd


def load_camera_poses_from_csv(filepath):
    df = pd.read_csv(filepath)

    # Extract camera positions as Nx3 array
    translations = df[['x_friendly', 'y_friendly', 'z_friendly']].values

    # Convert to list of (3,1) column vectors
    translations_list = [translations[i].reshape(3,1) for i in range(len(translations))]

    return translations_list


def load_camera_poses_from_csv_with_quat(filepath):
    df = pd.read_csv(filepath)

    # Extract translations as Nx3 array
    translations = df[['x_friendly', 'y_friendly', 'z_friendly']].values
    translations_list = [translations[i].reshape(3,1) for i in range(len(translations))]

    # Extract quaternions as Nx4 array
    quats = df[['x_r_friendly', 'y_r_friendly', 'z_r_friendly', 'w_r_friendly']].values
    rotation_matrices = []
    
    # new
    # R_fixed = euler_to_rotation_matrix(0.0, 270.0, 0.0, order='zyx')
    
    for qx, qy, qz, qw in quats:
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # # new
        # R_quat = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        # R = R_fixed @ R_quat
        
        rotation_matrices.append(R)

    return translations_list, rotation_matrices


def load_camera_poses(N, camera_pose_csv_path):
    # K, R_single = load_camera_config()  # now only returns two variables
    # R_list = [R_single] * N  # replicate R for all frames
    
    # T_list = load_camera_poses_from_csv(camera_pose_csv_path)
    # if len(T_list) < N:
    #     raise ValueError("Not enough camera poses in CSV for N frames")
    
    # return K, R_list[:N], T_list[:N]
    
    """
    Loads camera intrinsics, rotations and translations for N frames.
    - K: 3x3 intrinsics matrix (assumed constant)
    - R_list: list of 3x3 rotation matrices for frames 0..N-1
    - T_list: list of 3x1 translation vectors for frames 0..N-1 (camera positions)
    """
    K, _ = load_camera_config()  # load intrinsics only

    T_list, R_list_full = load_camera_poses_from_csv_with_quat(camera_pose_csv_path)

    if len(T_list) < N or len(R_list_full) < N:
        raise ValueError("Not enough camera poses in CSV for N frames")

    R_list = R_list_full[:N]
    T_list = T_list[:N]

    return K, R_list, T_list


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


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    Quaternion should be normalized.
    """
    # Normalize quaternion to be safe
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ])
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
    # # From your example config:
    # RX = 0.0
    # RY = 270.0
    # RZ = 0.0

    # img_width = 640
    # img_height = 360

    # sensor_width = 32.0
    # sensor_height = 18.0

    # focal_length = 35.0  # mm

    # K = get_intrinsic_matrix(focal_length, sensor_width, sensor_height, img_width, img_height)
    # R = euler_to_rotation_matrix(RX, RY, RZ, order='zyx')

    # # For now, return just one R and K
    # return K, R
    
    # From your example config (intrinsics only)
    img_width = 640
    img_height = 360
    sensor_width = 32.0
    sensor_height = 18.0
    focal_length = 35.0  # mm

    K = get_intrinsic_matrix(focal_length, sensor_width, sensor_height, img_width, img_height)
    return K, None  # No fixed rotation anymore
