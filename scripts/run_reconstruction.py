import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from img_to_traj3d.camera.camera_model import load_camera_config, load_camera_poses_from_csv  # Your method to load K, R, T arrays
from img_to_traj3d.tracking.kcf_tracker import load_tracked_results  # optional helper
from img_to_traj3d.utils.math_utils import (
    compute_ray, construct_Theta, build_matrices, ridge_regression,
    reconstruct_positions, select_optimal_order
)

def load_ground_truth(gt_csv_path):
    """
    Load ground truth trajectory for the target.
    Returns numpy arrays for friendly and unauthorized drone positions indexed by frame.
    """
    df = pd.read_csv(gt_csv_path)
    # Extract friendly and unauthorized positions, matching order with tracked images if needed
    friendly = df[['x_friendly', 'y_friendly', 'z_friendly']].values
    unauthorized = df[['x_unauthorized', 'y_unauthorized', 'z_unauthorized']].values
    images = df['image'].values
    return friendly, unauthorized, images

def load_tracked_points(tracked_csv_path, N=70):
    """
    Load first N tracked points and pixel coordinates
    Return list of pixel coords, frame numbers, image names, bbox centers
    """
    df = pd.read_csv(tracked_csv_path)
    df = df[df['success'] == 1].iloc[:N]
    
    # Use (u,v) pixel coords from CSV
    pixels = df[['u', 'v']].values  # shape (N, 2)
    frames = df['frame'].values
    images = df['image'].values
    
    # bbox centers can also be computed as x + w/2, y + h/2 if you want
    # centers = df[['x', 'y', 'w', 'h']].apply(lambda r: (r['x']+r['w']/2, r['y']+r['h']/2), axis=1)
    
    return frames, images, pixels

def load_camera_poses(N, camera_pose_csv_path):
    """
    Dummy function: You need to implement loading actual R_i, T_i, and K for each frame i.
    Return:
      - K: 3x3 intrinsics matrix (assumed constant)
      - R_list: list of 3x3 rotation matrices for frames 0..N-1
      - T_list: list of 3x1 translation vectors for frames 0..N-1 (camera positions)
    """
    K, R_single = load_camera_config()  # now only returns two variables
    R_list = [R_single] * N  # replicate R for all frames
    
    T_list = load_camera_poses_from_csv(camera_pose_csv_path)
    if len(T_list) < N:
        raise ValueError("Not enough camera poses in CSV for N frames")
    
    return K, R_list[:N], T_list[:N]

def main():
    N = 70
    # Paths
    tracked_csv_path = 'data/outputs/csv/tracked_results.csv'
    gt_csv_path = 'data/csv/grouped_data.csv'
    
    # Load data
    frames, images, pixels = load_tracked_points(tracked_csv_path, N)
    friendly_gt, unauthorized_gt, gt_images = load_ground_truth(gt_csv_path)
    K, R_list, T_list = load_camera_poses(N, 'data/csv/grouped_data.csv')
    
    # Compute normalized rays for all tracked points
    L = []
    C = []
    # t_list = (frames - frames[0]) / (frames[-1] - frames[0])  # normalize time to [0,1]
    t_list = frames
    
    for i in range(N):
        l_i = compute_ray(K, R_list[i], T_list[i], pixels[i])
        L.append(l_i)
        # C.append(T_list[i].reshape(3))  # camera center
        # use friendly_gt as camera center
        C.append(friendly_gt[i].reshape(3))
    
    # Select optimal polynomial order and estimate motion parameters
    K_max = 3
    best_K, best_beta = select_optimal_order(K_max, L, C, t_list)
    print(f"Optimal polynomial degree: {best_K}")
    
    # Reconstruct 3D positions using best_beta
    Theta_list = [construct_Theta(t, best_K) for t in t_list]
    P_pred = reconstruct_positions(best_beta, Theta_list)  # shape (N,3)
    
    # Prepare camera trajectory (camera positions)
    # cam_positions = np.array(C)  # (N, 3)
    # Prepare camera trajectory from ground truth
    cam_positions = friendly_gt[:N]
    
    # Prepare ground truth trajectory for unauthorized drone for comparison
    # Match GT images with tracked images, or assume order is consistent
    # We'll take first N points from unauthorized_gt as example
    P_gt = unauthorized_gt[:N]
    
    print("Predicted positions (P_pred) stats:")
    print(f"Min: {P_pred.min(axis=0)}")
    print(f"Max: {P_pred.max(axis=0)}")
    print(f"Mean: {P_pred.mean(axis=0)}")
    
    print("First frame:", frames[0])
    print("Last frame:", frames[-1])
    print("Total rows loaded:", len(frames))

    print("Camera Trajectory:")
    print(f"  Start: {cam_positions[0]}")
    print(f"  End:   {cam_positions[-1]}")

    print("\nGround Truth Unauthorized Drone Trajectory:")
    print(f"  Start: {P_gt[0]}")
    print(f"  End:   {P_gt[-1]}")

    print("\nPredicted Drone Trajectory:")
    print(f"  Start: {P_pred[0]}")
    print(f"  End:   {P_pred[-1]}")
    
    # Plot all trajectories
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cam_positions[:,0], cam_positions[:,1], cam_positions[:,2], 'b-', label='Camera Trajectory')
    ax.plot(P_gt[:,0], P_gt[:,1], P_gt[:,2], 'g-', label='Target Trajectory')
    # ax.plot(P_pred[:,0], P_pred[:,1], P_pred[:,2], 'r--', label='Predicted Trajectory')
    ax.scatter(P_pred[:,0], P_pred[:,1], P_pred[:,2], c='r', marker='o', label='Reconstructed Target Positions')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Reconstruction from Mono-Camera Images')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
