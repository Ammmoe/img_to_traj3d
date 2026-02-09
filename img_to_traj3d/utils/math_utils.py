import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as ax

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def compute_ray(K, R, T, p):
    """
    Compute normalized sight ray l_i = R_i^T * K^{-1} * p_i / || ... ||
    p: 2D pixel coordinate (u,v), assumed homogeneous coord (u,v,1)
    R: 3x3 rotation matrix
    T: 3x1 translation vector (unused here but you may want for C_i)
    """
    # Convert from camera frame (x_cam, y_cam, z_cam) to world frame (x_world, y_world, z_world)
    R_cam_to_world = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0,  -1,  0]
    ])
    
    p_h = np.array([p[0], p[1], 1.0])
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ p_h          # vector in camera coordinates
    
    # First rotate ray_cam by camera-to-world fixed rotation
    ray_cam_corrected = R_cam_to_world @ ray_cam
    
    # ray_world = R.T @ ray_cam_corrected      # rotate to world coordinate
    # ray_world = R.T @ ray_cam      # rotate to world coordinate
    
    l_i = normalize(ray_cam_corrected)

    # # debugging camera coordinate axes in world frame
    # origin = np.zeros(3)
    # axes_cam = np.eye(3)  # x, y, z unit vectors in camera frame

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i, axis in enumerate(axes_cam.T):
    #     # Apply fixed rotation
    #     axis_corrected = R_cam_to_world @ axis

    #     # Apply camera pose rotation to world
    #     axis_world = R.T @ axis_corrected

    #     ax.quiver(origin[0], origin[1], origin[2],
    #               axis_world[0], axis_world[1], axis_world[2],
    #               length=1.0, normalize=True,
    #               color=['r', 'g', 'b'][i], label=f'Axis {["x", "y", "z"][i]}')

    # ax.legend()
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Camera Axes in World Coordinates')
    # plt.show()
    
    return l_i

def construct_Theta(t, K):
    """
    Construct polynomial vector:
    Θ̃_i = [t^0, t^1, ..., t^K]^T shape (K+1,)
    """
    return np.array([t**k for k in range(K+1)])

def kronecker_eye_3_theta(Theta):
    """
    Compute A_i = I_3x3 ⊗ Θ_i^T
    Θ_i shape: (K+1,)
    Output shape: (3, 3*(K+1))
    """
    I3 = np.eye(3)
    # Kronecker product of I3 with Theta^T: 3x3 ⊗ 1x(K+1) = 3x3(K+1)
    return np.kron(I3, Theta.reshape(1, -1))

def build_matrices(L, C, Theta_list):
    """
    L: list of l_i (3,)
    C: list of C_i (3,) [camera center or translation]
    Theta_list: list of Theta_i (K+1,)
    
    Compute A (3N x 3(K+1)) and B (3N x 1) matrices stacking Eq. (12)
    A_i = (I - l_i l_i^T) * Θ_i  (Kronecker product)
    B_i = (I - l_i l_i^T) * C_i
    
    Returns stacked A, B matrices for all observations
    """
    N = len(L)
    K = Theta_list[0].shape[0] - 1
    rows = 3 * N
    cols = 3 * (K + 1)
    
    A = np.zeros((rows, cols))
    B = np.zeros((rows, 1))
    
    for i in range(N):
        l = L[i].reshape(3,1)
        C_i = C[i].reshape(3,1)
        Theta_i = Theta_list[i]
        
        V_i = np.eye(3) - l @ l.T    # 3x3
        B_i = V_i @ C_i             # 3x1
        A_i = V_i @ kronecker_eye_3_theta(Theta_i)  # 3 x 3(K+1)
        
        A[3*i:3*i+3, :] = A_i
        B[3*i:3*i+3, 0] = B_i.flatten()
        
    return A, B

def ridge_regression(A, B, beta_ls=None):
    """
    Solve beta = (A^T A - r I)^(-1) A^T B with ridge parameter r
    If beta_ls (least squares beta) is None, compute it first
    
    Calculate:
    r = t * delta_0^2 / (beta_ls^T A^T A beta_ls)
    where 
    t = rank(A) (or 3(K+1) if full rank)
    delta_0^2 = B^T (I - A (A^T A)^{-1} A^T) B / (n - t)
    """
    n, m = A.shape
    t = np.linalg.matrix_rank(A)
    
    if beta_ls is None:
        ATA_inv = np.linalg.pinv(A.T @ A)
        beta_ls = ATA_inv @ A.T @ B
    
    ATA = A.T @ A
    ATA_beta = ATA @ beta_ls
    
    # projection matrix P = A (A^T A)^{-1} A^T
    P = A @ np.linalg.pinv(ATA) @ A.T
    
    delta0_squared = (B.T @ (np.eye(n) - P) @ B)[0,0] / (n - t)
    
    r = (t * delta0_squared) / (beta_ls.T @ ATA_beta)[0,0]
    
    # ridge regression solution
    reg_matrix = ATA - r * np.eye(m)
    reg_inv = np.linalg.pinv(reg_matrix)
    beta_ridge = reg_inv @ A.T @ B
    
    return beta_ridge, r

def reconstruct_positions(beta, Theta_list):
    """
    Given beta and list of Theta_i, reconstruct 3D positions P_i = Θ_i β
    beta shape: (3(K+1), 1)
    For each Theta_i (K+1,), compute P_i (3,)
    """
    K_plus_1 = Theta_list[0].shape[0]
    N = len(Theta_list)
    P = []
    
    # Reshape beta to 3 x (K+1)
    beta_mat = beta.reshape(3, K_plus_1)
    
    for i in range(N):
        Theta_i = Theta_list[i]
        P_i = beta_mat @ Theta_i  # (3, K+1) @ (K+1,) = (3,)
        P.append(P_i)
    return np.array(P)  # shape (N, 3)

def squared_ray_error(l_pred, l_true):
    """
    Compute squared Euclidean distance between predicted and true normalized rays
    Both are shape (3,)
    """
    return np.sum((l_pred - l_true)**2)

def evaluate_model(beta, K, L, C, t_list):
    """
    For a given polynomial degree K, compute total squared sight-ray errors
    
    Steps:
    - Construct Theta for each time
    - Reconstruct 3D positions P_i
    - Compute predicted rays l_pred_i = normalized (P_i - C_i)
    - Compare with true rays L_i
    
    Return sum of squared errors
    """
    Theta_list = [construct_Theta(t, K) for t in t_list]
    P = reconstruct_positions(beta, Theta_list)  # Nx3
    
    total_error = 0.0
    N = len(L)
    for i in range(N):
        pred_ray = normalize(P[i] - C[i])
        # pred_ray = normalize(C[i] - P[i])  # Corrected direction from point to camera
        total_error += squared_ray_error(pred_ray, L[i])
    return total_error

def select_optimal_order(K_max, L, C, t_list):
    """
    Select optimal polynomial order K* in {0,...,K_max}
    by minimizing squared sight-ray errors over data
    
    Returns: K_star, beta_star
    """
    best_K = 0
    best_beta = None
    min_error = np.inf
    
    for K in range(K_max+1):
        Theta_list = [construct_Theta(t, K) for t in t_list]
        A, B = build_matrices(L, C, Theta_list)
        
        # Least squares beta
        ATA_inv = np.linalg.pinv(A.T @ A)
        beta_ls = ATA_inv @ A.T @ B
        
        # Ridge regression beta
        beta_ridge, r = ridge_regression(A, B, beta_ls)
        
        error = evaluate_model(beta_ridge, K, L, C, t_list)
        
        print(f"K={K}, ridge param r={r:.4e}, squared ray error={error:.4e}")
        
        if error < min_error:
            min_error = error
            best_K = K
            best_beta = beta_ridge
    
    # K_fixed = 20  # Fixed polynomial order for final model
    # Theta_list = [construct_Theta(t, K_fixed) for t in t_list]
    # A, B = build_matrices(L, C, Theta_list)
    
    # print(f"A shape: {A.shape}, B shape: {B.shape}")
    
    # # Least squares beta
    # ATA_inv = np.linalg.pinv(A.T @ A)
    # beta_ls = ATA_inv @ A.T @ B
    
    # # Ridge regression beta
    # beta_ridge, r = ridge_regression(A, B, beta_ls)
    
    # error = evaluate_model(beta_ridge, K_fixed, L, C, t_list)
    # print(f"Fixed K={K_fixed}, ridge param r={r:.4e}, squared ray error={error:.4e}")
    
    # return K_fixed, beta_ridge
    
    return best_K, best_beta
