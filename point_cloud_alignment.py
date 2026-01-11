import csv
import numpy as np
import matplotlib.pyplot as plt

with open("/home/asrl/Documents/Research/vicon_data_extraction/postprocessing/vicon_markers.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        alex_calib1 = np.array([row["alex_calib1_x"], row["alex_calib1_y"], row["alex_calib1_z"]], dtype=float) / 1000.0
        alex_calib2 = np.array([row["alex_calib2_x"], row["alex_calib2_y"], row["alex_calib2_z"]], dtype=float) / 1000.0
        alex_calib3 = np.array([row["alex_calib3_x"], row["alex_calib3_y"], row["alex_calib3_z"]], dtype=float) / 1000.0
        alex_calib4 = np.array([row["alex_calib4_x"], row["alex_calib4_y"], row["alex_calib4_z"]], dtype=float) / 1000.0
        alex_calib7 = np.array([row["alex_calib7_x"], row["alex_calib7_y"], row["alex_calib7_z"]], dtype=float) / 1000.0

aeva_1 = np.array([3.3671, -0.0384, -0.3232])
aeva_2 = np.array([3.3507, 0.0646, 0.0431])
aeva_3 = np.array([3.2768, 0.3548, -0.0771])
aeva_4 = np.array([3.2155, 0.5060, 0.4129])
aeva_7 = np.array([3.2230, 0.5151, -0.3363])

# ------------------Umeyama Algorithm---------------------

# Compute centroids
vicon_pc = [alex_calib1, alex_calib2, alex_calib3, alex_calib4, alex_calib7] # a
p_vicon = np.mean(vicon_pc, axis = 0)

aeva_pc = [aeva_1, aeva_2, aeva_3, aeva_4, aeva_7] # b
p_aeva = np.mean(aeva_pc, axis = 0)

W = np.zeros((3, 3), dtype=float)
N = len(vicon_pc)
for i in range(len(vicon_pc)):
    W += (vicon_pc[i] - p_vicon).reshape(-1,1) @ (aeva_pc[i] - p_aeva).reshape(-1,1).T
W = W / float(N)

# SVD
U, S, Vt = np.linalg.svd(W)
V = Vt.T
det = np.eye(3)
det[2][2] = np.linalg.det(U) * np.linalg.det(V)

# rotation matrix and translation vector
C_aeva_vicon = U @ det @ Vt # C_ba
r_ba_a = p_vicon - C_aeva_vicon.T @ p_aeva

# transformation matrix
T_aeva_vicon = np.eye(4)
T_aeva_vicon[:3, :3] = C_aeva_vicon
T_aeva_vicon[:3, 3] = -C_aeva_vicon @ r_ba_a

# transform points
transformed_vicon_pc = []
for point in vicon_pc:
    point_h = np.append(point, 1.0)
    transformed_vicon_pc.append(T_aeva_vicon @ point_h)

# --------------------Plot Before and After--------------------------

def set_equal_aspect_3d(ax, pts):
    """
    Make 3D axes have equal scale so spheres look like spheres, etc.
    pts: (N,3) array used to set limits.
    """
    pts = np.asarray(pts, dtype=float)
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = 0.5 * np.max(xyz_max - xyz_min)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

# --- Convert lists to (N,3) arrays ---
vicon_pts = np.asarray(vicon_pc, dtype=float)                 # (N,3)
aeva_pts  = np.asarray(aeva_pc, dtype=float)                  # (N,3)

# transformed_vicon_pc is currently a list of 4D homogeneous points
vicon_tf_h = np.asarray(transformed_vicon_pc, dtype=float)    # (N,4)
vicon_tf   = vicon_tf_h[:, :3] / vicon_tf_h[:, 3:4]           # (N,3) (safe even if w!=1)

# --- Figure ---
fig = plt.figure(figsize=(12, 5))

# ===== BEFORE =====
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(vicon_pts[:, 0], vicon_pts[:, 1], vicon_pts[:, 2], s=60, label="Vicon (raw)")
ax1.scatter(aeva_pts[:, 0],  aeva_pts[:, 1],  aeva_pts[:, 2],  s=60, label="Aeva (raw)")
ax1.set_title("Before alignment")
ax1.legend(loc="best")

# ===== AFTER =====
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(vicon_tf[:, 0], vicon_tf[:, 1], vicon_tf[:, 2], s=60, label="Vicon → Aeva (transformed)")
ax2.scatter(aeva_pts[:, 0], aeva_pts[:, 1], aeva_pts[:, 2], s=60, label="Aeva (raw)")
ax2.set_title("After alignment")
ax2.legend(loc="best")

# Use shared limits based on all points (so plots are comparable)
all_before = np.vstack([vicon_pts, aeva_pts])
all_after  = np.vstack([vicon_tf,  aeva_pts])

set_equal_aspect_3d(ax1, all_before)
set_equal_aspect_3d(ax2, all_after)

plt.tight_layout()
plt.show()

# --------------------Alignment Error Report--------------------------

# Euclidean Residuals
residuals = vicon_tf - aeva_pts           # (N,3)
errors = np.linalg.norm(residuals, axis=1)

mean_err = errors.mean()
rms_err  = np.sqrt(np.mean(errors**2))
max_err  = errors.max()

print(f"Mean error: {mean_err:.4f} m")
print(f"RMS error : {rms_err:.4f} m")
print(f"Max error : {max_err:.4f} m")

# Axis-wise Errors
mean_xyz = residuals.mean(axis=0)
std_xyz  = residuals.std(axis=0)

print("Mean error [x y z]:", mean_xyz)
print("Std  error [x y z]:", std_xyz)

rmse_xyz = np.sqrt(np.mean(residuals**2, axis=0))
print("RMSE [x y z]:", rmse_xyz)
