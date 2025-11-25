#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # <<< NEW
from matplotlib.colors import Normalize                                      # <<< NEW
from matplotlib import cm                                                    # <<< NEW

# ============
#  PARAMETERS
# ============

CSV_PATH = 'vicon_poses.csv'   # output from extract_vicon_poses.py

# Box half-extents in BODY frame (meters)
HX = 0.40
HY = 0.30
HZ = 0.50

# LiDAR pose in WORLD frame (assume Vicon/world frame)
LIDAR_POS_W = np.array([0.0, 0.0, 0.0])
LIDAR_VEL_W = np.array([0.0, 0.0, 0.0])  # unused for now

# Downsample factor for animation frames (1 = all frames)
ANIM_STEP = 1

# Speed factor for playback: 1.0 = real time
SPEED_FACTOR = 1.0

# Visualization toggles
SHOW_TRAJECTORY   = False   # full path of the box
SHOW_GHOST_TRAIL  = False   # short trail behind box
GHOST_FRAMES      = 20
SHOW_VELOCITY_VEC = False   # velocity arrow at box centroid
VEL_SCALE         = 1.0
SHOW_ANG_VEL_VEC  = False   # draw angular velocity arrow at box centroid
ANG_VEL_SCALE     = 0.5     # tuning factor for arrow length

SHOW_BOX_FRAME = False
BOX_AXIS_LEN   = 0.6
FACE_COLOR_MODE = "clear" # "default", "clear", "brown", "grey"

SHOW_LIDAR_AXES = True      # draw XYZ axes at LiDAR origin
LIDAR_AXIS_LEN  = 0.5

# LiDAR intrinsics / scanning model
H_FOV = np.deg2rad(360.0)   # horizontal FOV (full 360)
V_FOV = np.deg2rad(360.0)   # vertical FOV
H_RES = np.deg2rad(0.5)     # horizontal angular resolution (only used if you quantize)
V_RES = np.deg2rad(1.0)     # vertical angular resolution

# Number of samples per box face (for surface discretization)
BOX_SAMPLES_U = 1
BOX_SAMPLES_V = 1

SHOW_LIDAR_POINTS   = True    # draw simulated LiDAR returns
LIDAR_POINT_SIZE    = 20
COLOR_BY_DOPPLER    = True    # color points by radial velocity
GLOBAL_VMAX = 0.5  # e.g. expect ~±0.5 m/s

SAVE_MP4   = False
MP4_NAME   = 'box_motion.mp4'

# ===================
#  HELPER FUNCTIONS
# ===================

def compute_velocities(t, p, R_list):
    """
    Compute linear and angular velocity of the box in the WORLD frame.

    Inputs:
        t       : (N,) time stamps
        p       : (N, 3) positions of the box centroid in WORLD frame
        R_list  : (N, 3, 3) rotation matrices R_wb (BODY -> WORLD)

    Returns:
        v_world     : (N, 3) linear velocity in WORLD frame
        omega_world : (N, 3) angular velocity in WORLD frame
    """
    N = len(t)

    # -----------------------
    # Linear velocity (WORLD)
    # -----------------------
    v_world = np.zeros_like(p)

    dt = np.diff(t)

    # central differences for interior points, forward/backward for ends
    # v_i ≈ (p_{i+1} - p_{i-1}) / (t_{i+1} - t_{i-1})
    for i in range(1, N - 1):
        dt_c = t[i+1] - t[i-1]
        v_world[i] = (p[i+1] - p[i-1]) / dt_c

    # endpoints: copy neighbor
    v_world[0]  = v_world[1]
    v_world[-1] = v_world[-2]

    # -----------------------------------------
    # Angular velocity (BODY -> then WORLD)
    # -----------------------------------------
    # We first compute ω in BODY frame from:
    #   R_wb^T Ṙ_wb = [ω_b]×
    #
    # Then convert to WORLD frame via:
    #   ω_w = R_wb ω_b
    #
    omega_world = np.zeros_like(p)

    for i in range(1, N - 1):
        dt_c = t[i+1] - t[i-1]
        R_prev = R_list[i-1]
        R_next = R_list[i+1]
        R_mid  = R_list[i]

        # approximate time derivative of R using central difference
        Rdot = (R_next - R_prev) / dt_c

        # BODY-frame skew matrix: [ω_b]× = R^T Ṙ
        skew_b = R_mid.T @ Rdot

        # Extract ω_b from skew matrix
        wx_b = 0.5 * (skew_b[2, 1] - skew_b[1, 2])
        wy_b = 0.5 * (skew_b[0, 2] - skew_b[2, 0])
        wz_b = 0.5 * (skew_b[1, 0] - skew_b[0, 1])
        omega_body = np.array([wx_b, wy_b, wz_b])

        # Convert to WORLD frame: ω_w = R_wb ω_b
        omega_world[i] = R_mid @ omega_body

    # endpoints: copy neighbors
    omega_world[0]  = omega_world[1]
    omega_world[-1] = omega_world[-2]

    return v_world, omega_world


def box_corners_body():
    corners = []
    for sx in [-HX, HX]:
        for sy in [-HY, HY]:
            for sz in [-HZ, HZ]:
                corners.append([sx, sy, sz])
    return np.array(corners)

def box_faces_from_corners(c):
    faces = []
    faces.append([c[0], c[1], c[3], c[2]])  # -X
    faces.append([c[4], c[5], c[7], c[6]])  # +X
    faces.append([c[0], c[1], c[5], c[4]])  # -Y
    faces.append([c[2], c[3], c[7], c[6]])  # +Y
    faces.append([c[0], c[2], c[6], c[4]])  # -Z
    faces.append([c[1], c[3], c[7], c[5]])  # +Z
    return faces

def sample_box_surface_body(nu=BOX_SAMPLES_U, nv=BOX_SAMPLES_V):
    """
    Return a (M x 3) array of points sampled on the box surface in BODY frame.
    We sample each of the 6 faces on a regular grid.
    Order: (-X, +X, -Y, +Y, -Z, +Z), each face has nu*nv points.
    """
    us_x = np.linspace(-HY, HY, nu)
    vs_x = np.linspace(-HZ, HZ, nv)
    us_y = np.linspace(-HX, HX, nu)
    vs_y = np.linspace(-HZ, HZ, nv)
    us_z = np.linspace(-HX, HX, nu)
    vs_z = np.linspace(-HY, HY, nv)

    pts = []

    # Faces at x = ±HX
    for sx in [-HX, HX]:
        U, V = np.meshgrid(us_x, vs_x, indexing='ij')
        pts.append(np.column_stack([np.full(U.size, sx), U.ravel(), V.ravel()]))

    # Faces at y = ±HY
    for sy in [-HY, HY]:
        U, V = np.meshgrid(us_y, vs_y, indexing='ij')
        pts.append(np.column_stack([U.ravel(), np.full(U.size, sy), V.ravel()]))

    # Faces at z = ±HZ
    for sz in [-HZ, HZ]:
        U, V = np.meshgrid(us_z, vs_z, indexing='ij')
        pts.append(np.column_stack([U.ravel(), V.ravel(), np.full(U.size, sz)]))

    return np.vstack(pts)

def simulate_lidar_scan_for_pose(R_wb, p_wb, surface_b):
    """
    Simple MVP LiDAR simulation:
    - Sample all 6 faces in BODY frame (surface_b).
    - For each face:
        * Check if the face normal points toward the LiDAR.
        * If not, drop that entire face.
    - For visible faces:
        * Transform their points to WORLD frame.
        * Apply a simple FOV filter.
    Returns:
        pts_w_visible : (K x 3) visible LiDAR points in WORLD frame.
    """
    num_per_face = BOX_SAMPLES_U * BOX_SAMPLES_V

    # Face centers and outward normals in BODY frame
    face_centers_b = [
        np.array([-HX, 0.0, 0.0]),  # -X
        np.array([ HX, 0.0, 0.0]),  # +X
        np.array([0.0, -HY, 0.0]),  # -Y
        np.array([0.0,  HY, 0.0]),  # +Y
        np.array([0.0, 0.0, -HZ]),  # -Z
        np.array([0.0, 0.0,  HZ]),  # +Z
    ]
    face_normals_b = [
        np.array([-1.0, 0.0, 0.0]),  # -X
        np.array([ 1.0, 0.0, 0.0]),  # +X
        np.array([0.0, -1.0, 0.0]),  # -Y
        np.array([0.0,  1.0, 0.0]),  # +Y
        np.array([0.0, 0.0, -1.0]),  # -Z
        np.array([0.0, 0.0,  1.0]),  # +Z
    ]

    pts_visible_world = []
    offset = 0

    for center_b, normal_b in zip(face_centers_b, face_normals_b):
        # Points on this face in BODY frame
        pts_face_b = surface_b[offset:offset + num_per_face]
        offset += num_per_face

        # Transform center and normal to WORLD
        center_w = R_wb @ center_b + p_wb
        normal_w = R_wb @ normal_b

        # Vector from face center to LiDAR
        v_face_to_lidar = LIDAR_POS_W - center_w

        # Check if face is facing the LiDAR (angle < 90 deg)
        if normal_w @ v_face_to_lidar <= 0.0:
            # Back-facing or exactly perpendicular -> not visible
            continue

        # Transform face points to WORLD
        pts_face_w = (R_wb @ pts_face_b.T).T + p_wb

        # FOV filter (LiDAR at origin)
        x = pts_face_w[:, 0]
        y = pts_face_w[:, 1]
        z = pts_face_w[:, 2]

        rng = np.linalg.norm(pts_face_w - LIDAR_POS_W, axis=1)
        az  = np.arctan2(y - LIDAR_POS_W[1], x - LIDAR_POS_W[0])
        el  = np.arctan2(z - LIDAR_POS_W[2],
                         np.sqrt((x - LIDAR_POS_W[0])**2 + (y - LIDAR_POS_W[1])**2))

        mask = (
            (np.abs(az) <= H_FOV / 2.0) &
            (np.abs(el) <= V_FOV / 2.0) &
            (rng > 0.0)
        )

        pts_face_visible = pts_face_w[mask]
        if pts_face_visible.size > 0:
            pts_visible_world.append(pts_face_visible)

    if not pts_visible_world:
        return np.empty((0, 3))

    return np.vstack(pts_visible_world)

# =====================
#  MAIN OFFLINE SCRIPT
# =====================

def main():
    # ---- Load CSV ----
    df = pd.read_csv(CSV_PATH)
    t = df['t'].values
    p = df[['px', 'py', 'pz']].values
    q = df[['qx', 'qy', 'qz', 'qw']].values

    R_list = R.from_quat(q).as_matrix()

    v_world, omega_world = compute_velocities(t, p, R_list)

    print("Loaded", len(t), "poses.")
    print("Example v [m/s]:", v_world[0])
    print("Example w [rad/s]:", omega_world[0])

    corners_b = box_corners_body()
    surface_b = sample_box_surface_body()

    # ============
    #  ANIMATION
    # ============

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # On-screen legend for controls
    controls_text = (
        "Controls:\n"
        "  Space / P : Pause / Resume\n"
        "  ←         : Step backward (paused)\n"
        "  →         : Step forward (paused)\n"
        "  R         : Reset to first frame\n"
    )

    fig.text(
        0.02, 0.95, controls_text,
        fontsize=10,
        verticalalignment='top',
        family='monospace',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    margin = max(HX, HY, HZ) * 2.0
    x_min, x_max = p[:, 0].min() - margin, p[:, 0].max() + margin
    y_min, y_max = p[:, 1].min() - margin, p[:, 1].max() + margin
    z_min, z_max = p[:, 2].min() - margin, p[:, 2].max() + margin

    # make sure origin (LiDAR) is visible
    x_min = min(x_min, 0.0); x_max = max(x_max, 0.0)
    y_min = min(y_min, 0.0); y_max = max(y_max, 0.0)
    z_min = min(z_min, 0.0); z_max = max(z_max, 0.0)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Vicon Box Motion')

    ax.view_init(elev=20, azim=-60)

    # LiDAR at origin (axes only, no marker)
    if SHOW_LIDAR_AXES:
        ax.quiver(*LIDAR_POS_W, LIDAR_AXIS_LEN, 0, 0,
                  color='r', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*LIDAR_POS_W, 0, LIDAR_AXIS_LEN, 0,
                  color='g', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*LIDAR_POS_W, 0, 0, LIDAR_AXIS_LEN,
                  color='b', arrow_length_ratio=0.2, linewidth=2)

    # Initial box (with distinct face colors + black edges)
    R_wb0 = R_list[0]
    p_wb0 = p[0]
    corners_w0 = (R_wb0 @ corners_b.T).T + p_wb0
    faces0 = box_faces_from_corners(corners_w0)

    if FACE_COLOR_MODE == "default":
        # Original pretty colours (6 distinct faces)
        face_colors = ['#FF9999', '#99FF99', '#9999FF',
                       '#FFCC66', '#66CCFF', '#CC99FF']

    elif FACE_COLOR_MODE == "clear":
        # Transparent faces — keep edge lines
        face_colors = ['#FFFFFF00'] * 6   # RGBA, 00 means fully transparent

    elif FACE_COLOR_MODE == "brown":
        # Wooden / cardboard box look
        face_colors = ['#8B4513'] * 6     # saddle brown

    elif FACE_COLOR_MODE == "grey":
        # Neutral grey box (good for Doppler visualization)
        face_colors = ['#888888'] * 6     # medium grey

    box_collection = Poly3DCollection(
        faces0,
        facecolors=face_colors,
        edgecolors='k',
        linewidths=1,
        alpha=0.7
    )
    ax.add_collection3d(box_collection)

    # Optional full trajectory
    traj_line = None
    if SHOW_TRAJECTORY:
        traj_line, = ax.plot(p[:, 0], p[:, 1], p[:, 2],
                             linestyle='--', color='0.5')

    # Ghost trail (short recent path)
    ghost_line = None
    if SHOW_GHOST_TRAIL:
        ghost_line, = ax.plot([], [], [], 'k-', linewidth=1)

    # Dynamic artists
    vel_quiver = None
    ang_vel_quiver = None
    box_axis_x = box_axis_y = box_axis_z = None

    # Simulated LiDAR point cloud (scatter)
    lidar_scatter = None
    line_collection = None
    range_texts = []   # store ax.text handles for each LiDAR point label
    norm = Normalize(-GLOBAL_VMAX, GLOBAL_VMAX)


    if SHOW_LIDAR_POINTS:
        if COLOR_BY_DOPPLER:
            lidar_scatter = ax.scatter(
                [], [], [],
                s=LIDAR_POINT_SIZE,
                c=[],
                cmap='coolwarm',
                alpha=0.9
            )
        else:
            lidar_scatter = ax.scatter(
                [], [], [],
                s=LIDAR_POINT_SIZE,
                c='k',
                alpha=0.8
            )

        # Fix colour scale range for Doppler visualization
        lidar_scatter.set_clim(-GLOBAL_VMAX, GLOBAL_VMAX)

        # Line collection for dotted rays (LiDAR → points)
        line_collection = Line3DCollection(
            np.zeros((0, 2, 3)),   # valid empty
            linestyles='dotted',
            linewidths=1.0
        )
        ax.add_collection3d(line_collection)


    # Frame indices
    frame_indices = np.arange(0, len(t), ANIM_STEP)

    # Real-time interval
    dt_mean = np.mean(np.diff(t))
    dt_phys = dt_mean * ANIM_STEP
    interval_ms = dt_phys * 1000.0 / SPEED_FACTOR

    print(f"Mean dt: {dt_mean:.6f} s")
    print(f"Physical dt per frame: {dt_phys:.6f} s")
    print(f"Animation interval: {interval_ms:.2f} ms")

    # Playback state
    paused = False
    current_frame_idx = 0  # index into frame_indices

    def update(frame_idx):
        nonlocal vel_quiver, ang_vel_quiver, box_axis_x, box_axis_y, box_axis_z
        nonlocal lidar_scatter, current_frame_idx, line_collection, range_texts

        # Track current frame for stepping logic
        current_frame_idx = frame_idx

        i = frame_indices[frame_idx]
        R_wb = R_list[i]
        p_wb = p[i]

        # Box mesh
        corners_w = (R_wb @ corners_b.T).T + p_wb
        faces = box_faces_from_corners(corners_w)
        box_collection.set_verts(faces)

        artists = [box_collection]

        # Ghost trail
        if SHOW_GHOST_TRAIL and ghost_line is not None:
            j_start = max(0, frame_idx - GHOST_FRAMES + 1)
            js = frame_indices[j_start:frame_idx + 1]
            trail_pts = p[js]
            ghost_line.set_data(trail_pts[:, 0], trail_pts[:, 1])
            ghost_line.set_3d_properties(trail_pts[:, 2])
            artists.append(ghost_line)

        # Velocity vector at box centroid
        if SHOW_VELOCITY_VEC:
            if vel_quiver is not None:
                vel_quiver.remove()
            v = v_world[i] * VEL_SCALE
            vel_quiver = ax.quiver(
                p_wb[0], p_wb[1], p_wb[2],
                v[0], v[1], v[2],
                color='m', linewidth=2
            )
            artists.append(vel_quiver)

        # Angular velocity vector at box centroid
        if SHOW_ANG_VEL_VEC:
            if ang_vel_quiver is not None:
                ang_vel_quiver.remove()
            w = omega_world[i] * ANG_VEL_SCALE
            ang_vel_quiver = ax.quiver(
                p_wb[0], p_wb[1], p_wb[2],
                w[0], w[1], w[2],
                color='y', linewidth=2
            )
            artists.append(ang_vel_quiver)

        # Box BODY coordinate frame
        if SHOW_BOX_FRAME:
            for q in (box_axis_x, box_axis_y, box_axis_z):
                if q is not None:
                    q.remove()

            x_axis = R_wb @ np.array([1, 0, 0])
            y_axis = R_wb @ np.array([0, 1, 0])
            z_axis = R_wb @ np.array([0, 0, 1])

            box_axis_x = ax.quiver(
                p_wb[0], p_wb[1], p_wb[2],
                x_axis[0] * BOX_AXIS_LEN,
                x_axis[1] * BOX_AXIS_LEN,
                x_axis[2] * BOX_AXIS_LEN,
                color='r', linewidth=2
            )
            box_axis_y = ax.quiver(
                p_wb[0], p_wb[1], p_wb[2],
                y_axis[0] * BOX_AXIS_LEN,
                y_axis[1] * BOX_AXIS_LEN,
                y_axis[2] * BOX_AXIS_LEN,
                color='g', linewidth=2
            )
            box_axis_z = ax.quiver(
                p_wb[0], p_wb[1], p_wb[2],
                z_axis[0] * BOX_AXIS_LEN,
                z_axis[1] * BOX_AXIS_LEN,
                z_axis[2] * BOX_AXIS_LEN,
                color='b', linewidth=2
            )

            artists.extend([box_axis_x, box_axis_y, box_axis_z])

        # Simulated LiDAR returns + Doppler radial velocity
        if SHOW_LIDAR_POINTS and lidar_scatter is not None:
            pts_w = simulate_lidar_scan_for_pose(R_wb, p_wb, surface_b)

            if pts_w.size > 0:
                # LiDAR-frame positions (same as WORLD here)
                p_i = pts_w - LIDAR_POS_W      # (N, 3)
                n_i = p_i / np.linalg.norm(p_i, axis=1, keepdims=True)  # LOS unit vectors

                # Centroid position relative to LiDAR
                p_o = p_wb - LIDAR_POS_W       # (3,)

                v   = v_world[i]               # (3,)
                w   = omega_world[i]           # (3,)

                # n_i^T v term
                term1 = n_i @ v                # (N,)

                # (p_o × n_i)^T w term
                cross_pn = np.cross(p_o, n_i)  # (N, 3), broadcasted cross
                term2 = cross_pn @ w           # (N,)

                v_r = term1 - term2            # (N,) radial Doppler velocities

                # Update scatter positions
                lidar_scatter._offsets3d = (pts_w[:, 0], pts_w[:, 1], pts_w[:, 2])

                if COLOR_BY_DOPPLER:
                    lidar_scatter.set_array(v_r)
                else:
                    # if not coloring by Doppler, keep black
                    lidar_scatter.set_array(None)

                # --- Dotted lines from LiDAR to each point (same color as point) ---
                if line_collection is not None:
                    # segments: shape (N, 2, 3), each is [LIDAR_POS_W, point]
                    segments = np.stack(
                        [np.repeat(LIDAR_POS_W[None, :], pts_w.shape[0], axis=0),
                         pts_w],
                        axis=1
                    )
                    line_collection.set_segments(segments)

                    if COLOR_BY_DOPPLER:
                        colors = cm.get_cmap('coolwarm')(norm(v_r))
                        line_collection.set_colors(colors)
                    else:
                        line_collection.set_colors('k')

                
                # --- Range text labels at midpoint of LOS line ---
                for txt in range_texts:
                    txt.remove()
                range_texts.clear()

                ranges = np.linalg.norm(pts_w - LIDAR_POS_W, axis=1)

                for P, r_val, v_r_val in zip(pts_w, ranges, v_r):

                    # label color: solid red or blue
                    if COLOR_BY_DOPPLER:
                        color = 'red' if v_r_val > 0 else 'blue'
                    else:
                        color = 'k'

                    # midpoint of LOS line
                    mid = (P + LIDAR_POS_W) / 2.0

                    # small offset perpendicular to line
                    direction = P - LIDAR_POS_W
                    norm_d = np.linalg.norm(direction)
                    if norm_d > 1e-6:
                        tmp = np.array([1,0,0]) if abs(direction[0]) < 0.9 else np.array([0,1,0])
                        perp = np.cross(direction, tmp)
                        perp /= np.linalg.norm(perp)
                        perp *= 0.05  # 5 cm offset
                    else:
                        perp = np.zeros(3)

                    label_pos = mid + perp

                    txt = ax.text(
                        label_pos[0], label_pos[1], label_pos[2],
                        f"{r_val:.2f}m", # f"{v_r_val:.2f} m/s" OR f"{r_val:.2f}m\n{v_r_val:.2f}m/s"
                        fontsize=9,
                        color=color,
                        weight='bold'
                    )
                    range_texts.append(txt)


            else:
                lidar_scatter._offsets3d = ([], [], [])
                if COLOR_BY_DOPPLER:
                    lidar_scatter.set_array([])

                if line_collection is not None:
                    line_collection.set_segments([])

            artists.append(lidar_scatter)
            if line_collection is not None:
                artists.append(line_collection)

        if traj_line is not None:
            artists.append(traj_line)

        ax.set_title(f'Vicon Box Motion (t = {t[i]:.3f} s)')
        return artists

    # Keyboard controls: pause/resume, step, reset
    def on_key(event):
        nonlocal paused, anim, current_frame_idx

        if event.key in (' ', 'p'):
            paused = not paused
            if paused:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            print("Paused" if paused else "Resumed")

        elif event.key == 'r':
            # Reset to first frame and pause
            paused = True
            anim.event_source.stop()
            current_frame_idx = 0
            update(current_frame_idx)
            fig.canvas.draw_idle()
            print("Reset to first frame")

        elif event.key == 'left':
            # Step one frame backward (only when paused)
            if paused:
                current_frame_idx = (current_frame_idx - 1) % len(frame_indices)
                update(current_frame_idx)
                fig.canvas.draw_idle()
                print(f"Step back to frame {current_frame_idx}")

        elif event.key == 'right':
            # Step one frame forward (only when paused)
            if paused:
                current_frame_idx = (current_frame_idx + 1) % len(frame_indices)
                update(current_frame_idx)
                fig.canvas.draw_idle()
                print(f"Step forward to frame {current_frame_idx}")

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False
    )

    # Connect keyboard handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Radial Velocity Colour Bar
    if SHOW_LIDAR_POINTS and COLOR_BY_DOPPLER and lidar_scatter is not None:
        fig.colorbar(lidar_scatter, ax=ax, label='Radial velocity [m/s]')

    plt.tight_layout()

    if SAVE_MP4:
        fps = max(1, int(round(1.0 / dt_phys)))
        print(f"Saving MP4 to {MP4_NAME} at ~{fps} fps...")
        anim.save(MP4_NAME, fps=fps, dpi=150)
        print("Saved.")

    plt.show()

if __name__ == '__main__':
    main()
