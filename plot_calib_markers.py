# make sure in ASRL conda env
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal(ax):
    """Make 3D plot axes have equal scale so spheres look like spheres."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

# (mm) python plot_calib_markers.py --csv /home/asrl/Documents/Research/vicon_data_extraction/postprocessing/vicon_markers.csv
# (m) python plot_calib_markers.py --csv /home/asrl/Documents/Research/vicon_data_extraction/postprocessing/vicon_markers.csv --convert-to-m
def main():
    parser = argparse.ArgumentParser(description="Plot Vicon marker coordinates from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV (e.g., vicon_markers.csv)")
    parser.add_argument("--units", choices=["mm", "m"], default="mm",
                        help="Interpret input units as mm (default) or m. If 'm', no scaling. If 'mm', you can optionally convert.")
    parser.add_argument("--convert-to-m", action="store_true",
                        help="If set, convert coordinates from mm to meters before plotting.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {args.csv}")

    # Identify marker base names from columns like "<name>_x", "<name>_y", "<name>_z"
    cols = list(df.columns)
    marker_names = sorted(
        {c[:-2] for c in cols if c.endswith("_x") and (c[:-2] + "_y") in cols and (c[:-2] + "_z") in cols}
    )

    if not marker_names:
        raise RuntimeError("Could not find any markers with _x/_y/_z columns.")

    scale = 1.0
    if args.convert_to_m:
        # Most Vicon exports are in mm; convert to meters
        scale = 1e-3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each marker
    for name in marker_names:
        xs = df[f"{name}_x"].to_numpy(dtype=float) * scale
        ys = df[f"{name}_y"].to_numpy(dtype=float) * scale
        zs = df[f"{name}_z"].to_numpy(dtype=float) * scale

        # If multiple rows -> trajectory; if single row -> single point
        if len(df) > 1:
            ax.plot(xs, ys, zs)  # line trajectory
            ax.scatter(xs[-1], ys[-1], zs[-1])  # last point
            ax.text(xs[-1], ys[-1], zs[-1], f" {name}", fontsize=9)
        else:
            ax.scatter(xs[0], ys[0], zs[0])
            ax.text(xs[0], ys[0], zs[0], f" {name}", fontsize=9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_units = "m" if args.convert_to_m else args.units
    ax.set_title(f"Vicon markers ({title_units})")

    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
