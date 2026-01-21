#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def mm_to_m_if_needed(xyz: np.ndarray, units: str) -> tuple[np.ndarray, float, str]:
    """
    Returns (xyz_m, scale, units_used).
    scale is the divisor applied to raw (e.g., 1000 if raw is mm).
    """
    if units == "m":
        return xyz.astype(float), 1.0, "m"
    if units == "mm":
        return (xyz.astype(float) / 1000.0), 1000.0, "mm"

    # auto: heuristic based on typical Vicon magnitudes
    med = float(np.median(np.abs(xyz)))
    if med > 10.0:  # likely mm (values like 1500, 900, ...)
        return (xyz.astype(float) / 1000.0), 1000.0, "mm(auto)"
    return xyz.astype(float), 1.0, "m(auto)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in", help="Input CSV (e.g., aeva_marker_bag_markers.csv)")
    ap.add_argument("--eps", type=float, default=0.03, help="DBSCAN eps in meters (default 0.03 = 3 cm)")
    ap.add_argument("--min-samples", type=int, default=20, help="DBSCAN min_samples (default 20)")
    ap.add_argument("--units", choices=["auto", "mm", "m"], default="auto",
                    help="Units of x,y,z in the input CSV. Vicon is usually mm. Default: auto.")
    ap.add_argument("--out", default="marker_cluster_centroids.csv",
                    help="Output centroid CSV written in current folder (default: marker_cluster_centroids.csv)")
    ap.add_argument("--show-points", action="store_true",
                    help="Also plot a thin scatter of all points (can be heavy).")
    args = ap.parse_args()

    csv_in = Path(args.csv_in).expanduser().resolve()
    if not csv_in.exists():
        raise SystemExit(f"Input CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)

    # Filter occlusions if column exists
    if "occluded" in df.columns:
        df = df[df["occluded"] == False].copy()

    # Extract points
    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {csv_in}")

    xyz_raw = df[["x", "y", "z"]].to_numpy()
    xyz_m, scale, units_used = mm_to_m_if_needed(xyz_raw, args.units)

    # Cluster
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels = db.fit_predict(xyz_m)

    df["cluster"] = labels

    # Drop noise (-1)
    df_in = df[df["cluster"] != -1].copy()
    xyz_in_m = xyz_m[labels != -1]

    if len(df_in) == 0:
        raise SystemExit(
            "DBSCAN found no clusters (everything labeled as noise).\n"
            "Try increasing --eps (e.g., 0.05) or decreasing --min-samples (e.g., 5)."
        )

    # Compute centroids (in meters) and counts
    centroids_m = pd.DataFrame(xyz_in_m, columns=["x_m", "y_m", "z_m"])
    centroids_m["cluster"] = df_in["cluster"].to_numpy()

    centroid_tbl = centroids_m.groupby("cluster")[["x_m", "y_m", "z_m"]].mean()
    counts = df_in.groupby("cluster").size().rename("count")

    out = centroid_tbl.join(counts).reset_index().sort_values("count", ascending=False)

    # Write output CSV in *current folder* unless user gave a path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out.to_csv(out_path, index=False)

    print(f"Loaded points (after occlusion filter): {len(df)}")
    print(f"Units detected/used for input xyz: {units_used}  (converted to meters for clustering)")
    print(f"DBSCAN params: eps={args.eps} m, min_samples={args.min_samples}")
    print(f"Found clusters: {out.shape[0]}  (noise points dropped: {(labels == -1).sum()})")
    print(f"Wrote centroid CSV: {out_path}")

    # ---- Plot centroids (XY) ----
    plt.figure()
    plt.scatter(out["x_m"], out["y_m"])
    for _, r in out.iterrows():
        plt.text(r["x_m"], r["y_m"], str(int(r["cluster"])))
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Marker cluster centroids (XY)")
    plt.grid(True)
    plt.show()

    # ---- Plot centroids (3D) ----
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(out["x_m"], out["y_m"], out["z_m"])
    for _, r in out.iterrows():
        ax.text(r["x_m"], r["y_m"], r["z_m"], str(int(r["cluster"])))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Marker cluster centroids (XYZ)")
    plt.show()

    # Optional: plot all points lightly (can be big)
    if args.show_points:
        plt.figure()
        plt.scatter(xyz_m[:, 0], xyz_m[:, 1], s=1)
        plt.scatter(out["x_m"], out["y_m"], s=50)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("All points (XY) with centroids")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()

# python3 cluster_markers_centroids.py \
# /home/asrl/Documents/Research/vicon_data_extraction/data/aeva_marker_bag_markers.csv \
# --eps 0.01   --min-samples 400   --out marker_cluster_centroids.csv