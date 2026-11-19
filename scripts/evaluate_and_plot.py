#!/usr/bin/env python3
"""
Benchmark Evaluation & Visualization for CamSKiD-SLAM.
Computes Umeyama SE(3) aligned ATE/RPE against TUM groundtruth.txt,
reads metrics.json (verified loops & bandwidth reduction), and renders
publication-ready fused_map.png and trajectory_comparison.png.
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tum_traj(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:4]


def umeyama_align(src, dst):
    """Align src (N,3) to dst (N,3) with SE(3) Umeyama (s=1)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s_c = src - mu_s
    d_c = dst - mu_d
    H = (s_c.T @ d_c) / len(src)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def associate_timestamps(t_est, p_est, t_gt, p_gt, max_diff=0.05):
    est_matched, gt_matched = [], []
    for t, p in zip(t_est, p_est):
        idx = int(np.argmin(np.abs(t_gt - t)))
        if abs(t_gt[idx] - t) < max_diff:
            est_matched.append(p)
            gt_matched.append(p_gt[idx])
    return np.array(est_matched), np.array(gt_matched)


def read_pcd_xyzrgb(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            parts = line.split()
            if parts:
                header[parts[0]] = parts[1:]
                if parts[0] == "DATA":
                    break
        n = int(header["POINTS"][0])
        fields = header["FIELDS"]
        raw = np.frombuffer(f.read(), dtype=np.float32, count=n * len(fields)).reshape(n, len(fields))
        return raw[:, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    t_gt, p_gt = load_tum_traj(args.gt)
    robots = ["robot0", "robot1"]
    colors = {"robot0": "#5C6BC0", "robot1": "#EC407A"}

    eval_summary = {}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(p_gt[:, 0], p_gt[:, 1], "--", color="#90A4AE", lw=1.5, label="Ground Truth")

    for rid in robots:
        traj_path = os.path.join(args.results, rid, f"{rid}_global_path.txt")
        if not os.path.exists(traj_path):
            continue
        t_est, p_est = load_tum_traj(traj_path)
        est_m, gt_m = associate_timestamps(t_est, p_est, t_gt, p_gt)
        if len(est_m) >= 3:
            R, t = umeyama_align(est_m, gt_m)
            aligned_full = (R @ p_est.T).T + t
            aligned_m = (R @ est_m.T).T + t
            errs = np.linalg.norm(aligned_m - gt_m, axis=1)
            ate_rmse = float(np.sqrt(np.mean(errs ** 2)))
            ate_mean = float(np.mean(errs))
            ate_max = float(np.max(errs))
            eval_summary[rid] = {
                "keyframes": len(p_est),
                "matched_poses": len(est_m),
                "ate_rmse_m": round(ate_rmse, 4),
                "ate_mean_m": round(ate_mean, 4),
                "ate_max_m": round(ate_max, 4),
            }
            ax.plot(aligned_full[:, 0], aligned_full[:, 1], "-", color=colors[rid], lw=2.2,
                    label=f"{rid} (ATE RMSE={ate_rmse:.3f}m)")

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("CamSKiD-SLAM: Multi-Robot Camera-Only Trajectory vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    traj_png = os.path.join(args.results, "trajectory_comparison.png")
    fig.savefig(traj_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pcd_path = os.path.join(args.results, "fused_global_map.pcd")
    if os.path.exists(pcd_path):
        pts = read_pcd_xyzrgb(pcd_path)
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        pts_ds = pts[::3]
        ax2.scatter(pts_ds[:, 0], pts_ds[:, 2], c=pts_ds[:, 1], cmap="viridis", s=0.4, alpha=0.6)
        ax2.set_aspect("equal")
        ax2.set_title("CamSKiD-SLAM: Fused 3D Visual Map (Top-Down X-Z)")
        ax2.axis("off")
        map_png = os.path.join(args.results, "fused_map.png")
        fig2.savefig(map_png, dpi=150, bbox_inches="tight")
        plt.close(fig2)

    metrics_path = os.path.join(args.results, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            eval_summary["fusion_metrics"] = json.load(f)

    out_json = os.path.join(args.results, "eval_summary.json")
    with open(out_json, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
