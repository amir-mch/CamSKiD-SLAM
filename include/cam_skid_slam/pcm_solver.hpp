#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>

namespace cam_skid_slam {

/**
 * @brief Estimate SE(3) transform (R, t) aligning src_pts -> dst_pts using SVD (Umeyama/Arun).
 */
inline Eigen::Isometry3d estimateRigidTransformSVD(
    const std::vector<Eigen::Vector3d>& src_pts,
    const std::vector<Eigen::Vector3d>& dst_pts) {
  Eigen::Vector3d mean_src = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_dst = Eigen::Vector3d::Zero();
  const size_t n = src_pts.size();
  for (size_t i = 0; i < n; ++i) {
    mean_src += src_pts[i];
    mean_dst += dst_pts[i];
  }
  mean_src /= static_cast<double>(n);
  mean_dst /= static_cast<double>(n);

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < n; ++i) {
    H += (src_pts[i] - mean_src) * (dst_pts[i] - mean_dst).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d R = V * U.transpose();
  if (R.determinant() < 0.0) {
    V.col(2) *= -1.0;
    R = V * U.transpose();
  }
  Eigen::Vector3d t = mean_dst - R * mean_src;

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = t;
  return T;
}

/**
 * @brief RANSAC 3D-to-3D rigid registration from matched feature points.
 * @return True if enough inliers are found.
 */
inline bool ransacRigidRegistration(
    const std::vector<Eigen::Vector3d>& src_pts,
    const std::vector<Eigen::Vector3d>& dst_pts,
    Eigen::Isometry3d& best_T,
    int& best_inliers,
    double inlier_thresh = 0.08,
    int max_iters = 200,
    int min_inliers = 12) {
  const int n = static_cast<int>(src_pts.size());
  if (n < min_inliers) {
    return false;
  }

  best_inliers = 0;
  best_T = Eigen::Isometry3d::Identity();
  std::vector<int> best_inlier_idx;

  unsigned int seed = 42;
  for (int iter = 0; iter < max_iters; ++iter) {
    int i0 = rand_r(&seed) % n;
    int i1 = rand_r(&seed) % n;
    int i2 = rand_r(&seed) % n;
    if (i0 == i1 || i1 == i2 || i0 == i2) continue;

    std::vector<Eigen::Vector3d> s3 = {src_pts[i0], src_pts[i1], src_pts[i2]};
    std::vector<Eigen::Vector3d> d3 = {dst_pts[i0], dst_pts[i1], dst_pts[i2]};
    Eigen::Isometry3d T = estimateRigidTransformSVD(s3, d3);

    int inliers = 0;
    std::vector<int> idx;
    idx.reserve(n);
    for (int i = 0; i < n; ++i) {
      double err = (T * src_pts[i] - dst_pts[i]).norm();
      if (err < inlier_thresh) {
        ++inliers;
        idx.push_back(i);
      }
    }
    if (inliers > best_inliers) {
      best_inliers = inliers;
      best_T = T;
      best_inlier_idx = std::move(idx);
    }
  }

  if (best_inliers < min_inliers) {
    return false;
  }

  std::vector<Eigen::Vector3d> inlier_src, inlier_dst;
  inlier_src.reserve(best_inlier_idx.size());
  inlier_dst.reserve(best_inlier_idx.size());
  for (int idx : best_inlier_idx) {
    inlier_src.push_back(src_pts[idx]);
    inlier_dst.push_back(dst_pts[idx]);
  }
  best_T = estimateRigidTransformSVD(inlier_src, inlier_dst);
  return true;
}

struct InterRobotLoop {
  std::string robot_a;
  std::string robot_b;
  uint32_t kf_a;
  uint32_t kf_b;
  Eigen::Isometry3d pose_a;
  Eigen::Isometry3d pose_b;
  Eigen::Isometry3d rel_pose_a_to_b;  // T_a_b: transforms point in b frame to a frame
  int inliers;
};

/**
 * @brief Pairwise Consistency Maximization (PCM) outlier rejection.
 * Checks pairwise SE(3) consistency between candidate inter-robot loops and
 * finds the maximum mutually consistent subset via greedy clique extraction.
 */
inline std::vector<size_t> filterLoopsWithPCM(
    const std::vector<InterRobotLoop>& loops,
    double consistency_thresh = 0.35) {
  const size_t m = loops.size();
  if (m == 0) return {};
  if (m == 1) return {0};

  // Build adjacency matrix of mutually consistent loop pairs
  std::vector<std::vector<bool>> adj(m, std::vector<bool>(m, false));
  std::vector<size_t> deg(m, 0);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = i + 1; j < m; ++j) {
      if (loops[i].robot_a != loops[j].robot_a || loops[i].robot_b != loops[j].robot_b) {
        continue;
      }
      // Consistency check:
      // T_a_i^{-1} * T_a_j  should match  T_ab_i * (T_b_i^{-1} * T_b_j) * T_ab_j^{-1}
      Eigen::Isometry3d odom_a = loops[i].pose_a.inverse() * loops[j].pose_a;
      Eigen::Isometry3d odom_b = loops[i].pose_b.inverse() * loops[j].pose_b;
      Eigen::Isometry3d loop_cycle =
          odom_a.inverse() * loops[i].rel_pose_a_to_b * odom_b * loops[j].rel_pose_a_to_b.inverse();

      double trans_err = loop_cycle.translation().norm();
      Eigen::AngleAxisd aa(loop_cycle.linear());
      double rot_err = std::abs(aa.angle());
      double combined_err = trans_err + 0.5 * rot_err;

      if (combined_err < consistency_thresh) {
        adj[i][j] = adj[j][i] = true;
        deg[i]++;
        deg[j]++;
      }
    }
  }

  // Greedy maximum-clique approximation seeded from highest-degree node
  size_t best_seed = 0;
  for (size_t i = 1; i < m; ++i) {
    if (deg[i] > deg[best_seed]) best_seed = i;
  }

  std::vector<size_t> clique = {best_seed};
  for (size_t i = 0; i < m; ++i) {
    if (i == best_seed) continue;
    bool connected_to_all = true;
    for (size_t member : clique) {
      if (!adj[i][member]) {
        connected_to_all = false;
        break;
      }
    }
    if (connected_to_all) {
      clique.push_back(i);
    }
  }
  return clique;
}

}  // namespace cam_skid_slam
