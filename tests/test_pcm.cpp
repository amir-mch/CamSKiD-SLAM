#include <cassert>
#include <iostream>
#include <vector>

#include "cam_skid_slam/pcm_solver.hpp"

int main() {
  using namespace cam_skid_slam;

  // 1. Test RANSAC 3D rigid registration with outliers
  Eigen::Isometry3d T_true = Eigen::Isometry3d::Identity();
  T_true.linear() = Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_true.translation() = Eigen::Vector3d(1.2, -0.5, 0.3);

  std::vector<Eigen::Vector3d> src, dst;
  for (int i = 0; i < 30; ++i) {
    Eigen::Vector3d p(0.2 * (i % 5), 0.3 * (i / 5), 1.0 + 0.1 * (i % 3));
    src.push_back(p);
    dst.push_back(T_true * p);
  }
  // Inject 6 gross outliers
  for (int i = 0; i < 6; ++i) {
    src.emplace_back(i * 0.5, -1.0, 2.0);
    dst.emplace_back(5.0, 5.0, -3.0);
  }

  Eigen::Isometry3d T_est;
  int inliers = 0;
  bool ok = ransacRigidRegistration(src, dst, T_est, inliers, 0.05, 200, 12);
  assert(ok);
  assert(inliers >= 25);
  assert((T_est.translation() - T_true.translation()).norm() < 1e-2);
  std::cout << "[PASS] RANSAC 3D registration: inliers=" << inliers
            << ", trans_err=" << (T_est.translation() - T_true.translation()).norm() << "\n";

  // 2. Test PCM outlier rejection
  Eigen::Isometry3d T_ab_true = Eigen::Isometry3d::Identity();
  T_ab_true.translation() = Eigen::Vector3d(2.0, 0.0, 0.0);

  std::vector<InterRobotLoop> loops;
  for (int i = 0; i < 4; ++i) {
    InterRobotLoop l;
    l.robot_a = "robot0";
    l.robot_b = "robot1";
    l.kf_a = static_cast<uint32_t>(i);
    l.kf_b = static_cast<uint32_t>(i);
    l.pose_a = Eigen::Isometry3d::Identity();
    l.pose_a.translation() = Eigen::Vector3d(0.5 * i, 0.0, 0.0);
    l.pose_b = T_ab_true.inverse() * l.pose_a;
    l.rel_pose_a_to_b = T_ab_true;
    l.inliers = 20;
    loops.push_back(l);
  }
  // Add 2 inconsistent false loops
  InterRobotLoop bad1 = loops[0];
  bad1.rel_pose_a_to_b.translation() = Eigen::Vector3d(-8.0, 4.0, 1.0);
  loops.push_back(bad1);

  InterRobotLoop bad2 = loops[1];
  bad2.rel_pose_a_to_b.translation() = Eigen::Vector3d(9.0, -3.0, 0.0);
  loops.push_back(bad2);

  auto verified = filterLoopsWithPCM(loops, 0.35);
  assert(verified.size() == 4);
  std::cout << "[PASS] PCM outlier rejection: kept " << verified.size()
            << " / " << loops.size() << " consistent loops\n";
  return 0;
}
