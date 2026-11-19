#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cam_skid_slam/msg/visual_keyframe.hpp"
#include "cam_skid_slam/pcm_solver.hpp"

namespace cam_skid_slam {

namespace {

Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::msg::Pose& p) {
  Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  q.normalize();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  return T;
}

gtsam::Pose3 isometryToGtsam(const Eigen::Isometry3d& T) {
  return gtsam::Pose3(gtsam::Rot3(T.linear()), gtsam::Point3(T.translation()));
}

Eigen::Isometry3d gtsamToIsometry(const gtsam::Pose3& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = p.rotation().matrix();
  T.translation() = p.translation();
  return T;
}

}  // namespace

class MapFusionNode : public rclcpp::Node {
 public:
  MapFusionNode() : Node("cam_skid_fusion") {
    anchor_robot_ = this->declare_parameter<std::string>("anchor_robot", "robot0");
    global_desc_thresh_ = this->declare_parameter<double>("global_desc_thresh", 0.75);
    min_ransac_inliers_ = this->declare_parameter<int>("min_ransac_inliers", 12);
    icp_fitness_thresh_ = this->declare_parameter<double>("icp_fitness_thresh", 0.08);
    pcm_thresh_ = this->declare_parameter<double>("pcm_consistency_thresh", 0.40);
    map_leaf_size_ = this->declare_parameter<double>("map_leaf_size", 0.018);
    save_dir_ = this->declare_parameter<std::string>("save_dir", "");

    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);

    kf_sub_ = this->create_subscription<cam_skid_slam::msg::VisualKeyframe>(
        "/shared/keyframes", 100,
        std::bind(&MapFusionNode::keyframeCallback, this, std::placeholders::_1));

    fused_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fused_global_map", 10);
    loop_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/loop_markers", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    world_to_odom_["robot0"] = Eigen::Isometry3d::Identity();
    world_to_odom_["robot1"] = Eigen::Isometry3d::Identity();

    pub_timer_ = this->create_wall_timer(
        std::chrono::seconds(2), std::bind(&MapFusionNode::optimizeAndPublish, this));
    tf_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100), std::bind(&MapFusionNode::publishWorldTFs, this));

    RCLCPP_INFO(this->get_logger(), "CamSKiD Map Fusion initialized (anchor=%s)",
                anchor_robot_.c_str());
  }

  ~MapFusionNode() override {
    optimizeAndPublish();
    saveResults();
  }

 private:
  struct StoredKeyframe {
    std::string robot_id;
    uint32_t kf_id;
    double stamp;
    Eigen::Isometry3d odom_pose;
    Eigen::Isometry3d global_pose;
    std::vector<Eigen::Vector3d> kps_3d;
    cv::Mat descriptors;
    std::vector<float> global_desc;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  };

  unsigned char robotChar(const std::string& rid) {
    auto it = robot_chars_.find(rid);
    if (it != robot_chars_.end()) return it->second;
    unsigned char ch = static_cast<unsigned char>('a' + robot_chars_.size());
    robot_chars_[rid] = ch;
    return ch;
  }

  double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
  }

  void keyframeCallback(const cam_skid_slam::msg::VisualKeyframe::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mtx_);

    StoredKeyframe kf;
    kf.robot_id = msg->robot_id;
    kf.kf_id = msg->keyframe_id;
    kf.stamp = rclcpp::Time(msg->header.stamp).seconds();
    kf.odom_pose = poseMsgToIsometry(msg->odom_pose);
    kf.global_pose = kf.odom_pose;
    kf.global_desc = msg->global_descriptor;

    const size_t n = msg->num_keypoints;
    kf.kps_3d.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      kf.kps_3d.emplace_back(msg->keypoints_3d[3 * i + 0],
                             msg->keypoints_3d[3 * i + 1],
                             msg->keypoints_3d[3 * i + 2]);
    }
    if (n > 0) {
      kf.descriptors = cv::Mat(static_cast<int>(n), 32, CV_8UC1,
                               const_cast<uint8_t*>(msg->descriptors.data())).clone();
    }

    kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(msg->submap_cloud, *kf.cloud);

    // Track communication bandwidth
    size_t desc_bytes = msg->global_descriptor.size() * sizeof(float) +
                        msg->descriptors.size() +
                        msg->keypoints_3d.size() * sizeof(float);
    size_t cloud_bytes = msg->submap_cloud.data.size();
    total_desc_bytes_ += desc_bytes;
    total_cloud_bytes_ += cloud_bytes;

    auto& vec = keyframes_[kf.robot_id];
    if (!vec.empty() && kf.kf_id == 0) {
      // Robot restarted: clear stale keyframes and loops for a clean run
      vec.clear();
      candidate_loops_.clear();
      intra_loops_.clear();
    }
    for (const auto& existing : vec) {
      if (existing.kf_id == kf.kf_id) {
        return;
      }
    }
    vec.push_back(std::move(kf));
    robotChar(msg->robot_id);

    if (path_pubs_.find(msg->robot_id) == path_pubs_.end()) {
      path_pubs_[msg->robot_id] =
          this->create_publisher<nav_msgs::msg::Path>("/" + msg->robot_id + "/global_path", 10);
    }

    const auto& new_kf = vec.back();
    detectLoopsForKeyframe(new_kf);
    dirty_ = true;
  }

  void detectLoopsForKeyframe(const StoredKeyframe& query) {
    if (query.descriptors.empty()) return;

    // Search across all robots (both inter-robot and non-adjacent intra-robot loops)
    for (const auto& [other_id, other_kfs] : keyframes_) {
      bool is_intra = (other_id == query.robot_id);

      // Rank candidates by compact global descriptor similarity
      std::vector<std::pair<double, size_t>> candidates;
      for (size_t i = 0; i < other_kfs.size(); ++i) {
        if (is_intra && query.kf_id < other_kfs[i].kf_id + 25) continue;
        double sim = cosineSimilarity(query.global_desc, other_kfs[i].global_desc);
        if (sim >= global_desc_thresh_) {
          candidates.emplace_back(sim, i);
        }
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& x, const auto& y) { return x.first > y.first; });

      // Verify top-3 candidates geometrically
      const size_t max_check = std::min<size_t>(3, candidates.size());
      for (size_t c = 0; c < max_check; ++c) {
        const auto& target = other_kfs[candidates[c].second];
        if (target.descriptors.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        matcher_->knnMatch(query.descriptors, target.descriptors, knn, 2);

        std::vector<Eigen::Vector3d> pts_q, pts_t;
        for (const auto& m : knn) {
          if (m.size() < 2 || m[0].distance >= 0.78f * m[1].distance) continue;
          pts_q.push_back(query.kps_3d[m[0].queryIdx]);
          pts_t.push_back(target.kps_3d[m[0].trainIdx]);
        }

        Eigen::Isometry3d T_target_query;
        int inliers = 0;
        if (!ransacRigidRegistration(pts_q, pts_t, T_target_query, inliers, 0.08, 250,
                                     min_ransac_inliers_)) {
          continue;
        }

        // Fine ICP refinement on local 3D clouds
        if (!query.cloud->empty() && !target.cloud->empty()) {
          pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
          icp.setInputSource(query.cloud);
          icp.setInputTarget(target.cloud);
          icp.setMaxCorrespondenceDistance(0.25);
          icp.setMaximumIterations(25);
          pcl::PointCloud<pcl::PointXYZRGB> aligned;
          icp.align(aligned, T_target_query.matrix().cast<float>());
          if (!icp.hasConverged() || icp.getFitnessScore() > icp_fitness_thresh_) {
            continue;
          }
          Eigen::Matrix4d M = icp.getFinalTransformation().cast<double>();
          T_target_query.linear() = M.block<3, 3>(0, 0);
          T_target_query.translation() = M.block<3, 1>(0, 3);
        }

        // Store candidate loop: T_target_query maps query -> target, so rel_pose_target_to_query = T_target_query
        InterRobotLoop loop;
        loop.robot_a = target.robot_id;
        loop.robot_b = query.robot_id;
        loop.kf_a = target.kf_id;
        loop.kf_b = query.kf_id;
        loop.pose_a = target.odom_pose;
        loop.pose_b = query.odom_pose;
        loop.rel_pose_a_to_b = T_target_query;
        loop.inliers = inliers;
        if (is_intra) {
          if (inliers >= min_ransac_inliers_ + 6) {
            intra_loops_.push_back(loop);
            RCLCPP_INFO(this->get_logger(),
                        "Intra-robot loop %s[%u] <-> %s[%u] (inliers=%d)",
                        loop.robot_a.c_str(), loop.kf_a,
                        loop.robot_b.c_str(), loop.kf_b, inliers);
          }
        } else {
          candidate_loops_.push_back(loop);
          RCLCPP_INFO(this->get_logger(),
                      "Candidate loop %s[%u] <-> %s[%u] (inliers=%d)",
                      loop.robot_a.c_str(), loop.kf_a,
                      loop.robot_b.c_str(), loop.kf_b, inliers);
        }
      }
    }
  }

  void optimizeAndPublish() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!dirty_ || keyframes_.empty()) return;
    dirty_ = false;

    // Run PCM on all candidate inter-robot loops
    verified_indices_ = filterLoopsWithPCM(candidate_loops_, pcm_thresh_);
    if (!candidate_loops_.empty()) {
      RCLCPP_INFO(this->get_logger(), "PCM: %zu / %zu inter-robot loops consistent",
                  verified_indices_.size(), candidate_loops_.size());
    }

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.03, 0.03, 0.03, 0.02, 0.02, 0.02).finished());
    auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.04, 0.04, 0.04).finished());
    auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), loop_noise);

    for (const auto& [rid, kfs] : keyframes_) {
      unsigned char ch = robotChar(rid);
      for (size_t i = 0; i < kfs.size(); ++i) {
        gtsam::Key key = gtsam::Symbol(ch, kfs[i].kf_id);
        if (!initial.exists(key)) {
          initial.insert(key, isometryToGtsam(kfs[i].global_pose));
        }
        if (i == 0 && rid == anchor_robot_) {
          graph.add(gtsam::PriorFactor<gtsam::Pose3>(key, isometryToGtsam(kfs[i].odom_pose), prior_noise));
        }
        if (i > 0) {
          Eigen::Isometry3d rel = kfs[i - 1].odom_pose.inverse() * kfs[i].odom_pose;
          gtsam::Key prev_key = gtsam::Symbol(ch, kfs[i - 1].kf_id);
          graph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_key, key, isometryToGtsam(rel), odom_noise));
        }
      }
    }

    for (const auto& loop : intra_loops_) {
      gtsam::Key key_a = gtsam::Symbol(robotChar(loop.robot_a), loop.kf_a);
      gtsam::Key key_b = gtsam::Symbol(robotChar(loop.robot_b), loop.kf_b);
      if (initial.exists(key_a) && initial.exists(key_b)) {
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            key_a, key_b, isometryToGtsam(loop.rel_pose_a_to_b), robust_loop_noise));
      }
    }

    for (size_t idx : verified_indices_) {
      const auto& loop = candidate_loops_[idx];
      gtsam::Key key_a = gtsam::Symbol(robotChar(loop.robot_a), loop.kf_a);
      gtsam::Key key_b = gtsam::Symbol(robotChar(loop.robot_b), loop.kf_b);
      if (initial.exists(key_a) && initial.exists(key_b)) {
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            key_a, key_b, isometryToGtsam(loop.rel_pose_a_to_b), robust_loop_noise));
      }
    }

    // Give unanchored robots a weak prior from their first verified loop if available
    for (const auto& [rid, kfs] : keyframes_) {
      if (rid == anchor_robot_ || kfs.empty()) continue;
      bool has_loop = false;
      for (size_t idx : verified_indices_) {
        if (candidate_loops_[idx].robot_a == rid || candidate_loops_[idx].robot_b == rid) {
          has_loop = true;
          break;
        }
      }
      if (!has_loop) {
        gtsam::Key key0 = gtsam::Symbol(robotChar(rid), kfs[0].kf_id);
        auto weak_prior = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).finished());
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(key0, isometryToGtsam(kfs[0].odom_pose), weak_prior));
      }
    }

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    gtsam::Values result = optimizer.optimize();

    // Update global poses & build fused 3D map
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZRGB>());
    auto now = this->now();

    for (auto& [rid, kfs] : keyframes_) {
      unsigned char ch = robotChar(rid);
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp = now;
      path_msg.header.frame_id = "world";

      for (auto& kf : kfs) {
        gtsam::Key key = gtsam::Symbol(ch, kf.kf_id);
        if (result.exists(key)) {
          kf.global_pose = gtsamToIsometry(result.at<gtsam::Pose3>(key));
        }

        geometry_msgs::msg::PoseStamped ps;
        ps.header = path_msg.header;
        ps.pose.position.x = kf.global_pose.translation().x();
        ps.pose.position.y = kf.global_pose.translation().y();
        ps.pose.position.z = kf.global_pose.translation().z();
        Eigen::Quaterniond q(kf.global_pose.linear());
        q.normalize();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        path_msg.poses.push_back(ps);

        if (kf.cloud && !kf.cloud->empty()) {
          Eigen::Matrix4f M = kf.global_pose.matrix().cast<float>();
          for (const auto& pt : kf.cloud->points) {
            Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
            Eigen::Vector4f pw = M * p;
            pcl::PointXYZRGB gpt = pt;
            gpt.x = pw.x();
            gpt.y = pw.y();
            gpt.z = pw.z();
            global_map->push_back(gpt);
          }
        }
      }
      path_pubs_[rid]->publish(path_msg);
      if (!kfs.empty()) {
        world_to_odom_[rid] = kfs.back().global_pose * kfs.back().odom_pose.inverse();
      }
    }

    // Voxel filter & publish fused global map
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_map(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (!global_map->empty()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_map(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl::VoxelGrid<pcl::PointXYZRGB> vg;
      vg.setInputCloud(global_map);
      vg.setLeafSize(static_cast<float>(map_leaf_size_),
                     static_cast<float>(map_leaf_size_),
                     static_cast<float>(map_leaf_size_));
      vg.filter(*voxel_map);

      if (voxel_map->size() > 100) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(voxel_map);
        sor.setMeanK(16);
        sor.setStddevMulThresh(1.2);
        sor.filter(*filtered_map);
      } else {
        filtered_map = voxel_map;
      }
    }
    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*filtered_map, map_msg);
    map_msg.header.stamp = now;
    map_msg.header.frame_id = "world";
    fused_map_pub_->publish(map_msg);
    latest_fused_map_ = filtered_map;

    // Publish loop closure markers
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now;
    marker.header.frame_id = "world";
    marker.ns = "inter_robot_loops";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.03;
    marker.color.g = 1.0f;
    marker.color.a = 1.0f;
    for (size_t idx : verified_indices_) {
      const auto& loop = candidate_loops_[idx];
      const auto& p_a = keyframes_[loop.robot_a][loop.kf_a].global_pose.translation();
      const auto& p_b = keyframes_[loop.robot_b][loop.kf_b].global_pose.translation();
      geometry_msgs::msg::Point pa, pb;
      pa.x = p_a.x(); pa.y = p_a.y(); pa.z = p_a.z();
      pb.x = p_b.x(); pb.y = p_b.y(); pb.z = p_b.z();
      marker.points.push_back(pa);
      marker.points.push_back(pb);
    }
    loop_marker_pub_->publish(marker);
    saveResults();
  }

  void publishWorldTFs() {
    std::lock_guard<std::mutex> lk(mtx_);
    auto now = this->now();
    for (const auto& [rid, T] : world_to_odom_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = now;
      tf_msg.header.frame_id = "world";
      tf_msg.child_frame_id = rid + "/odom";
      tf_msg.transform.translation.x = T.translation().x();
      tf_msg.transform.translation.y = T.translation().y();
      tf_msg.transform.translation.z = T.translation().z();
      Eigen::Quaterniond q(T.linear());
      q.normalize();
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

  void saveResults() {
    if (save_dir_.empty() || keyframes_.empty()) return;
    std::filesystem::create_directories(save_dir_);

    for (const auto& [rid, kfs] : keyframes_) {
      std::string rdir = save_dir_ + "/" + rid;
      std::filesystem::create_directories(rdir);
      std::ofstream ofs(rdir + "/" + rid + "_global_path.txt");
      ofs << std::fixed << std::setprecision(6);
      for (const auto& kf : kfs) {
        Eigen::Quaterniond q(kf.global_pose.linear());
        q.normalize();
        ofs << kf.stamp << " "
            << kf.global_pose.translation().x() << " "
            << kf.global_pose.translation().y() << " "
            << kf.global_pose.translation().z() << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
      }
    }

    if (latest_fused_map_ && !latest_fused_map_->empty()) {
      pcl::io::savePCDFileBinary(save_dir_ + "/fused_global_map.pcd", *latest_fused_map_);
    }

    std::ofstream mfs(save_dir_ + "/metrics.json");
    double ratio = total_cloud_bytes_ > 0
                       ? 100.0 * (1.0 - static_cast<double>(total_desc_bytes_) /
                                            static_cast<double>(total_cloud_bytes_))
                       : 0.0;
    mfs << "{\n"
        << "  \"candidate_loops\": " << candidate_loops_.size() << ",\n"
        << "  \"verified_loops\": " << verified_indices_.size() << ",\n"
        << "  \"descriptor_bytes\": " << total_desc_bytes_ << ",\n"
        << "  \"raw_submap_cloud_bytes\": " << total_cloud_bytes_ << ",\n"
        << "  \"bandwidth_reduction_percent\": " << std::fixed << std::setprecision(2) << ratio << "\n"
        << "}\n";

    RCLCPP_INFO(this->get_logger(),
                "Saved CamSKiD-SLAM results to %s (verified loops=%zu/%zu, bw reduction=%.1f%%)",
                save_dir_.c_str(), verified_indices_.size(), candidate_loops_.size(), ratio);
  }

  std::string anchor_robot_, save_dir_;
  double global_desc_thresh_, icp_fitness_thresh_, pcm_thresh_, map_leaf_size_;
  int min_ransac_inliers_;

  cv::Ptr<cv::BFMatcher> matcher_;
  std::mutex mtx_;
  bool dirty_ = false;

  std::map<std::string, unsigned char> robot_chars_;
  std::map<std::string, Eigen::Isometry3d> world_to_odom_;
  std::map<std::string, std::vector<StoredKeyframe>> keyframes_;
  std::vector<InterRobotLoop> candidate_loops_;
  std::vector<InterRobotLoop> intra_loops_;
  std::vector<size_t> verified_indices_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr latest_fused_map_;

  size_t total_desc_bytes_ = 0;
  size_t total_cloud_bytes_ = 0;

  rclcpp::Subscription<cam_skid_slam::msg::VisualKeyframe>::SharedPtr kf_sub_;
  std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> path_pubs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_map_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr loop_marker_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr pub_timer_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
};

}  // namespace cam_skid_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cam_skid_slam::MapFusionNode>());
  rclcpp::shutdown();
  return 0;
}
