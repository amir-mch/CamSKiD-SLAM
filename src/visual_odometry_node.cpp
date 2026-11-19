#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cam_skid_slam/msg/visual_keyframe.hpp"

namespace cam_skid_slam {

class VisualOdometryNode : public rclcpp::Node {
 public:
  VisualOdometryNode() : Node("cam_skid_vo") {
    robot_id_ = this->declare_parameter<std::string>("robot_id", "robot0");
    fx_ = this->declare_parameter<double>("fx", 517.306408);
    fy_ = this->declare_parameter<double>("fy", 516.469215);
    cx_ = this->declare_parameter<double>("cx", 318.643040);
    cy_ = this->declare_parameter<double>("cy", 255.313989);
    depth_scale_ = this->declare_parameter<double>("depth_scale", 1.0);
    max_depth_ = this->declare_parameter<double>("max_depth", 5.0);
    min_depth_ = this->declare_parameter<double>("min_depth", 0.2);
    num_features_ = this->declare_parameter<int>("num_features", 800);
    max_kf_features_ = this->declare_parameter<int>("max_keyframe_features", 260);
    kf_dist_thresh_ = this->declare_parameter<double>("keyframe_dist_thresh", 0.12);
    kf_angle_thresh_ = this->declare_parameter<double>("keyframe_angle_thresh", 0.15);
    voxel_leaf_ = this->declare_parameter<double>("voxel_leaf_size", 0.025);
    save_dir_ = this->declare_parameter<std::string>("save_dir", "");

    k1_ = this->declare_parameter<double>("k1", 0.262383);
    k2_ = this->declare_parameter<double>("k2", -0.953104);
    p1_ = this->declare_parameter<double>("p1", -0.005358);
    p2_ = this->declare_parameter<double>("p2", 0.002628);
    k3_ = this->declare_parameter<double>("k3", 1.163314);

    K_ = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    dist_coeffs_ = (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
    orb_ = cv::ORB::create(num_features_, 1.2f, 8);
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);

    auto qos = rclcpp::QoS(10);
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/rgb/image_raw", qos,
        std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));
    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/depth/image_raw", qos,
        std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 50);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("path", 10);
    kf_pub_ = this->create_publisher<cam_skid_slam::msg::VisualKeyframe>("/shared/keyframes", 50);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    path_msg_.header.frame_id = robot_id_ + "/odom";

    RCLCPP_INFO(this->get_logger(), "[%s] Visual Odometry initialized (fx=%.1f, fy=%.1f)",
                robot_id_.c_str(), fx_, fy_);
  }

  ~VisualOdometryNode() override {
    if (!save_dir_.empty() && !traj_log_.empty()) {
      std::string path = save_dir_ + "/" + robot_id_ + "_odom_path.txt";
      std::ofstream ofs(path);
      if (ofs.is_open()) {
        ofs << std::fixed << std::setprecision(6);
        for (const auto& entry : traj_log_) {
          ofs << entry.stamp << " "
              << entry.pose.translation().x() << " "
              << entry.pose.translation().y() << " "
              << entry.pose.translation().z() << " ";
          Eigen::Quaterniond q(entry.pose.linear());
          ofs << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        RCLCPP_INFO(this->get_logger(), "[%s] Saved %zu odom poses to %s",
                    robot_id_.c_str(), traj_log_.size(), path.c_str());
      }
    }
  }

 private:
  struct TimedImage {
    double stamp;
    cv::Mat image;
    builtin_interfaces::msg::Time ros_stamp;
  };

  struct TrajEntry {
    double stamp;
    Eigen::Isometry3d pose;
  };

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cv_bridge depth error: %s", e.what());
      return;
    }
    double t = rclcpp::Time(msg->header.stamp).seconds();
    std::lock_guard<std::mutex> lk(buf_mtx_);
    depth_buf_.push_back({t, cv_ptr->image.clone(), msg->header.stamp});
    if (depth_buf_.size() > 30) depth_buf_.pop_front();
  }

  void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "cv_bridge rgb error: %s", e.what());
      return;
    }

    double t_rgb = rclcpp::Time(msg->header.stamp).seconds();
    cv::Mat depth;
    {
      std::lock_guard<std::mutex> lk(buf_mtx_);
      double best_dt = 0.08;
      for (const auto& d : depth_buf_) {
        double dt = std::abs(d.stamp - t_rgb);
        if (dt < best_dt) {
          best_dt = dt;
          depth = d.image;
        }
      }
    }
    if (depth.empty()) {
      return;
    }

    processFrame(cv_ptr->image, depth, msg->header.stamp, t_rgb);
  }

  double readDepthMeters(const cv::Mat& depth, int u, int v) const {
    if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows) return 0.0;
    double z = 0.0;
    if (depth.type() == CV_16UC1) {
      z = static_cast<double>(depth.at<uint16_t>(v, u)) / 5000.0 * depth_scale_;
    } else if (depth.type() == CV_32FC1) {
      z = static_cast<double>(depth.at<float>(v, u)) * depth_scale_;
    }
    if (!std::isfinite(z) || z < min_depth_ || z > max_depth_) return 0.0;
    return z;
  }

  std::vector<float> computeGlobalDescriptor(const cv::Mat& descriptors,
                                             const std::vector<cv::KeyPoint>& kps,
                                             int width, int height) const {
    // 64-dim compact descriptor: 2x2 spatial grid x 16-bin bit-density profile
    std::vector<float> desc(64, 0.0f);
    if (descriptors.empty()) return desc;

    std::vector<int> counts(4, 0);
    for (int i = 0; i < descriptors.rows; ++i) {
      int qx = (kps[i].pt.x >= width * 0.5f) ? 1 : 0;
      int qy = (kps[i].pt.y >= height * 0.5f) ? 1 : 0;
      int quad = qy * 2 + qx;
      counts[quad]++;
      const uint8_t* row = descriptors.ptr<uint8_t>(i);
      for (int b = 0; b < 16; ++b) {
        uint8_t byte_val = row[b * 2];
        int pop = __builtin_popcount(byte_val);
        desc[quad * 16 + b] += static_cast<float>(pop) / 8.0f;
      }
    }

    float norm = 0.0f;
    for (float v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-6f;
    for (float& v : desc) v /= norm;
    return desc;
  }

  void processFrame(const cv::Mat& bgr, const cv::Mat& depth,
                    const builtin_interfaces::msg::Time& ros_stamp, double t_sec) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kps;
    cv::Mat descs;
    orb_->detectAndCompute(gray, cv::noArray(), kps, descs);

    if (kps.size() < 20 || descs.empty()) {
      return;
    }

    if (!initialized_) {
      prev_kps_ = kps;
      prev_descs_ = descs.clone();
      prev_depth_ = depth.clone();
      kf_kps_ = kps;
      kf_descs_ = descs.clone();
      kf_depth_ = depth.clone();
      curr_pose_ = Eigen::Isometry3d::Identity();
      last_kf_pose_ = curr_pose_;
      initialized_ = true;
      publishPose(ros_stamp, t_sec);
      publishKeyframe(bgr, depth, kps, descs, ros_stamp);
      return;
    }

    // Track directly against last keyframe first (eliminates frame-to-frame drift),
    // falling back to previous frame if overlap with keyframe is insufficient.
    Eigen::Isometry3d delta_kf = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d delta_prev = Eigen::Isometry3d::Identity();
    if (trackAgainstReference(kf_kps_, kf_descs_, kf_depth_, kps, descs, delta_kf, 15)) {
      if (delta_kf.translation().norm() < 1.2) {
        curr_pose_ = last_kf_pose_ * delta_kf;
      }
    } else if (trackAgainstReference(prev_kps_, prev_descs_, prev_depth_, kps, descs, delta_prev, 10)) {
      if (delta_prev.translation().norm() < 0.8) {
        curr_pose_ = curr_pose_ * delta_prev;
      }
    }

    publishPose(ros_stamp, t_sec);

    Eigen::Isometry3d diff = last_kf_pose_.inverse() * curr_pose_;
    double dist = diff.translation().norm();
    Eigen::AngleAxisd aa(diff.linear());
    double angle = std::abs(aa.angle());

    if (dist >= kf_dist_thresh_ || angle >= kf_angle_thresh_) {
      last_kf_pose_ = curr_pose_;
      kf_kps_ = kps;
      kf_descs_ = descs.clone();
      kf_depth_ = depth.clone();
      publishKeyframe(bgr, depth, kps, descs, ros_stamp);
    }

    prev_kps_ = std::move(kps);
    prev_descs_ = descs.clone();
    prev_depth_ = depth.clone();
  }

  bool trackAgainstReference(const std::vector<cv::KeyPoint>& ref_kps,
                             const cv::Mat& ref_descs,
                             const cv::Mat& ref_depth,
                             const std::vector<cv::KeyPoint>& cur_kps,
                             const cv::Mat& cur_descs,
                             Eigen::Isometry3d& delta_out,
                             int min_inliers) const {
    if (ref_descs.empty() || cur_descs.empty()) return false;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(ref_descs, cur_descs, knn_matches, 2);

    std::vector<cv::Point2f> raw_ref_pts, norm_ref_pts, matched_img_pts;
    std::vector<double> matched_depths;
    for (const auto& m : knn_matches) {
      if (m.size() < 2 || m[0].distance >= 0.75f * m[1].distance) continue;
      const auto& kp_ref = ref_kps[m[0].queryIdx];
      double z = readDepthMeters(ref_depth, static_cast<int>(std::round(kp_ref.pt.x)),
                                 static_cast<int>(std::round(kp_ref.pt.y)));
      if (z <= 0.0) continue;
      raw_ref_pts.push_back(kp_ref.pt);
      matched_depths.push_back(z);
      matched_img_pts.push_back(cur_kps[m[0].trainIdx].pt);
    }

    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    if (!raw_ref_pts.empty()) {
      cv::undistortPoints(raw_ref_pts, norm_ref_pts, K_, dist_coeffs_);
      obj_pts.reserve(raw_ref_pts.size());
      img_pts = std::move(matched_img_pts);
      for (size_t i = 0; i < norm_ref_pts.size(); ++i) {
        double z = matched_depths[i];
        obj_pts.emplace_back(static_cast<float>(norm_ref_pts[i].x * z),
                             static_cast<float>(norm_ref_pts[i].y * z),
                             static_cast<float>(z));
      }
    }

    if (static_cast<int>(obj_pts.size()) < min_inliers + 2) return false;

    cv::Mat rvec, tvec, inliers;
    bool ok = cv::solvePnPRansac(obj_pts, img_pts, K_, dist_coeffs_, rvec, tvec,
                                 false, 120, 2.5, 0.99, inliers, cv::SOLVEPNP_EPNP);
    if (!ok || inliers.rows < min_inliers) return false;

    std::vector<cv::Point3f> inl_obj;
    std::vector<cv::Point2f> inl_img;
    inl_obj.reserve(inliers.rows);
    inl_img.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i) {
      int idx = inliers.at<int>(i, 0);
      inl_obj.push_back(obj_pts[idx]);
      inl_img.push_back(img_pts[idx]);
    }
    cv::solvePnP(inl_obj, inl_img, K_, dist_coeffs_, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R_cam;
    Eigen::Vector3d t_cam;
    for (int r = 0; r < 3; ++r) {
      t_cam(r) = tvec.at<double>(r, 0);
      for (int c = 0; c < 3; ++c) {
        R_cam(r, c) = R_cv.at<double>(r, c);
      }
    }
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    T_cur_ref.linear() = R_cam;
    T_cur_ref.translation() = t_cam;
    delta_out = T_cur_ref.inverse();
    return true;
  }

  void publishPose(const builtin_interfaces::msg::Time& stamp, double t_sec) {
    Eigen::Quaterniond q(curr_pose_.linear());
    q.normalize();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = robot_id_ + "/odom";
    odom.child_frame_id = robot_id_ + "/camera_link";
    odom.pose.pose.position.x = curr_pose_.translation().x();
    odom.pose.pose.position.y = curr_pose_.translation().y();
    odom.pose.pose.position.z = curr_pose_.translation().z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom_pub_->publish(odom);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(ps);
    path_pub_->publish(path_msg_);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = odom.header;
    tf_msg.child_frame_id = odom.child_frame_id;
    tf_msg.transform.translation.x = odom.pose.pose.position.x;
    tf_msg.transform.translation.y = odom.pose.pose.position.y;
    tf_msg.transform.translation.z = odom.pose.pose.position.z;
    tf_msg.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);

    traj_log_.push_back({t_sec, curr_pose_});
  }

  void publishKeyframe(const cv::Mat& bgr, const cv::Mat& depth,
                       const std::vector<cv::KeyPoint>& kps, const cv::Mat& descs,
                       const builtin_interfaces::msg::Time& stamp) {
    cam_skid_slam::msg::VisualKeyframe kf;
    kf.header.stamp = stamp;
    kf.header.frame_id = robot_id_ + "/odom";
    kf.robot_id = robot_id_;
    kf.keyframe_id = keyframe_count_++;

    Eigen::Quaterniond q(curr_pose_.linear());
    q.normalize();
    kf.odom_pose.position.x = curr_pose_.translation().x();
    kf.odom_pose.position.y = curr_pose_.translation().y();
    kf.odom_pose.position.z = curr_pose_.translation().z();
    kf.odom_pose.orientation.x = q.x();
    kf.odom_pose.orientation.y = q.y();
    kf.odom_pose.orientation.z = q.z();
    kf.odom_pose.orientation.w = q.w();

    std::vector<std::pair<float, size_t>> ranked;
    ranked.reserve(kps.size());
    for (size_t i = 0; i < kps.size(); ++i) {
      double z = readDepthMeters(depth, static_cast<int>(std::round(kps[i].pt.x)),
                                 static_cast<int>(std::round(kps[i].pt.y)));
      if (z > 0.0) {
        ranked.emplace_back(kps[i].response, i);
      }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    const size_t keep_n = std::min<size_t>(ranked.size(), static_cast<size_t>(max_kf_features_));

    std::vector<cv::Point2f> raw_kf_pts, norm_kf_pts;
    std::vector<double> kf_depths;
    std::vector<size_t> kf_indices;
    for (size_t k = 0; k < keep_n; ++k) {
      size_t i = ranked[k].second;
      double z = readDepthMeters(depth, static_cast<int>(std::round(kps[i].pt.x)),
                                 static_cast<int>(std::round(kps[i].pt.y)));
      raw_kf_pts.push_back(kps[i].pt);
      kf_depths.push_back(z);
      kf_indices.push_back(i);
    }
    if (!raw_kf_pts.empty()) {
      cv::undistortPoints(raw_kf_pts, norm_kf_pts, K_, dist_coeffs_);
      for (size_t k = 0; k < norm_kf_pts.size(); ++k) {
        double z = kf_depths[k];
        kf.keypoints_3d.push_back(static_cast<float>(norm_kf_pts[k].x * z));
        kf.keypoints_3d.push_back(static_cast<float>(norm_kf_pts[k].y * z));
        kf.keypoints_3d.push_back(static_cast<float>(z));
        const uint8_t* row = descs.ptr<uint8_t>(static_cast<int>(kf_indices[k]));
        kf.descriptors.insert(kf.descriptors.end(), row, row + 32);
      }
    }
    kf.num_keypoints = static_cast<uint32_t>(keep_n);
    kf.global_descriptor = computeGlobalDescriptor(descs, kps, bgr.cols, bgr.rows);

    // Back-project dense RGB-D cloud (stride 2, max 3.2m, rejecting depth-edge flying pixels)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int v = 2; v < depth.rows - 2; v += 2) {
      for (int u = 2; u < depth.cols - 2; u += 2) {
        double z = readDepthMeters(depth, u, v);
        if (z <= 0.0 || z > 3.2) continue;
        double zx = readDepthMeters(depth, u + 2, v);
        double zy = readDepthMeters(depth, u, v + 2);
        if (zx <= 0.0 || zy <= 0.0 || std::abs(zx - z) > 0.12 || std::abs(zy - z) > 0.12) {
          continue;
        }
        pcl::PointXYZRGB pt;
        pt.x = static_cast<float>((u - cx_) * z / fx_);
        pt.y = static_cast<float>((v - cy_) * z / fy_);
        pt.z = static_cast<float>(z);
        const cv::Vec3b& c = bgr.at<cv::Vec3b>(v, u);
        pt.r = c[2];
        pt.g = c[1];
        pt.b = c[0];
        cloud->push_back(pt);
      }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(static_cast<float>(voxel_leaf_),
                   static_cast<float>(voxel_leaf_),
                   static_cast<float>(voxel_leaf_));
    vg.filter(*filtered);

    pcl::toROSMsg(*filtered, kf.submap_cloud);
    kf.submap_cloud.header = kf.header;

    kf_pub_->publish(kf);
  }

  std::string robot_id_;
  double fx_, fy_, cx_, cy_, depth_scale_, max_depth_, min_depth_;
  double k1_, k2_, p1_, p2_, k3_;
  int num_features_, max_kf_features_;
  double kf_dist_thresh_, kf_angle_thresh_, voxel_leaf_;
  std::string save_dir_;

  cv::Mat K_, dist_coeffs_;
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;

  std::mutex buf_mtx_;
  std::deque<TimedImage> depth_buf_;

  bool initialized_ = false;
  std::vector<cv::KeyPoint> prev_kps_, kf_kps_;
  cv::Mat prev_descs_, prev_depth_, kf_descs_, kf_depth_;
  Eigen::Isometry3d curr_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d last_kf_pose_ = Eigen::Isometry3d::Identity();
  uint32_t keyframe_count_ = 0;

  nav_msgs::msg::Path path_msg_;
  std::vector<TrajEntry> traj_log_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<cam_skid_slam::msg::VisualKeyframe>::SharedPtr kf_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace cam_skid_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cam_skid_slam::VisualOdometryNode>());
  rclcpp::shutdown();
  return 0;
}
