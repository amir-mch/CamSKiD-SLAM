cmake_minimum_required(VERSION 3.16)
project(cam_skid_slam)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread")

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree registration)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VisualKeyframe.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

set(ros_deps
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions cv_bridge
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

add_executable(cam_skid_vo src/visual_odometry_node.cpp)
ament_target_dependencies(cam_skid_vo ${ros_deps})
target_link_libraries(cam_skid_vo ${PCL_LIBRARIES} ${OpenCV_LIBS} "${cpp_typesupport_target}")

add_executable(cam_skid_fusion src/map_fusion_node.cpp)
ament_target_dependencies(cam_skid_fusion ${ros_deps})
target_link_libraries(cam_skid_fusion ${PCL_LIBRARIES} ${OpenCV_LIBS} gtsam "${cpp_typesupport_target}")

install(TARGETS
  cam_skid_vo
  cam_skid_fusion
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
