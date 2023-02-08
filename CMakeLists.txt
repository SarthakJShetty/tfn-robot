cmake_minimum_required(VERSION 3.0.2)
project(mixed_media)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs cv_bridge image_transport genmsg OpenCV pcl_ros)
# add_message_files(DIRECTORY msg FILES Num.msg)
# add_service_files(DIRECTORY srv FILES AddTwoInts.srv)
generate_messages(DEPENDENCIES std_msgs)
include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
## System dependencies are found with CMake's conventions
# find_package(catkin REQUIRED COMPONENTS system)

catkin_package()

add_executable(testScript src/temp.cpp)
target_link_libraries(testScript
${catkin_LIBRARIES}
)
add_dependencies(testScript ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${OpenCV_LIBRARIES})