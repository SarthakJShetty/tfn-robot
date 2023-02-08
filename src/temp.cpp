#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

cv::Mat image_cv, rgb_cv, dpt_cv;
sensor_msgs::ImagePtr image_data;

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    image_cv = cv_bridge::toCvShare(msg, "bgr8") -> image;
    // std::cout << image_cv << std::endl;
}

void img_cb(const sensor_msgs::Image::ConstPtr& rgb_img, const sensor_msgs::Image::ConstPtr& dpt_img){
    ROS_INFO("SYNCYING!");
    rgb_cv = cv_bridge::toCvShare(rgb_img, "bgr8") -> image;
    dpt_cv = cv_bridge::toCvShare(dpt_img, "16UC1") -> image;
    image_data = cv_bridge::CvImage(std_msgs::Header(), "bgr8", dpt_cv).toImageMsg();
    ROS_INFO("HERE!");
}

ros::NodeHandle* nh;

int main(int argc, char** argv){
    ros::init(argc, argv, "testNode");
    nh = new ros::NodeHandle();
    ros::Rate rate(100);
    ros::NodeHandle pnh;
    image_transport::ImageTransport it(pnh);
    // cv::namedWindow("view");
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(*nh, "/k4a_top/rgb/image_rect_color", 10);
    message_filters::Subscriber<sensor_msgs::Image> dpt_sub(*nh, "/k4a_top/depth_to_rgb/image_raw", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncer;
    message_filters::Synchronizer<syncer> img_sync(syncer(10), rgb_sub, dpt_sub);
    img_sync.registerCallback(boost::bind(&img_cb, _1, _2));
    // ROS_INFO("HERE1");
    image_transport::Publisher pub = it.advertise("/hello_there", 1);
    // ROS_INFO("HERE2");
    ros::spinOnce();
    // ROS_INFO("HERE3");
    while(ros::ok()){
        // ROS_INFO("HERE4");
        pub.publish(image_data);
        // ROS_INFO("HERE5");
        ros::spinOnce();
        rate.sleep();
    }
    // image_transport::Subscriber sub = it.subscribe("k4a_top/rgb/image_rect_color", 1, imageCallback);
}