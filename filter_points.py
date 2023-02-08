#!/usr/bin/env python
from time import time
import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np
import tf
import pcl_ros
import message_filters
import ctypes
import colorsys
from cv_bridge import CvBridge
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import struct
from std_msgs.msg import Header
import cv2
from scipy.spatial.transform import Rotation as R
from os.path import join
import utils_robot as U
from config import (K_matrices, PC_HEAD, CROP_X, CROP_Y, CROP_W, CROP_H)


def get_hsv_mask(img_hsv, light, dark):
    return cv2.inRange(img_hsv, light, dark)


def callback2(color_data, depth_data):
    """Callback function for getting point cloud data.

    NOTE(daniel): deprecated, this is based on Carl's dough manipulation code.
    When testing this code, put the wooden cutting board under the camera and get
    the blue dough (kinetic sand) on it.
    """
    debug_print = True

    # Get (aligned) color and depth images. Convert mm to meters.
    depth_im = bridge.imgmsg_to_cv2(depth_data)
    rgb_im = bridge.imgmsg_to_cv2(color_data)
    rgb_im = rgb_im[:, :, :3]
    depth_im = np.nan_to_num(depth_im)
    depth_im = depth_im / 1000.0

    # Get the points w.r.t. the base or world frame (they should be equivalent).
    h, w = depth_im.shape
    us, vs = np.repeat(np.arange(h), w), np.tile(np.arange(w), h)
    points = uv_to_camera_pos(us, vs, depth_im.flatten())[:, :3]
    points = cam_pos_to_world(points)

    # We want blue points for dough specifically.
    img_hsv = cv2.cvtColor(rgb_im.astype(np.float32), cv2.COLOR_RGB2HSV)
    mask = get_hsv_mask(img_hsv, light=(20,0.7,25), dark=(117,1.0,255))
    dough_points = points[np.where(mask.flatten()>0)]

    # Filter out invalid points. We only want dough within a 3D box.
    cropped_points = dough_points[~np.isnan(dough_points).any(axis=1)]
    if debug_print:
        print("\nDepth: {:0.3f} to {:0.3f}, median {:0.3f}".format(
                np.min(depth_im), np.max(depth_im), np.median(depth_im)))
        print("count of NaNs in dough_points: {}".format(np.sum(np.isnan(dough_points))))
        print("cropped points before filter: {}".format(cropped_points.shape))
        print("range 0 (x): {:0.3f} {:0.3f}".format(np.min(cropped_points[:,0]), np.max(cropped_points[:,0])))
        print("range 1 (y): {:0.3f} {:0.3f}".format(np.min(cropped_points[:,1]), np.max(cropped_points[:,1])))
        print("range 2 (z): {:0.3f} {:0.3f}".format(np.min(cropped_points[:,2]), np.max(cropped_points[:,2])))
        print("cropped points in range: {}".format(
                np.sum(np.logical_and(cropped_points[:,0]>0.35, cropped_points[:,0]<0.7))))
        print("cropped points in range: {}".format(
                np.sum(np.logical_and(cropped_points[:,1]>-0.6, cropped_points[:,1]<-0.2))))
        print("cropped points in range: {}".format(
                np.sum(np.logical_and(cropped_points[:,2]>-0.1, cropped_points[:,2]<0.2))))
    cropped_points = cropped_points[ np.logical_and(cropped_points[:,0] >  0.3, cropped_points[:,0] <  0.8) ]
    cropped_points = cropped_points[ np.logical_and(cropped_points[:,1] > -0.6, cropped_points[:,1] < -0.2) ]
    cropped_points = cropped_points[ np.logical_and(cropped_points[:,2] > -0.1, cropped_points[:,2] <  0.2) ]

    # Debugging, but we always _save_ so we can use code for later.
    if debug_print:
        print("({}) found a pointcloud: shape {} type {}".format(
                ns, points.shape, points.dtype))
        print("({}) after color segmentation: shape {} type {}".format(
                ns, dough_points.shape, dough_points.dtype))
        print("({}) after cropping: shape {} type {}".format(
                ns, cropped_points.shape, cropped_points.dtype))
    np.save(join(PC_HEAD, "pc_raw.npy"), points)
    np.save(join(PC_HEAD, "pc_dough.npy"), dough_points)
    np.save(join(PC_HEAD, "pc_dough_crop.npy"), cropped_points)
    debug_img = np.hstack([rgb_im, U.triplicate(mask)])
    cv2.imwrite(join(PC_HEAD, 'pc_mask_debug.png'), debug_img)

    # What we can actually use for learning.
    dough_pts_arr = np.zeros((len(cropped_points),), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
    ])
    dough_pts_arr['x'] = cropped_points[:, 0]
    dough_pts_arr['y'] = cropped_points[:, 1]
    dough_pts_arr['z'] = cropped_points[:, 2]

    msg = ros_numpy.msgify(PointCloud2, dough_pts_arr)
    # msg.header = data1.header
    publisher.publish(msg)


def callback_mm(color_data, depth_data):
    """Callback function for getting point cloud data for mixed media.

    Get point cloud information by taking each pixel in our camera, and then using
    depth to see the 3D camera (then world) point. So the point cloud fidelity is
    based on the pixel resolution of the (aligned) depth / color images.

    In the data collector, we crop first, then do a mask. But we should not crop because
    it affects the way the code computes conversions from pixels to camera position? We
    can still restrict the point clouds appropriately by adjusting the mask?

    Be careful about whether the camera uses meters or millimeters for the depth.
    It will be specified in this launch file in our catkin workspace:
    https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/launch/driver.launch

    To debug this, run and save files, and use `view.py` (in Python3). This can be done
    just with the launch file, no `robot.py` needed.

    NOTE(daniel): for time-aligning with images, it's a bit tricky as the data will not
    be collected at similar rates. With `debug` option here, PCL data takes significantly
    longer to save, so we'll get it less frequently.
    """
    debug = False

    # Get (aligned) color and depth images. Convert mm to meters.
    depth_im = bridge.imgmsg_to_cv2(depth_data)
    bgr_im = bridge.imgmsg_to_cv2(color_data, "bgr8")
    depth_im = np.nan_to_num(depth_im)
    depth_im = depth_im / 1000.0

    # Get the points w.r.t. the base or world frame (they should be equivalent).
    h, w = depth_im.shape
    us, vs = np.repeat(np.arange(h), w), np.tile(np.arange(w), h)
    points = uv_to_camera_pos(us, vs, depth_im.flatten())[:, :3]
    points = cam_pos_to_world(points)

    # Segment from the HSV image.
    img_hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)

    # Segment the items. If using color, in BGR mode (not RGB) but HSV seems
    # better. See segmentor.py for more details.
    targ_lo = np.array([ 26,  70, 170], dtype='uint8')
    targ_up = np.array([ 60, 255, 255], dtype='uint8')
    dist_lo = np.array([ 70,  70,  70], dtype='uint8')
    dist_up = np.array([155, 230, 255], dtype='uint8')

    # tool_lo = np.array([  0,   0,   0], dtype='uint8')
    # tool_up = np.array([255, 255,  45], dtype='uint8')

    #! Since the tool is now spray painted we had to change the HSV values
    tool_lo = np.array([ 65,  0,   0], dtype='uint8')
    tool_up = np.array([155, 255,  255], dtype='uint8')

    area_lo = np.array([  0,  70,   0], dtype='uint8')
    area_up = np.array([255, 255, 255], dtype='uint8')
    targ_mask = cv2.inRange(img_hsv, targ_lo, targ_up)
    dist_mask = cv2.inRange(img_hsv, dist_lo, dist_up)
    tool_mask = cv2.inRange(img_hsv, tool_lo, tool_up)
    area_mask = cv2.inRange(img_hsv, area_lo, area_up)

    # For cropped images. The w,h indicate width,height of cropped images.
    # This is to get rid of points OUTSIDE cropped regions. A bit clumsy...
    targ_mask[:, :CROP_X] = 0
    dist_mask[:, :CROP_X] = 0
    tool_mask[:, :CROP_X] = 0
    area_mask[:, :CROP_X] = 0
    targ_mask[:CROP_Y, :] = 0
    dist_mask[:CROP_Y, :] = 0
    tool_mask[:CROP_Y, :] = 0
    area_mask[:CROP_Y, :] = 0
    targ_mask[:, CROP_X+CROP_W:] = 0
    dist_mask[:, CROP_X+CROP_W:] = 0
    tool_mask[:, CROP_X+CROP_W:] = 0
    area_mask[:, CROP_X+CROP_W:] = 0
    targ_mask[CROP_Y+CROP_H:, :] = 0
    dist_mask[CROP_Y+CROP_H:, :] = 0
    tool_mask[CROP_Y+CROP_H:, :] = 0
    area_mask[CROP_Y+CROP_H:, :] = 0

    # Now can get points for each to get the segmented PC?
    targ_pts = points[ np.where(targ_mask.flatten() > 0) ]
    dist_pts = points[ np.where(dist_mask.flatten() > 0) ]
    tool_pts = points[ np.where(tool_mask.flatten() > 0) ]
    area_pts = points[ np.where(area_mask.flatten() > 0) ]

    # Filter out invalid points.
    targ_pts_c = targ_pts[ ~np.isnan(targ_pts).any(axis=1) ]
    dist_pts_c = dist_pts[ ~np.isnan(dist_pts).any(axis=1) ]
    tool_pts_c = tool_pts[ ~np.isnan(tool_pts).any(axis=1) ]
    area_pts_c = area_pts[ ~np.isnan(area_pts).any(axis=1) ]

    # Segmented point cloud for learning? Use c to indicate the class?
    # 0 = targ, 1 = dist, 2 = tool?
    N1 = len(targ_pts_c)
    N2 = len(targ_pts_c) + len(dist_pts_c)
    pcl_all = np.concatenate((targ_pts_c, dist_pts_c, tool_pts_c))
    pcl_class = np.zeros( (len(pcl_all),1) )
    pcl_class[N1:N2, 0] = 1.0
    pcl_class[N2:, 0] = 2.0
    pcl_data = np.hstack((pcl_all, pcl_class)).astype(np.float32)

    # Not sure if there is a need to have these 'x', etc. here? This is
    # producing a pcl_pts_arr of shape (K,) so it seems like we need to
    # put an array inside each of the variables?
    pcl = pcl_data  # x, then y, then z, then class
    pcl_pts_arr = np.zeros((len(pcl),), dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('c', np.float32),
    ])
    pcl_pts_arr['x'] = pcl[:, 0]
    pcl_pts_arr['y'] = pcl[:, 1]
    pcl_pts_arr['z'] = pcl[:, 2]
    pcl_pts_arr['c'] = pcl[:, 3]

    if debug:
        print('\nDebugging:')
        print("targ_pts: {}".format(targ_pts.shape))
        print("dist_pts: {}".format(dist_pts.shape))
        print("tool_pts: {}".format(tool_pts.shape))
        print("area_pts: {}".format(area_pts.shape))
        cv2.imwrite(join(PC_HEAD,'mask_targ.png'), targ_mask)
        cv2.imwrite(join(PC_HEAD,'mask_dist.png'), dist_mask)
        cv2.imwrite(join(PC_HEAD,'mask_tool.png'), tool_mask)
        cv2.imwrite(join(PC_HEAD,'mask_area.png'), area_mask)
        debug_img = np.hstack([
            bgr_im,
            U.triplicate(targ_mask),
            U.triplicate(dist_mask),
            U.triplicate(tool_mask),
            U.triplicate(area_mask),
        ])
        cv2.imwrite(join(PC_HEAD, 'mask_together.png'), debug_img)
        print('See PCs stored in {}'.format(PC_HEAD))
        print("pcl_all:   {}".format(pcl_all.shape))
        print("pcl_class: {}".format(pcl_class.shape))
        print("pcl_data:  {}".format(pcl_data.shape))

        # Visualize with `python view.py` later.
        np.save(join(PC_HEAD, "pts_targ_crop.npy"), targ_pts_c)
        np.save(join(PC_HEAD, "pts_dist_crop.npy"), dist_pts_c)
        np.save(join(PC_HEAD, "pts_tool_crop.npy"), tool_pts_c)
        np.save(join(PC_HEAD, "pts_area_crop.npy"), area_pts_c)
        np.save(join(PC_HEAD, "pts_combo_data.npy"), pcl_data)

    msg = ros_numpy.msgify(PointCloud2, pcl_pts_arr)
    msg.header.stamp = depth_data.header.stamp
    publisher.publish(msg)


def uv_to_camera_pos(u, v, z):
    """The standard way to go from pixels to a _camera_ position."""
    K = K_matrices[ns]
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    one = np.ones(u.shape)
    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    return cam_coords


def cam_pos_to_world(points):
    """Given points in camera frame, convert to world (i.e., base) frame."""
    camera_pos = np.array([
        camera_transform.transform.translation.x,
        camera_transform.transform.translation.y,
        camera_transform.transform.translation.z])
    camera_rot = R.from_quat([
        camera_transform.transform.rotation.x,
        camera_transform.transform.rotation.y,
        camera_transform.transform.rotation.z,
        camera_transform.transform.rotation.w])
    points = camera_rot.apply(points) + camera_pos
    return points


publisher = None
camera_transform =  None
tool_transform = None
bridge = None
ns = ''

def main():
    # Same subscribers that we use for the robot data collector.
    rospy.init_node("pointcloud_filter")
    rgbsub = message_filters.Subscriber('rgb/image_rect_color', Image)
    depthsub = message_filters.Subscriber("depth_to_rgb/image_raw", Image)
    ts = message_filters.ApproximateTimeSynchronizer([rgbsub, depthsub], 10, 0.1)
    ts.registerCallback(callback_mm)

    global publisher, camera_transform, bridge, ns
    bridge = CvBridge()
    ns = rospy.get_param("camera_ns")
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    while not rospy.is_shutdown():
        try:
            if ns == "k4a":
                camera_transform = buffer.lookup_transform(
                        'base', 'rgb_camera_link', rospy.Time(0))
            else:
                camera_transform = buffer.lookup_transform(
                        'base', 'top_rgb_camera_link', rospy.Time(0))
            print("Camera to base transformation: ", camera_transform)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    print("DONE")
    publisher = rospy.Publisher('filtered_points_world_xyz', PointCloud2, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()