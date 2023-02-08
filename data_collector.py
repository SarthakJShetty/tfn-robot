"""The DataCollector class polls data from the rostopics periodically.

Building on top of prior code at:
    https://github.com/sjyk/my-new-project
    https://github.com/DanielTakeshi/debridement-code
    https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/src/il_ros_hsr/core/sensors.py
and other code from R-PAD.

See also:
    http://wiki.ros.org/cv_bridge
    https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md

A few notes on how this code works:

- It's designed to be run in parallel with the SawyerRobot() class. In rospy as soon
  as we call `rospy.Subsriber()`, it spins off another thread.

- From my old ROS question: https://answers.ros.org/question/277209/
  Generally it seems good practice to NOT do much image processing in callbacks.

- It depends on what ROS topics we use, but for using color (rgb/image_rect_raw) and
  depth (depth/image_raw), they are NOT pixel aligned and have these sizes and types.
    color: (1536, 2048, 3), dtype('uint8')
    depth: (1024, 1024),    dtype('<u2')
  Thus, to make sure they are in the same space, we use `depth_to_rgb`. From GitHub:
    The depth image, transformed into the color camera co-ordinate space by the
    Azure Kinect Sensor SDK.  This image has been resized to match the color
    image resolution. Note that since the depth image is now transformed into
    the color camera co-ordinate space, some depth information may have been
    discarded if it was not visible to the depth camera.
"""
import os
from os.path import join
import cv2
import numpy as np

from pyquaternion import Quaternion as quat
import tf2_ros
import ros_numpy
import rospy
import cv_bridge
import message_filters
from sensor_msgs.msg import (Image, PointCloud2)
from config import (CROP_X, CROP_Y, CROP_H, CROP_W)
import utils_robot as U
from intera_core_msgs.msg import EndpointState

class DataCollector:

    def __init__(self, SawyerRobot, buffer):
        self.timestep = 0
        self.record_img = False
        self.record_pcl = False
        self.debug_print = False

        # Will this be OK for code? Any 'gotcha's? TODO(daniel) check.
        self.buffer = buffer
        self.SawyerRobot = SawyerRobot

        # Store color and depth images.
        self.c_image = None
        self.c_image_proc = None
        self.c_image_l = []
        self.c_image_proc_l = []
        self.d_image = None
        self.d_image_proc = None
        self.d_image_l = []
        self.d_image_header_l = []
        self.d_image_proc_l = []

        # Point clouds (segmented, subsampled)? FlowBot3D used 1200, Dough used 1000.
        self.max_pts = 5000
        self.pcl = None
        self.pcl_l = []
        self.pcl_l_header = []

        self.clr_imgs_l = []
        self.clr_imgs_header_l = []
        self.eep_pose_l = []
        self.eep_pose_l_header = []
        self.eep_pose_p = []
        self.eep_pose_r = []

        # Storing other info. Here, `_b` means w.r.t. the 'base'.
        self.ee_poses_b = []
        self.tool_poses_b = []

        # For cropped images. The w,h indicate width,height of cropped images.
        self.crop_x = CROP_X
        self.crop_y = CROP_Y
        self.crop_w = CROP_W
        self.crop_h = CROP_H

        # Segment the items. If using color, in BGR mode (not RGB) but HSV seems
        # better. See segmentory.py for more details.
        self.targ_lo = np.array([ 15,  70, 170], dtype='uint8')
        self.targ_up = np.array([ 60, 255, 255], dtype='uint8')
        self.targ_mask = None
        self.targ_mask_l = []

        self.dist_lo = np.array([ 70,  70,  70], dtype='uint8')
        self.dist_up = np.array([155, 230, 255], dtype='uint8')
        self.dist_mask = None
        self.dist_mask_l = []

        #! Tool mask after applying the spray-painting tape on the ladle
        self.tool_lo = np.array([ 65,  0,   0], dtype='uint8')
        self.tool_up = np.array([155, 255,  255], dtype='uint8')
        self.tool_mask = None
        self.tool_mask_l = []

        #! This is the mask for the earlier demos when the ladle was completely black
        # self.tool_lo = np.array([  0,   0,   0], dtype='uint8')
        # self.tool_up = np.array([255, 255,  45], dtype='uint8')
        # self.tool_mask = None
        # self.tool_mask_l = []

        self.area_lo = np.array([  0,  70,   0], dtype='uint8')
        self.area_up = np.array([255, 255, 255], dtype='uint8')
        self.area_mask = None
        self.area_mask_l = []

        # The usual 'bridge' between ROS and cv2. We need to get RGB and
        # and depth aligned, but there are some (unavoidable?) artifacts.
        self.bridge = cv_bridge.CvBridge()

        # "Depth to RGB". Don't do RGB to Depth, produces lots of empty space.
        # rospy.Subscriber('k4a_top/rgb/image_rect_color',
        #         Image, self.color_image_callback, queue_size=1)
        # rospy.Subscriber('k4a_top/depth_to_rgb/image_raw',
        #         Image, self.depth_image_callback, queue_size=1)

        # Point clouds (segmented). Not usually collected at same rate as images.

        # rospy.Subscriber('k4a_top/filtered_points_world_xyz',
        #         PointCloud2, self.get_point_cloud, queue_size=1)

        rgb_sub = message_filters.Subscriber('k4a_top/rgb/image_rect_color', Image)
        dpt_sub = message_filters.Subscriber('k4a_top/depth_to_rgb/image_raw', Image)
        pcl_sub = message_filters.Subscriber('k4a_top/filtered_points_world_xyz', PointCloud2)
        eep_sub = message_filters.Subscriber('/robot/limb/right/endpoint_state', EndpointState)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, dpt_sub, pcl_sub, eep_sub], 1000, 0.1)
        ts.registerCallback(self.synced_function)

    def synced_function(self, rgb, dpt, pcl, eep_pose):

        pc = ros_numpy.numpify(pcl)
        points = np.zeros((pc.shape[0], 4))
        points[:,0] = pc['x']
        points[:,1] = pc['y']
        points[:,2] = pc['z']
        points[:,3] = pc['c']

        if rospy.is_shutdown():
            return
        print('syncing!')
        self.c_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.c_image_header = {'secs': rgb.header.stamp.secs, 'nsecs':rgb.header.stamp.nsecs}
        self.c_image_proc = self._process_color(self.c_image)

        # Segment the target(s) and distractor(s).
        self.hsv = cv2.cvtColor(self.c_image_proc, cv2.COLOR_BGR2HSV)
        self.targ_mask = cv2.inRange(self.hsv, self.targ_lo, self.targ_up)
        self.dist_mask = cv2.inRange(self.hsv, self.dist_lo, self.dist_up)
        self.tool_mask = cv2.inRange(self.hsv, self.tool_lo, self.tool_up)
        self.area_mask = cv2.inRange(self.hsv, self.area_lo, self.area_up)

        if self.record_img:
            # Record color images and the EE pose (so we can track it later).
            # tool_pose_b = self.SawyerRobot.get_tool_pose()
            # self.tool_poses_b.append(tool_pose_b)

            if self.debug_print:
                rospy.loginfo('New color image, total: {}'.format(len(self.c_image_l)))
            #! Commenting these lines since they were overloading the memory on twofish
            self.c_image_l.append(self.c_image)
            self.c_image_proc_l.append(self.c_image_proc)
            self.targ_mask_l.append(self.targ_mask)
            self.dist_mask_l.append(self.dist_mask)
            self.tool_mask_l.append(self.tool_mask)
            self.area_mask_l.append(self.area_mask)

        if rospy.is_shutdown():
            return
        self.d_image = self.bridge.imgmsg_to_cv2(dpt)
        self.d_image_header = {'secs': dpt.header.stamp.secs, 'nsecs':dpt.header.stamp.nsecs}

        if self.record_pcl:
            if self.debug_print:
                rospy.loginfo('New PCL, shape: {}, tot before adding: {}'.format(
                        points.shape, len(self.pcl_l)))
            if len(points) > self.max_pts:
                # Due to this, PCL's are unsorted.
                choices = np.random.choice(len(points), size=self.max_pts, replace=False)
                points = points[choices]
            self.pcl = points
            #! Again for the same reasons as above, we comment out these lines so that the memory doesn't overload
            self.pcl_l.append(points)
            self.pcl_l_header.append({'secs':pcl.header.stamp.secs, 'nsecs':pcl.header.stamp.nsecs})

            eep = np.array([eep_pose.pose.position.x, eep_pose.pose.position.y, eep_pose.pose.position.z, eep_pose.pose.orientation.w, eep_pose.pose.orientation.x, eep_pose.pose.orientation.y, eep_pose.pose.orientation.z])
            ldp = np.array([eep_pose.pose.position.x, eep_pose.pose.position.y, eep_pose.pose.position.z])
            ldr_quat = quat(eep_pose.pose.orientation.w, eep_pose.pose.orientation.x, eep_pose.pose.orientation.y, eep_pose.pose.orientation.z)
            ldr = ldr_quat.rotation_matrix

            self.eep_pose_l.append(eep.copy())
            self.eep_pose_l_header.append({'secs':eep_pose.header.stamp.secs, 'nsecs':eep_pose.header.stamp.nsecs})
            self.eep_pose_p.append(ldp.copy())
            self.clr_imgs_l.append(self.c_image.copy())
            self.clr_imgs_header_l.append(self.c_image_header)
            self.d_image_l.append(self.d_image.copy())
            self.d_image_header_l.append(self.d_image_header)
            self.eep_pose_r.append(ldr.copy())

    def color_image_callback(self, msg):
        """If `self.record`, then this saves the color images.

        Also amazingly, just calling the same ee pose code seems to work?
        Careful, this might decrease the rate that images get called, we don't
        want much computation here. Profile it?

        From trying both 'rgb8' and 'bgr8', and saving with `cv2.imwrite(...)`, the
        'bgr8' mode seems to preserve colors as we see it.
        """
        if rospy.is_shutdown():
            return
        self.c_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.c_image_header = {'secs': msg.header.stamp.secs, 'nsecs':msg.header.stamp.nsecs}
        self.c_image_proc = self._process_color(self.c_image)

        # Segment the target(s) and distractor(s).
        self.hsv = cv2.cvtColor(self.c_image_proc, cv2.COLOR_BGR2HSV)
        self.targ_mask = cv2.inRange(self.hsv, self.targ_lo, self.targ_up)
        self.dist_mask = cv2.inRange(self.hsv, self.dist_lo, self.dist_up)
        self.tool_mask = cv2.inRange(self.hsv, self.tool_lo, self.tool_up)
        self.area_mask = cv2.inRange(self.hsv, self.area_lo, self.area_up)

        if self.record_img:
            # Record color images and the EE pose (so we can track it later).
            tool_pose_b = self.SawyerRobot.get_tool_pose()
            self.tool_poses_b.append(tool_pose_b)

            if self.debug_print:
                rospy.loginfo('New color image, total: {}'.format(len(self.c_image_l)))
            #! Commenting these lines since they were overloading the memory on twofish
            self.c_image_l.append(self.c_image)
            self.c_image_proc_l.append(self.c_image_proc)
            self.targ_mask_l.append(self.targ_mask)
            self.dist_mask_l.append(self.dist_mask)
            self.tool_mask_l.append(self.tool_mask)
            self.area_mask_l.append(self.area_mask)

    def depth_image_callback(self, msg):
        """Callback for the depth image.

        We use a ROS topic that makes the depth image into the same coordinate space
        as the RGB image, so the depth at pixel (x,y) should correspond to the 'depth'
        of the pixel (x,y) in the RGB image.

        For encoding (32FC1): a 32-bit float (32F) with a single channel (C1).
        """
        if rospy.is_shutdown():
            return
        self.d_image = self.bridge.imgmsg_to_cv2(msg)
        self.d_image_header = {'secs': msg.header.stamp.secs, 'nsecs':msg.header.stamp.nsecs}
        self.d_image_proc = self._process_depth(self.d_image)  # not cropped

        if self.record_img:
            # Record the depth image. Not sure if we need the processed one?
            if self.debug_print:
                rospy.loginfo('New depth image, total: {}'.format(len(self.d_image_l)))
            #! Again for the same reasons as above, we comment out these lines so that the memory doesn't overload
            # self.d_image_l.append(self.d_image)
            # self.d_image_proc_l.append(self.d_image_proc)

    def get_point_cloud(self, msg):
        """Amazingly, this works to collect point clouds.

        Be mindful that the frequency may not be the same as compared to images.
        If we do heavy data processing, it takes more time to 'produce' the data.
        """
        pc = ros_numpy.numpify(msg)
        points = np.zeros((pc.shape[0], 4))
        points[:,0] = pc['x']
        points[:,1] = pc['y']
        points[:,2] = pc['z']
        points[:,3] = pc['c']
        if self.record_pcl:
            if self.debug_print:
                rospy.loginfo('New PCL, shape: {}, tot before adding: {}'.format(
                        points.shape, len(self.pcl_l)))
            if len(points) > self.max_pts:
                # Due to this, PCL's are unsorted.
                choices = np.random.choice(len(points), size=self.max_pts, replace=False)
                points = points[choices]
            self.pcl = points
            ee_pose = self.buffer.lookup_transform('base', 'right_gripper_base', rospy.Time(0))
            #! Again for the same reasons as above, we comment out these lines so that the memory doesn't overload
            self.pcl_l.append(points)
            self.pcl_l_header.append({'secs':msg.header.stamp.secs, 'nsecs':msg.header.stamp.nsecs})

            ldp = np.array([ee_pose.transform.translation.x, ee_pose.transform.translation.y, ee_pose.transform.translation.z])
            ldr_quat = quat(ee_pose.transform.rotation.w, ee_pose.transform.rotation.x, ee_pose.transform.rotation.y, ee_pose.transform.rotation.z)
            ldr = ldr_quat.rotation_matrix
            eep = np.array([ee_pose.transform.translation.x, ee_pose.transform.translation.y, ee_pose.transform.translation.z, ee_pose.transform.rotation.w, ee_pose.transform.rotation.x, ee_pose.transform.rotation.y, ee_pose.transform.rotation.z])

            self.eep_pose_l.append(eep.copy())
            self.eep_pose_l_header.append({'secs':ee_pose.header.stamp.secs, 'nsecs':ee_pose.header.stamp.nsecs})
            self.eep_pose_p.append(ldp.copy())
            self.clr_imgs_l.append(self.c_image.copy())
            self.clr_imgs_header_l.append(self.c_image_header)
            self.d_image_l.append(self.d_image.copy())
            self.d_image_header_l.append(self.d_image_header)
            self.eep_pose_r.append(ldr.copy())

    def start_recording(self, img_record = True):
        self.record_img = img_record
        self.record_pcl = True

    def stop_recording(self):
        self.record_img = False
        self.record_pcl = False

    def get_color_image(self):
        return self.c_image

    def get_color_images(self):
        return self.c_image_l

    def get_depth_image(self):
        return self.d_image

    def get_depth_image_proc(self):
        return self.d_image_proc

    def get_depth_images(self):
        return self.d_image_l

    def get_ee_poses(self):
        return self.ee_poses_b

    def get_tool_poses(self):
        return self.tool_poses_b

    def _process_color(self, cimg):
        """Process the color image. Don't make this too computationally heavy."""
        cimg_crop = self.crop_img(cimg,
            x=self.crop_x, y=self.crop_y, w=self.crop_w, h=self.crop_h)
        return cimg_crop

    def _process_depth(self, dimg):
        """Process the depth image. Don't make this too computationally heavy."""
        return U.process_depth(dimg)

    def put_bbox_on_img(self, img, x, y, w, h):
        """Test bounding boxes on images.

        When visualizing images (e.g., with `eog cimg.png`) coordinates (x,y) start at
        (0,0) in the upper LEFT corner. Increasing x moves RIGHTWARD, but increasing y
        moves DOWNWARD. The w is for the width (horizontal length) of the image.

        We use (x,y) here and use cropping code that does y first, then x. This means
        the bounding box we visualize can be interpreted as how the image is cropped.
        """
        new_img = img.copy()
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (0,255,0), 3)
        return new_img

    def crop_img(self, img, x, y, w, h):
        """Crop the image. See documentation for bounding box."""
        new_img = img[y:y+h, x:x+w]
        return new_img
