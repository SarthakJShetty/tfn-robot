"""Put test scripts here.

(c) Daniel Seita
    With significant help from Carl Qi, Sarthak Shetty, etc.
"""
import os
from os.path import split, join, basename
import sys
import cv2
import argparse
import json
import io
import time
import datetime
import struct
import numpy as np
np.set_printoptions(suppress=True, linewidth=120, precision=4)
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion as quat
import cv2
from collections import defaultdict
import open3d as o3d
from dslr_capture import VideoRecorder, DoneCam

#! Packages required to run inference
import zmq
from zmq import ssh

# Other stuff from this project.
from data_collector import DataCollector
import utils_robot as U

from glob import glob
# ROS stuff.
from cv_bridge import CvBridge
import rospy
import tf2_ros
import intera_interface as ii
from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest
)
#! Need these functions for managing the impedence while collecting BC demonstrations
from intera_motion_interface.utility_functions import int2bool, boolToggle
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions,
    InteractionOptions,
    InteractionPublisher
)
from intera_core_msgs.msg import InteractionControlCommand
from intera_motion_msgs.msg import TrajectoryOptions
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

# --------------------------- Various constants for MM -------------------------- #
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# NOTE/TODO(daniel) need to tune this and discount Fisheye effects.
PIX_TO_MM = 180.0 / 300.0  # roughly 180 mm corresponds to 300 pixels

# NOTE(daniel) these poses are w.r.t. a coordinate system that assumes it is
# centered at the gripper, where +z points down in direction of gripper. I
# measured the 33cm with a ruler and empirically it seems OK.

#TOOL_REL_POS = np.array([0.015, 0, 0.215])  # What Carl used for the dough roller.
TOOL_REL_POS = np.array([0.025, 0.0, 0.330])  # Ladle (but we might change)

# Tentative 'home' pose for MM.
MM_HOME_POSE_EE = [0.6562, 0.0113, 0.5282, -0.1823, -0.724, 0.6477, -0.1518]
# UPTIGHT_MM_HOME_POSE_EE = [0.65, -0.157, 0.5282, -0.1823, -0.724, 0.6477, -0.1518]

# Critical: tune these carefully! We have different safety constraints based on
# different stages of the robot movement.
SAFETY_LIMITS = {
    # Strict constraints when the policy is in action in the water.
    'policy': {
        'lim_x_ee'  : [ 0.40,  0.80],
        'lim_y_ee'  : [-0.15,  0.15],
        'lim_z_ee'  : [ 0.32,  0.52],
        'lim_x_tool': [ 0.60,  0.70],  # can be similar to EE
        'lim_y_tool': [-0.30, -0.18],  # lower since we tilt EE at angle
        'lim_z_tool': [ 0.02,  0.22],  # a lot lower (should be >0 at min)
    },

    # For when the policy does resets. These can be more flexible. If we go
    # close to the lower limit of z, for example, ideally we should be doing
    # some rotation so the ladle isn't 'digging into' the table.
    'resets': {
        'lim_x_ee'  : [ 0.40, 0.74],
        'lim_y_ee'  : [-0.35, 0.20],
        'lim_z_ee'  : [ 0.30, 0.70],
        'lim_x_tool': [ 0.40, 0.74],
        'lim_y_tool': [-0.35, 0.20],
        'lim_z_tool': [ 0.00, 0.50],
    },
}
# ------------------------------------------------------------------------------ #


def solver(pose):
    """From one (or both) of: Carl Qi and Sarthak Shetty.

    NOTE(daniel): I think, given a (position, quaternion) pose, it returns the
    target joint angles.
    """
    ns = 'ExternalTools/right/PositionKinematicsNode/IKService'
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    header = Header(stamp = rospy.Time.now(), frame_id = 'base')
    poses = {
        'right': PoseStamped(
            header = header,
            pose = Pose(
                position = Point(
                    x=pose[0],
                    y=pose[1],
                    z=max(pose[2],0.245),
                ),
                orientation = Quaternion(
                    w=pose[3],
                    x=pose[4],
                    y=pose[5],
                    z=pose[6]
                )
                )
            )
        }
    ikreq.pose_stamp.append(poses['right'])
    ikreq.tip_names.append('right_hand')
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
        return resp
    except (rospy.ServiceException, rospy.ROSException) as e:
        print('Failed due to:', e)
        return False


class SawyerRobot:
    """Encapsulate things in a class to make it simpler."""

    def __init__(self, policy=None):
        # Is this needed?
        def shutdown_func():
            print("Exit the mixed media.")

        # Rospy setups. The rate was borrowed from Carl, but I don't see a difference
        # with my testing for 1-100 for images and EE pose querying...

        rospy.init_node('thing1', anonymous=True, disable_signals=True)
        rospy.Rate(100)
        rospy.on_shutdown(shutdown_func)
        self.buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buffer)

        # Get limb through intera interface.
        self.limb = ii.Limb('right')
        #* Adding this attribute to the Sawyer so that we can control the joint speed of the robot
        self.traj = MotionTrajectory(limb=self.limb)

        if 'alg_t_' in policy:
            """Setting the joint_acceleration to the default value if we don't use
            algorithmic demonstrator"""
            rospy.loginfo('Setting special joint acceleration limits for policy: {}'.format(policy))
            self.set_joint_acceleration()
        else:
            self.wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.1,
                                              max_joint_accel=0.1)
            self.waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(),
                                       limb=self.limb)

        self.interaction_options = InteractionOptions()

        # Gripper, only if we want to use open/closing (normally we don't), which
        # we might need for swapping the tool that's being gripped.
        self.gripper = ii.Gripper('right_gripper')

        # Data collector. This has `CvBridge()` in it.
        self.DC = DataCollector(SawyerRobot=self, buffer=self.buffer)
        rospy.sleep(2)

    def set_joint_acceleration(self, joint_accel=0.1):
        '''This function sets the joint acceleration for the 7 joints, so that we can significantly reduce the lag
        between the pointcloud and the end-effector pose'''
        joint_accel_list = [joint_accel for _ in range(len(self.limb.joint_names()))]
        self.wpt_opts = MotionWaypointOptions(max_joint_accel=joint_accel_list, max_linear_speed=0.001, max_linear_accel=0.001, max_rotational_speed=0.001, max_rotational_accel=0.001)
        self.waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(),
                                       limb=self.limb)
    def set_impedence(self):
        #* This function makes some select axes unconstrained. We mainly use this to collect demonstrations, and we want
        #* either translation only or all 6DoFs available. Read here for more details on this funcitionality: https://sdk.rethinkrobotics.com/intera/Interaction_Control_Tutorial#Impedance.2FForce_Control_Examples
        rospy.loginfo('Initializing Impedence Publishers...')
        self.interaction_options.set_K_impedance([0, 0, 0, 0, 0, 0])
        self.interaction_options.set_max_impedance(boolToggle(int2bool([1, 1, 1, 0, 0, 0])))
        self.interaction_options.set_in_endpoint_frame(False)
        self.interaction_options.set_rotations_for_constrained_zeroG(True)
        ic_pub = InteractionPublisher()
        ic_pub.send_command(self.interaction_options, 0)
        # rospy.sleep(0.1)

    def default_impedence(self):
        #* We set the arm to default impedance when we want to reset to the starting position
        rospy.loginfo('Reverting Impedance...')
        self.interaction_options.set_K_impedance([0, 0, 0, 0, 0, 0])
        self.interaction_options.set_max_impedance(boolToggle(int2bool([1, 1, 1, 1, 1, 1])))
        self.interaction_options.set_in_endpoint_frame(False)
        self.interaction_options.set_rotations_for_constrained_zeroG(True)
        ic_pub = InteractionPublisher()
        ic_pub.send_command(self.interaction_options, 0)
        # rospy.sleep(0.1)

    def open_gripper(self):
        """Opens the gripper."""
        self.gripper.open()

    def close_gripper(self):
        """Closes the gripper."""
        self.gripper.close()

    def reboot_gripper(self):
        """If the gripper got messed up, reboot.

        Do NOT forcibly adjust the gripper width by manually pushing its fingers apart.
        Then we likely have to call this (or try `gripper.calibrate()`).
        """
        self.gripper.reboot()

    def get_ee_pose(self):
        """Get the EE pose based on the `right_connector_plate_base`, w.r.t. the base.

        The `lookup_transform` has 2 frames, the _second_ is the source, the _first_
        is the target, hence we want 'base' to be listed first.

        Unfortunately we have a hack of -0.004 because without it, if we query the
        pose, and ask the robot to move, it seems to strangely move up by 0.004
        (meters)? We might want to check this.

        Also, that hack only works if the robot is pointing downwards, otherwise
        we have to apply a correction transform. :-( TODO(daniel)
        """
        while not rospy.is_shutdown():
            try:
                #! Modifying this from Daniel's original code, where he used reference/right_connector_plate_base -> right_connector_plate_base 
                #! It seems that if we use the impedence control then the reference/* topics stop updating
                ee_transform = self.buffer.lookup_transform(
                    'base', 'right_connector_plate_base', rospy.Time(0))
                ee_pose = np.array([
                    ee_transform.transform.translation.x,
                    ee_transform.transform.translation.y,
                    ee_transform.transform.translation.z,
                    ee_transform.transform.rotation.w,
                    ee_transform.transform.rotation.x,
                    ee_transform.transform.rotation.y,
                    ee_transform.transform.rotation.z
                ])
                break
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                continue
        ee_pose[2] -= 0.004
        return ee_pose

    def get_tool_pose(self):
        """Get the tool pose, in the world frame.

        Depends on how we define the relative pose from the tool to the robot's EE. The
        robot's EE is the `right_connector_base_plate`. For mixed media, define the tool
        pose to be the 'center' of the approximate sphere formed from the ladle's 'bowl'?
        """
        ee_pose = self.get_ee_pose()
        tool_pose = self.ee_to_tool(ee_pose)
        return tool_pose

    def ee_to_tool(self, ee_pose):
        """Given EE pose (world frame), determine the tool pose (world frame).

        Assume the EE and tool poses (world frame) have the same quaternion.
        """
        pos, Q = ee_pose[:3], ee_pose[3:]
        r_mat = U.quaternion_rotation_matrix(Q)
        TOOL_REL_POS_world = r_mat.dot(TOOL_REL_POS)
        tool_pos_world = pos + TOOL_REL_POS_world
        return np.concatenate([tool_pos_world, Q])

    def tool_to_ee(self, tool_pose):
        """TODO"""
        pos, Q = tool_pose[:3], tool_pose[3:]
        r_mat = U.quaternion_rotation_matrix(Q)
        ee_rel_pos = -TOOL_REL_POS
        ee_rel_pos_world = r_mat.dot(ee_rel_pos)
        ee_pos_world = pos + ee_rel_pos_world
        return np.concatenate([ee_pos_world, Q])

    def world_to_pixel(self, points_world):
        """Convert from world points to pixel space.
        See documentation for `world_to_uv`.
        """
        uu,vv = U.world_to_uv(buffer=self.buffer,
                              world_coordinate=points_world,
                              camera_ns='k4a_top')
        return (uu,vv)

    # TODO(daniel) need to check / test.
    def pixel_to_world(self, u, v, z):
        """Convert from pixels to world space.

        Input: np.array of shape (n x 1) of type integers, representing pixels.
        See documentation in other method.
        Might be OK to convert theseto float32?
        """
        if len(np.array(u).shape) == 0:
            u = np.array([u]).astype(np.float32)
        if len(np.array(v).shape) == 0:
            v = np.array([v]).astype(np.float32)
        if len(np.array(z).shape) == 0:
            z = np.array([z]).astype(np.float32)  # depth

        points_w = U.uv_to_world_pos(buffer=self.buffer,
                                     u=u, v=v, z=z,
                                     camera_ns='k4a_top')
        return points_w

    def is_safe(self, ee_pose=None, tool_pose=None, bounds=''):
        """Enforce safety checks.

        Each time there's an action to take, we should call this to check bounds.
        There are different bounds involved depending on the current action. For
        example, actions when the tool is in the water have stricter bounds.

        Need to test how this works if we have rotations near edgfes of containers.

        Parameters
        ----------
        ee_pose: Desired EE pose, world/base frame.
        tool_pose: Desired tool pose, world/base frame.
        bounds: (string) Indicates which bounds to use.
        """
        assert not (ee_pose is None and tool_pose is None), 'Set one of these.'
        assert bounds in SAFETY_LIMITS, 'Bounds {} is not present'.format(bounds)
        is_safe = True
        LIM_X_EE   = SAFETY_LIMITS[bounds]['lim_x_ee']
        LIM_Y_EE   = SAFETY_LIMITS[bounds]['lim_y_ee']
        LIM_Z_EE   = SAFETY_LIMITS[bounds]['lim_z_ee']
        LIM_X_TOOL = SAFETY_LIMITS[bounds]['lim_x_tool']
        LIM_Y_TOOL = SAFETY_LIMITS[bounds]['lim_y_tool']
        LIM_Z_TOOL = SAFETY_LIMITS[bounds]['lim_z_tool']

        # Check positional bounds of the EE.
        if ee_pose is not None:
            cond_x = LIM_X_EE[0] <= ee_pose[0] <= LIM_X_EE[1]
            cond_y = LIM_Y_EE[0] <= ee_pose[1] <= LIM_Y_EE[1]
            cond_z = LIM_Z_EE[0] <= ee_pose[2] <= LIM_Z_EE[1]
            inbounds = cond_x and cond_y and cond_z
            if not inbounds:
                print('WARNING: EE safety check failed: {} {} {} {}'.format(
                        ee_pose[:3], cond_x, cond_y, cond_z))
                is_safe = False

        # Check positional bounds of the tool.
        if tool_pose is not None:
            cond_x = LIM_X_TOOL[0] <= tool_pose[0] <= LIM_X_TOOL[1]
            cond_y = LIM_Y_TOOL[0] <= tool_pose[1] <= LIM_Y_TOOL[1]
            cond_z = LIM_Z_TOOL[0] <= tool_pose[2] <= LIM_Z_TOOL[1]
            inbounds = cond_x and cond_y and cond_z
            if not inbounds:
                print('WARNING: tool safety check failed: {} {} {} {}'.format(
                        tool_pose[:3], cond_x, cond_y, cond_z))
                is_safe = False

        return is_safe

    def move_to_ee_pose(self, ee_pose, bounds):
        """Moves the robot's EE to the given pose using solver + waypoint.

        Parameters
        ----------
        ee_pose: Desired EE pose w.r.t. world/base frame, in (position, quaternion)
            format. The `solver` then converts this target pose to joint angles.
        bounds: (string) Indicates which safety check to use.

        Returns
        -------
        True if action was executed successfully, False if not.
        """
        tool_pose = self.ee_to_tool(ee_pose)
        # if not self.is_safe(ee_pose=ee_pose, tool_pose=tool_pose, bounds=bounds):
        #     print('Warning! Moving is deemed not safe.')
        #     return False
        # else:
        #     print('Executing!')
        resp = solver(pose=ee_pose)
        self.execute_waypoint(resp)
        return True

    def execute_waypoint(self, resp):
        """Actually move the robot to the target joint angles.

        References:
        https://sdk.rethinkrobotics.com/intera/Motion_Interface_Tutorial
        https://github.com/RethinkRobotics/intera_sdk/blob/development/intera_examples/scripts/go_to_joint_angles.py

        Question: can we control the robot speed?

        Note from the docs:
            [This] moves the arm from its current position to some target set of joint
            angles. Note that this script will not plan around self-collisions or obstacles.
        """
        try:
            self.traj.clear_waypoints()

            # Get current joint angles, append to `self.traj`.
            joint_angles = self.limb.joint_ordered_angles()
            self.waypoint.set_joint_angles(joint_angles=joint_angles)
            self.traj.append_waypoint(self.waypoint.to_msg())

            # Get target joint angles, append to `self.traj`. NOTE(daniel): the GitHub
            # code does not do this but assigns it using `set_joint_angles`.
            joint_angles = []
            for joint in range(0, 7):
                joint_angles.append(resp.joints[0].position[joint])
            self.waypoint.set_joint_angles(joint_angles=joint_angles)
            self.traj.append_waypoint(self.waypoint.to_msg())

            # Move the robot!
            result = self.traj.send_trajectory()
            if result is None:
                rospy.logerr('Trajectory FAILED to send!')
                return

            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory with interaction options set!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                             result.errorId)
        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. %s',
                         'Exiting before trajectory completion.')

    def rotate(self, radian, axis):
        """Rotates along some axis, keeping the position fixed.

        Keep the tool position the same. But be careful, if we just specify the
        axes like (1,0,0), (0,1,0), or (0,0,1), these are aligned with the world
        or base frame (they're the same here) because we are getting the ee pose
        and tool poses in that frame. The robot is actually gripping the ladle at
        an angle. This might induce some complexities.

        If we do things and keep the tool position the same, then this seems to
        nicely adjust the tool. I guess that's all that matters for us?

        Parameters
        ----------
        radian: amount to rotate by in the axis (in radians).
        axis: the axis that we rotate by, in the world or base frame.
        """
        assert axis in  ['x', 'y', 'z'], axis
        if axis == 'x':
            axis_arr = np.array([1, 0, 0])
        elif axis == 'y':
            axis_arr = np.array([0, 1, 0])
        elif axis == 'z':
            axis_arr = np.array([0, 0, 1])

        # Getting poses and making a `new_tool_pose` with the new quaternion.
        ee_pose = self.get_ee_pose()
        tool_pose = self.get_tool_pose()
        print('EE pose:   {}'.format(ee_pose))
        print('Tool pose: {}'.format(tool_pose))
        new_tool_pose = np.zeros(7)
        new_tool_pose[:3] = tool_pose[:3]

        # Compute the new tool (NOT EE!) quat by left-multiplying it by `transform`.
        transform = R.from_rotvec(radian * axis_arr)
        tp_quat = [tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]]
        new_quat = (transform * R.from_quat(tp_quat)).as_quat()
        new_tool_pose[3:] = [new_quat[-1], new_quat[0], new_quat[1], new_quat[2]]

        # Get the EE target pose from the new tool pose, and then move to it.
        target_ee_pose = self.tool_to_ee(new_tool_pose)
        print('Policy action:  {}'.format(new_tool_pose[:3] - tool_pose[:3]))
        print('Target EE pose: {}'.format(target_ee_pose))
        self.move_to_ee_pose(target_ee_pose, bounds='resets')


    def reset_to_rotated_start(self, robot_z, quat, robot_y=[-0.0207], start_recording=False, img_record = True, rotations = False):
        """Reset to begin a trial for algorithmic policy.

        Currently moves to the home position, then enters water and begins recording.
        """
        self.move_to_ee_pose(MM_HOME_POSE_EE, bounds='resets')

        inter_pose_1 = [0.6562,  0.07  , 0.5202,  -0.107530688176,  -0.475557657203,  0.827073107583, -0.279700090256]
        inter_pose_2 = [0.6562, 0.0113, 0.5282, -0.1823, -0.724, 0.6477, -0.1518]

        if rotations:
            self.move_to_ee_pose(inter_pose_1, bounds='resets')         

            print("pcl_size before rotation: {}".format(len(self.DC.pcl_l)))

            self.DC.pcl_l = []
            self.DC.eep_pose_l = []
            self.DC.eep_pose_p = []
            self.DC.eep_pose_r = []
            self.DC.clr_imgs_l = []
            self.DC.d_image_l = []
            self.DC.pcl_l_header = []
            self.DC.eep_pose_l_header = []
            self.DC.clr_imgs_header_l = []
            self.DC.d_image_header_l = []

            print("pcl_size after rotation: {}".format(len(self.DC.pcl_l)))

            if start_recording:
                rospy.loginfo('Started Recording')
                self.DC.start_recording(img_record = img_record)

            print("pcl_size before home: {}".format(len(self.DC.pcl_l)))

            # self.move_to_ee_pose(inter_pose_2, bounds='resets')
            ''''Move to some slightly offset positionm from MM_HOME_POSE_EE, then start recording'''

        # Initialize by inserting ladle closer to the 'top' as viewed from the image.
        ee_pose_close = [0.6562, robot_y[0], robot_z[0]] + quat

        # self.move_to_ee_pose(ee_pose_close, bounds='resets')

        print("pcl_size after home: {}".format(len(self.DC.pcl_l)))

        rospy.sleep(1)

    def reset_to_start(self, robot_z, quat, robot_y=[-0.0207], start_recording=False, img_record = True):
        """Reset to begin a trial for algorithmic policy.

        Currently moves to the home position, then enters water and begins recording.
        """
        self.move_to_ee_pose(MM_HOME_POSE_EE, bounds='resets')

        # Initialize by inserting ladle closer to the 'top' as viewed from the image.
        ee_pose_close = [0.6562, robot_y[0], robot_z[0]] + quat
        ee_pose_enter = [0.65, -0.0409] + robot_z + quat
        if start_recording:
            self.DC.start_recording(img_record = img_record)
        self.move_to_ee_pose(ee_pose_close, bounds='resets')
        rospy.sleep(1)
        # self.move_to_ee_pose(ee_pose_enter, bounds='policy')
        # rospy.sleep(1)

    def reset_to_neutral_start(self, robot_z, quat, start_recording=False):
        #! Copy of the above function, where the robot doesn't move to a start position but just resets in the same place
        #! mainly incorporated this to avoid jerky movements when the robot resets to the start posiiton
        """Reset to begin a trial for algorithmic policy.

        Currently moves to the home position, then enters water and begins recording.
        """
        self.move_to_ee_pose(MM_HOME_POSE_EE, bounds='resets')

        # Initialize by inserting ladle closer to the 'top' as viewed from the image.
        ee_pose_close = [0.65, -0.0207, 0.550] + quat
        if start_recording:
            self.DC.start_recording()
        self.move_to_ee_pose(ee_pose_close, bounds='resets')
        rospy.sleep(1)

    def reset_to_end(self, robot_z, quat, stop_recording=True):
        """Performs a reset procedure to randomize the starting configuration.

        (1) Enter the center of the water, roughly.
        (2) Do some stirring action.
        (3) Lift + lower to release anything.
        (4) Return to some home position.

        Update: actually after testing, I don't know if we want the stirring. It
        seems like we might get just as good data diversity by the way the robot
        moves and from the dropping action.
        """
        stir = False
        rospy.sleep(1)
        time.sleep(1)
        if stop_recording:
            self.DC.stop_recording()
        print('Robot is now resetting!')

        if stir:
            # (1,2) Enter the water (possibly holding items), then stir.
            ee_pose_enter = [0.65, -0.0409] + robot_z + quat
            self.move_to_ee_pose(ee_pose_enter, bounds='policy')

            # Move randomly to these poses to do the reset.
            random_poses = [
                [0.67, -0.04] + robot_z + quat,
                [0.67, -0.09] + robot_z + quat,
                [0.62, -0.09] + robot_z + quat,
                [0.62, -0.04] + robot_z + quat,
            ]
            n_attempts = np.random.randint(2, 5)
            last_idx = -1
            for _ in range(n_attempts):
                pose_idx = np.random.randint(len(random_poses))
                while pose_idx == last_idx:
                    pose_idx = np.random.randint(len(random_poses))
                last_idx = pose_idx
                self.move_to_ee_pose(random_poses[pose_idx], bounds='policy')
            print('Finished \'stirring\' for {} attempts'.format(n_attempts))

            # Back to this, then handle the lift + dumping part.
            self.move_to_ee_pose(ee_pose_enter, bounds='policy')

        # (3,4) Lift and rotate to release item(s) and water from ladle.
        ee_poses_reset = [
            [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # lift more (better for rot, updated 03/20)
            [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation
            [0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], # heavy rotation, more extreme
            [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation (duplicate)
            [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # return to this duplicate
            MM_HOME_POSE_EE,
        ]
        for pose in ee_poses_reset:
            rospy.sleep(1)
            self.move_to_ee_pose(pose, bounds='resets')
        print('Robot is done with reset.')

    def save_episode_results(self, args, epis, im_dict, num_demo = 0):
        """Save results from this episode for analysis later.

        Saves quite a lot of information, and in particular saves both all the images
        coming from ROS (to make nice videos, or get optical flow) and the images at
        each particular policy time step, which comes at less frequent intervals.
        """
        policy_folder = join(args.data_dir, 'policy_data')
        if num_demo==0:
            os.makedirs(policy_folder)

        dd = join(args.data_dir, 'demo_' + str(num_demo))
        '''NOTE (sarthak):Here im_dict is just a sequence of 4 images that contain: 1. The collor image, 2. The depth image with pixels
        above the depth cutoff, 3. Target points that are above this depth value and 4. Distractor points that are above this depth value'''

        # Store the image used for debugging. We do need the im_dict since we have
        # already done the 'reset' calls beforehand, so we don't want newer images.
        eval_debug_img = np.hstack([
            im_dict['eval_color_proc'],
            im_dict['eval_depth_cutoff'],
            im_dict['targ_raised'],
            im_dict['dist_raised'],
        ])

        if self.DC.record_img == True:
            # For images stored at fast rates to show a 'continuous' video.
            U.make_video(dd, self.DC.c_image_proc_l, key='color_crop')
            U.make_video(dd, self.DC.targ_mask_l,    key='targ_mask')
            U.make_video(dd, self.DC.dist_mask_l,    key='dist_mask')
            U.make_video(dd, self.DC.tool_mask_l,    key='tool_mask')
            U.make_video(dd, self.DC.area_mask_l,    key='area_mask')
            combo_all = []
            min_l = min([
                len(self.DC.c_image_proc_l),
                len(self.DC.targ_mask_l),
                len(self.DC.dist_mask_l),
                len(self.DC.tool_mask_l),
                len(self.DC.area_mask_l),
            ])
            for t in range(min_l):
                combo_t = np.hstack([
                    self.DC.c_image_proc_l[t],
                    U.triplicate(self.DC.targ_mask_l[t]),
                    U.triplicate(self.DC.dist_mask_l[t]),
                    U.triplicate(self.DC.tool_mask_l[t]),
                    U.triplicate(self.DC.area_mask_l[t]),
                ]).astype(np.uint8)
                combo_all.append(combo_t)
            U.make_video(dd, combo_all, key='combo_mask')

        # For images (and point clouds) used in the actual policy.
        pol_img_subdir = join(dd, 'policy_img')
        pol_pcl_subdir = join(dd, 'policy_pcl')
        pol_segm_subdir = join(dd, 'policy_segm')
        os.makedirs(pol_img_subdir)
        os.makedirs(pol_pcl_subdir)
        os.makedirs(pol_segm_subdir)

        cv2.imwrite(join(dd, 'eval_debug_img.png'), eval_debug_img)

        epis_T = len(epis['cimgs'])

        for t in range(epis_T):
            tt = str(t).zfill(2)

            # Save (roughly) aligned color images and point clouds.
            pth_cimg = join(dd, 'policy_img', 't{}_cimg.png'.format(tt))
            pth_draw = join(dd, 'policy_img', 't{}_cimg_draw.png'.format(tt))
            pth_pcl = join(dd, 'policy_pcl', 't{}_pcl.npy'.format(tt))
            cv2.imwrite(pth_cimg, epis['cimgs'][t])
            cv2.imwrite(pth_draw, epis['cimgs_draw'][t])
            np.save(pth_pcl, epis['pcls'][t])

            # Possible way of providing input (approximately) to policy?
            pth_segm = join(dd, 'policy_segm', 't{}_segm.png'.format(tt))
            combo_t = np.hstack([
                epis['cimgs'][t],
                U.triplicate(epis['targs'][t]),
                U.triplicate(epis['dists'][t]),
                U.triplicate(epis['tools'][t]),
                U.triplicate(epis['areas'][t]),
            ])
            cv2.imwrite(pth_segm, combo_t)

        # We can use this for evaluation.
        targ_raised_pix = np.sum(im_dict['targ_raised'][:,:,0] > 0)
        dist_raised_pix = np.sum(im_dict['dist_raised'][:,:,0] > 0)
        print('Pixels of targ. (raised): {}'.format(targ_raised_pix))
        print('Pixels of dist. (raised): {}'.format(dist_raised_pix))

        # Finally, save a variety of results (in json). The 'success' criteria is a bit
        # noisy so we may want to be careful. I have a heuristic now.
        success = False
        success = (targ_raised_pix >= 200) and (dist_raised_pix <= 200)
        results = {
            'targ_raised_pix': targ_raised_pix,
            'dist_raised_pix': dist_raised_pix,
            'success': int(success),
            # Rest of these should be lists.
            'dist_pix': epis['dist_pix'],
            'dist_pix_mm': epis['dist_pix_mm'],
            'ee_pose_b': epis['ee_pose_b'],
            'tool_pose_b': epis['tool_pose_b'],
            'posi_xy': epis['posi_xy'],
            'attempts': epis['attempts'],
        }
        results_pth = join(dd, 'results.json')
        with open(results_pth, 'w') as fh:
            json.dump(results, fh, indent=4)
        print('DONE! Success: {} (but check).'.format(success))

        assert len(self.DC.eep_pose_r) == len(self.DC.pcl_l) == len(self.DC.eep_pose_p) == len(self.DC.eep_pose_l), 'Check your callback functions. Some lists are not equally shaped. EEP: {} LDR: {} LDP: {} PCL: {} IMG: {} DEPTH: {}'.format(len(self.DC.eep_pose_l), len(self.DC.eep_pose_r), len(self.DC.eep_pose_p), len(self.DC.pcl_l), len(self.DC.clr_imgs_l), len(self.DC.d_image_l))

        for t in range(len(self.DC.pcl_l)):
            pcl_pth = join(policy_folder, 'pcl_{}_{}_{}_{}.npy'.format(num_demo, t, self.DC.pcl_l_header[t]['secs'], self.DC.pcl_l_header[t]['nsecs']))
            ldr_pth = join(policy_folder, 'ldr_{}_{}.npy'.format(num_demo, t))
            ldp_pth = join(policy_folder, 'ldp_{}_{}.npy'.format(num_demo, t))
            eep_pth = join(policy_folder, 'eep_{}_{}_{}_{}.npy'.format(num_demo, t, self.DC.eep_pose_l_header[t]['secs'], self.DC.eep_pose_l_header[t]['nsecs']))
            img_pth = join(policy_folder, 'img_{}_{}_{}_{}.png'.format(num_demo, t, self.DC.clr_imgs_header_l[t]['secs'], self.DC.clr_imgs_header_l[t]['nsecs']))
            dpt_pth = join(policy_folder, 'dpt_{}_{}_{}_{}.png'.format(num_demo, t, self.DC.d_image_header_l[t]['secs'], self.DC.d_image_header_l[t]['nsecs']))
            np.save(pcl_pth, self.DC.pcl_l[t])
            np.save(eep_pth, self.DC.eep_pose_l[t])
            np.save(ldp_pth, self.DC.eep_pose_p[t])
            np.save(ldr_pth, self.DC.eep_pose_r[t])
            cv2.imwrite(img_pth, self.DC.clr_imgs_l[t])
            cv2.imwrite(dpt_pth, self.DC.d_image_l[t])

# ---------------------------------------------------------------------------------- #
# Random test scripts to check the setup (e.g., for safety).
# ---------------------------------------------------------------------------------- #

def test_image_crops():
    """Quickly test image cropping, change the ranges in DataCollector.

    Can also put a bounding box on the image to better understand crops. See the
    documentation in DataCollector. Also, tests if color and depth images can be
    aligned, and put them side-by-side, or top-down.

    This should be called each time we change the physical setup. Ideally the image
    crops should directly give us images for processing into a policy.
    """
    robot = SawyerRobot()
    c_img = robot.DC.get_color_image()
    d_img = robot.DC.get_depth_image_proc()

    # NOTE(daniel): careful! Slight changes in physical setup mean different values.
    # The values used here should ideally be also used in the data collector class.
    # Increasing the x value means moving the crop region 'rightwards'.
    c_img_crop = robot.DC.crop_img(c_img, x=840, y=450, w=300, h=300)
    d_img_crop = robot.DC.crop_img(d_img, x=840, y=450, w=300, h=300)

    print('Image sizes: {}, {}'.format(c_img.shape, d_img.shape))
    cv2.imwrite('img_color.png', c_img)
    cv2.imwrite('img_depth.png', d_img)
    cv2.imwrite('img_crop_color.png', c_img_crop)
    cv2.imwrite('img_crop_depth.png', d_img_crop)

    # Helps to visualize alignment better.
    H,W,_ = c_img.shape
    img_aligned_1 = c_img.copy()
    img_aligned_2 = c_img.copy()
    img_aligned_3 = c_img.copy()
    img_aligned_4 = c_img.copy()
    img_aligned_1[:, int(W/2):, :] = (d_img.copy())[:, int(W/2):, :]
    cv2.imwrite('img_aligned_color_left.png', img_aligned_1)
    img_aligned_2[:, :int(W/2), :] = (d_img.copy())[:, :int(W/2), :]
    cv2.imwrite('img_aligned_color_right.png', img_aligned_2)
    img_aligned_3[int(H/2):, :, :] = (d_img.copy())[int(H/2):, :, :]
    cv2.imwrite('img_aligned_color_top.png', img_aligned_3)
    img_aligned_4[:int(H/2), :, :] = (d_img.copy())[:int(H/2), :, :]
    cv2.imwrite('img_aligned_color_bottom.png', img_aligned_4)

    # Stack all together.
    all_fname = 'img_aligned_all.png'
    row1 = np.hstack([img_aligned_1, img_aligned_2])
    row2 = np.hstack([img_aligned_3, img_aligned_4])
    combo = np.vstack([row1, row2])
    cv2.imwrite(all_fname, combo)
    print('See: {}, as well as cropped images, etc.'.format(all_fname))


def test_workspace_bounds():
    """Test workspace bounds.

    This should have the robot translate (maybe rotate) to test the workspace bounds.
    As a sanity check, the quaternions have similar values (if we're not rotating), as
    well as the heights. But we can check whatever we want.

    Fortunately we can get reasonable bounds by keeping the quaternion AND height fixed,
    and just do pure translation. Also, use different bounds for certain stages.
    """
    robot = SawyerRobot()
    rospy.sleep(2)

    # Getting started.
    poses_start = [
        MM_HOME_POSE_EE,
        [0.65  , -0.0207, 0.4753,  0.18,  0.70, -0.65,  0.16],  # close to water
    ]

    # Keep quaternions the same, and keep height fixed for the water poses.
    # FYI these are for EE poses, so the y values are a bit larger than expected.
    poses_water = [
        [0.65  , -0.0409,  0.36,  0.18,  0.70, -0.65,  0.16],  # go down
        [0.69  , -0.0251,  0.36,  0.18,  0.70, -0.65,  0.16],  # 1st corner
        [0.69  , -0.11  ,  0.36,  0.18,  0.70, -0.65,  0.16],  # 2nd corner
        [0.61  , -0.11  ,  0.36,  0.18,  0.70, -0.65,  0.16],  # 3rd corner
        [0.61  , -0.0324,  0.36,  0.18,  0.70, -0.65,  0.16],  # 4th corner
        [0.6559, -0.0628,  0.44,  0.18,  0.70, -0.65,  0.16],  # lift for evaluation
    ]

    # The reset procedure.
    poses_reset = [
        [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # lift more (better for rot, updated 03/20)
        [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation
        [0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], # heavy rotation, more extreme
        [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation (duplicate)
        [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # return to this duplicate
        MM_HOME_POSE_EE,
    ]

    for pose in poses_start:
        rospy.sleep(2)
        print('\n  Current TOOL posi: {}'.format(robot.get_tool_pose()[:3]))
        print('  Current EE posi:  {}'.format(robot.get_ee_pose()[:3]))
        print('  Going to EE pose: {}'.format(pose))
        robot.move_to_ee_pose(pose, bounds='resets')

    for pose in poses_water:
        rospy.sleep(2)
        print('\n  Current TOOL posi: {}'.format(robot.get_tool_pose()[:3]))
        print('  Current EE posi:  {}'.format(robot.get_ee_pose()[:3]))
        print('  Going to EE pose: {}'.format(pose))
        robot.move_to_ee_pose(pose, bounds='policy')

    for pose in poses_reset:
        rospy.sleep(2)
        print('\n  Current TOOL posi: {}'.format(robot.get_tool_pose()[:3]))
        print('  Current EE posi:  {}'.format(robot.get_ee_pose()[:3]))
        print('  Going to EE pose: {}'.format(pose))
        robot.move_to_ee_pose(pose, bounds='resets')

# ---------------------------------------------------------------------------------- #
# More test scripts
# ---------------------------------------------------------------------------------- #

def test_random_stuff(args):
    """Random tests and sanity checks.

    We can manually move the robot and query the EE pose, to get a sense of the
    appropriate environment boundaries. Then clip any poses (well, positions) that
    exceed such bounds. Make sure that `return` is commented out, though.
    """
    robot = SawyerRobot()
    rospy.sleep(3)  # Can let it sleep for a bit

    # Might as well check images.
    robot.DC.start_recording()

    # Can move robot around and exit to test workspace bounds.
    ee_pose = robot.get_ee_pose()
    tool_pose = robot.get_tool_pose()
    print('Current EE pose:   {}'.format(ee_pose))
    print('Current tool pose: {}'.format(tool_pose))

    #pose = [0.6378, -0.0736, 0.3641, -0.161 , -0.6951,  0.6796, -0.1705]  # enter the water!
    #robot.move_to_ee_pose(pose)

    # We only want this if replacing the tools.
    #robot.open_gripper()
    robot.close_gripper()

    return  # exit after checking poses.

    if False:
        # Sanity check: query current EE pose, then move to it. The robot should not move!
        print('SawyerRobot.reset_pose() [start] ...')
        robot.move_to_ee_pose(ee_pose)
        print('SawyerRobot.reset_pose() [end] ...')

        # Reset the robot to the starting 'home' position, tuned for this project.
        robot.move_to_ee_pose(MM_HOME_POSE_EE)

        # Test rotation.
        rotations = [-np.pi/2.0, np.pi/2.0, np.pi/2.0, -np.pi/2.0]
        for rot_z in rotations:
            print('\nRotating: {:0.1f} (deg)'.format(rot_z*RAD_TO_DEG))
            robot.rotate_z(rot_z)
            rospy.sleep(1)

        # Open or close the robot gripper. Careful! This will let go of the tool.
        return
        print('Opening and closing the robot gripper...')
        robot.open_gripper()
        rospy.sleep(1)
        robot.close_gripper()
        rospy.sleep(1)


# NOTE(daniel) uses deprecated `capture_image`.
def test_world_to_camera(args):
    """Intended use case: given world coordinates, check that we can get camera pixels.

    If we get negative pixels, they won't show up when we annotate via OpenCV.
    """
    robot = SawyerRobot()
    rospy.sleep(0)

    # Camera images. Make sure the roslaunch file 'activates' the camera nodes.
    cimg1, dimg1 = robot.capture_image(args.data_dir, camera_ns='k4a',     filename='k4a')
    cimg2, dimg2 = robot.capture_image(args.data_dir, camera_ns='k4a_top', filename='k4a_top')

    # Get the robot EE position (w.r.t. world) and get the pixels.
    ee_pose = robot.get_ee_pose()
    ee_posi = ee_pose[:3]
    print('Current EE position: {}'.format(ee_posi))

    # Use the top camera, make Nx3 matrix with world coordinates to be converted to camera.
    points_world = np.array([
        [0.0, 0.0, 0.0],  # base of robot (i.e., this is the world center origin)
        [0.1, 0.0, 0.0],  # more in x direction
        [0.2, 0.0, 0.0],  # more in x direction
        [0.3, 0.0, 0.0],  # more in x direction
        [0.4, 0.0, 0.0],  # more in x direction
        [0.5, 0.0, 0.0],  # more in x direction
        [0.6, 0.0, 0.0],  # more in x direction
        [0.7, 0.0, 0.0],  # more in x direction
        [0.8, 0.0, 0.0],  # more in x direction
        [0.9, 0.0, 0.0],  # more in x direction
        [1.0, 0.0, 0.0],  # more in x direction

        [0.5, -0.6, 0.0],  # check y
        [0.5, -0.5, 0.0],  # check y
        [0.5, -0.4, 0.0],  # check y
        [0.5, -0.3, 0.0],  # check y
        [0.5, -0.2, 0.0],  # check y
        [0.5, -0.1, 0.0],  # check y
        [0.5,  0.1, 0.0],  # check y
        [0.5,  0.2, 0.0],  # check y
        [0.5,  0.3, 0.0],  # check y

        [0.6, -0.1, 0.0],  # check z
        [0.6, -0.1, 0.1],  # check z
        [0.6, -0.1, 0.2],  # check z
        [0.6, -0.1, 0.3],  # check z
        [0.6, -0.2, 0.0],  # check z
        [0.6, -0.2, 0.1],  # check z
        [0.6, -0.2, 0.2],  # check z
        [0.6, -0.2, 0.3],  # check z
        [0.6, -0.3, 0.0],  # check z
        [0.6, -0.3, 0.1],  # check z
        [0.6, -0.3, 0.2],  # check z
        [0.6, -0.3, 0.3],  # check z
        [0.6, -0.4, 0.0],  # check z
        [0.6, -0.4, 0.1],  # check z
        [0.6, -0.4, 0.2],  # check z
        [0.6, -0.4, 0.3],  # check z
        [0.6, -0.5, 0.0],  # check z
        [0.6, -0.5, 0.1],  # check z
        [0.6, -0.5, 0.2],  # check z
        [0.6, -0.5, 0.3],  # check z
        [0.6, -0.6, 0.0],  # check z
        [0.6, -0.6, 0.1],  # check z
        [0.6, -0.6, 0.2],  # check z
        [0.6, -0.6, 0.3],  # check z

        [0.667, -0.374, 0.662],  # center of the camera (actually not visible)
    ])

    # Convert to pixels for 'k4a_top'!
    uu,vv = U.world_to_uv(buffer=robot.buffer,
                          world_coordinate=points_world,
                          camera_ns='k4a_top')

    # Now write over the image. For cv2 we need to reverse (u,v), right?
    # Actually for some reason we don't have to do that ...
    cimg = cimg2.copy()
    for i in range(points_world.shape[0]):
        p_w = points_world[i]
        u, v = uu[i], vv[i]
        print('World: {}  --> Pixels {} {}'.format(p_w, u, v))
        if i < 11:
            color = (0,255,255)
        elif i < 20:
            color = (255,0,0)
        else:
            color = (0,0,255)
        cv2.circle(cimg, center=(u,v), radius=10, color=color, thickness=-1)

    # Compare original vs new. The overlaid points visualize world coordinates.
    print('See image, size {}'.format(cimg.shape))
    cv2.imwrite('original_annotate.png', img=cimg2)
    cv2.imwrite('original.png', img=cimg)


def test_EE_tracking(args):
    """Intended use case: move some tool and track its pixels in the image.

    We have a separate data collector which polls ROS topics, and empirically it
    queries images at a good frequency. HOWEVER, we still need to query the EE
    pose, though? Can we ensure time alignment? Yes. :)
    """
    robot = SawyerRobot()

    # Get the robot EE position (w.r.t. world) and get the pixels.
    ee_pose = robot.get_ee_pose()
    ee_posi = ee_pose[:3]
    print('Current EE position: {}'.format(ee_posi))

    ee_poses = [
        MM_HOME_POSE_EE,
        [0.6986, -0.1559,  0.4709,  0.0625,  0.7226, -0.6865,  0.0517],
        [0.4567, -0.1768,  0.4729,  0.0569,  0.6996, -0.7112,  0.0402],
        MM_HOME_POSE_EE,
    ]
    robot.DC.start_recording()
    for ee_pose in ee_poses:
        rospy.sleep(1)
        print('Going to EE pose: {}'.format(ee_pose))
        robot.move_to_ee_pose(ee_pose)
    time.sleep(1)  # I've noticed it needs a delay to include the full action.
    robot.DC.stop_recording()

    # Then once things are over, make a video/GIF from saved images? Because we have
    # to use Python 2.7, I think it's going to be easier if this saves to the data
    # directory, and then we later call moviepy from our Python3 mixed media env.
    robot.DC.make_video(data_dir=args.data_dir, imgtype='raw')
    robot.DC.make_video(data_dir=args.data_dir, imgtype='proc')

    # We also want to track the EE location. That also helps with debugging.
    # Now make the video here but instead we overlay with images.
    points_world = np.array(robot.DC.get_ee_poses())
    points_world = points_world[:, :3]
    print('\nPoints world (shape {}):\n{}'.format(points_world.shape, points_world))

    # Convert to pixels for 'k4a_top'!
    uu,vv = U.world_to_uv(buffer=robot.buffer,
                          world_coordinate=points_world,
                          camera_ns='k4a_top')
    c_images = robot.DC.get_color_images()

    # Store the images, but annotate the pixels of the tool.
    imgtype = 'track_tool'
    for idx in range(len(c_images)):
        tail = 'color_{}_{}.png'.format(imgtype, str(idx).zfill(4))
        cname = join(args.data_dir, tail)
        cimg = c_images[idx].copy()
        p_w = points_world[idx]
        u, v = uu[idx], vv[idx]
        print('World: {}  --> Pixels {} {}'.format(p_w, u, v))
        color = (0,255,0)
        cv2.circle(cimg, center=(u,v), radius=16, color=color, thickness=-1)
        cv2.imwrite(cname, cimg)
    print('Done saving {} color images in {}'.format(len(c_images), args.data_dir))

    # Make the actual video using standard processing code as earlier.
    video_name = join(args.data_dir, 'recording_{}.avi'.format(imgtype))
    images = sorted(
        [img for img in os.listdir(args.data_dir)
            if img.endswith(".png") and imgtype in img]
    )
    video = cv2.VideoWriter(video_name, 0, fps=10, frameSize=(640,640))
    for image in images:
        img_t = cv2.imread(join(args.data_dir, image))
        img_t = cv2.resize(img_t, (640,640))
        video.write(img_t)
    cv2.destroyAllWindows()
    video.release()
    print('See video: {}'.format(video_name))


def test_action_params(args):
    """Intended use case: test manipulation. :)

    Let's now test different rotations. We should later record videos, store the
    point clouds, etc.

    When doing this test, I wouldn't do more than 15 degrees of variation. The
    Sawyer arm ends up having to move quite a lot to make these rotations w.r.t.
    the ladle's bowl center. Hopefully +/- 15 degrees of variation across each
    axis is all we need? Also there are imperfections in that the ladle is not
    returning to the same spot each time.
    """
    robot = SawyerRobot()
    rospy.sleep(5)
    robot.DC.start_recording()

    # Get the robot EE position (w.r.t. world).
    ee_posi = robot.get_ee_pose()[:3]
    print('Starting EE: {}'.format(ee_posi))

    # Set of rotations to perform. Items are tuples: (axis, deg). These are a
    # set of rotations that we should perform. All these are 'neutral' actions
    # and should theoretically get the ladle back to the same spot it started.
    # If it's off, we can consider tuning the pose of the tool wrt the EE?
    D = 15
    rotations = [
        ('x', -D),
        ('x',  D),
        ('x',  D),
        ('x', -D),
        ('y', -D),
        ('y',  D),
        ('y',  D),
        ('y', -D),
        ('z', -D),
        ('z',  D),
        ('z',  D),
        ('z', -D),
    ]
    for rots in rotations:
        angle_axis, angle_deg = rots
        angle_rad = angle_deg * DEG_TO_RAD
        robot.rotate(radian=angle_rad, axis=angle_axis)
        rospy.sleep(1)
        ee_posi = robot.get_ee_pose()[:3]
        print('After rot {} in {}, current EE: {}\n'.format(
                angle_deg, angle_axis, ee_posi))
    rospy.sleep(2)
    print('\nAll done!')
    print('EE pose:   {}'.format(robot.get_ee_pose()))
    print('Tool pose: {}'.format(robot.get_tool_pose()))

    # Save videos.
    robot.DC.stop_recording()
    U.make_video(args.data_dir, robot.DC.c_image_proc_l, key='color_crop')
    U.make_video(args.data_dir, robot.DC.tool_mask_l,    key='tool_mask')
    print('Done!')

# ---------------------------------------------------------------------------------- #
# Test kinesthetic teaching from human instruction.
# ---------------------------------------------------------------------------------- #

def test_kinesthetic_continuous(args):
    """Kinesthetic teaching, but in a more 'continuous' form.

    Here, the human should move the robot continuously, no need for `raw_input`.
    For a 'slower' version, see `kinesthetic_teaching`.

    Question: I think there's a way to do this but how do you actually recover
    from a CTRL+C in Python, and have the code continue? TODO(daniel). Right now
    I just save repeatedly so that CTRL+C will exit?
    """
    dd = args.data_dir
    robot = SawyerRobot()
    ee_poses_b = []
    robot.DC.start_recording()

    while True:
        time.sleep(0.5)
        ee_pose = robot.get_ee_pose()
        tool_pose = robot.get_tool_pose()
        if not robot.is_safe(ee_pose, tool_pose, bounds='resets'):
            print('Warning!')
        ee_poses_b.append(ee_pose)
        print('Have {} poses after adding: {}.'.format(len(ee_poses_b), ee_pose))
        U.make_video(dd, robot.DC.c_image_proc_l, key='color_crop')

def test_kinesthetic_teaching():
    """Kinesthetic teaching.

    Here, we should move the robot, press space, which saves the pose. Then, we
    move the robot again, press space, etc. This saves a sequence of poses to a
    file, which we can then copy and paste to test with `test_waypoints()`.
    """
    robot = SawyerRobot()
    ee_poses_b = []

    while True:
        usr_input = raw_input("Record pose? (y for yes). Else exit (n): ")
        while (usr_input.lower() != "y") and (usr_input.lower() != "n"):
            usr_input = raw_input("Please enter a valid option. (y/n)").lower()
        if usr_input.lower() == "n":
            break
        ee_pose = robot.get_ee_pose()
        tool_pose = robot.get_tool_pose()
        if not robot.is_safe(ee_pose, tool_pose, bounds='resets'):
            print('Warning!')
        ee_poses_b.append(ee_pose)
        print('Have {} poses after adding: {}.'.format(len(ee_poses_b), ee_pose))

    # Here it prints on the command line, we can just copy and paste.
    print('\nDone with {} poses. In string format:'.format(len(ee_poses_b)))
    print('[')
    for p in ee_poses_b:
        print('[{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}],'.format(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]))
    print(']')


def test_waypoints(args, amt_sleep=0.0):
    """Intended use case: test a sequence of waypoints.

    Similar to testing workspace bounds, but here we can have an arbitrary set
    of waypoints to test different motions, etc. We can get the waypoints by
    (for example) using `test_kinesthetic_teaching()`.

    Not sure what values yet to use for the amount to sleep between poses.
    """
    dd = args.data_dir

    # # Scoop towards wall facing me, for initial testing 03/31/2022.
    # waypoints = [
    #     [0.6364, 0.0200, 0.4556, -0.2305, -0.7374, 0.5942, -0.2239],
    #     [0.6605, -0.0199, 0.4049, 0.1528, 0.5300, -0.7610, 0.3414],
    #     [0.6456, -0.0238, 0.3421, 0.1235, 0.4530, -0.8136, 0.3430],
    #     [0.6305, -0.0096, 0.3183, 0.1674, 0.4695, -0.8038, 0.3247],
    #     [0.6254, -0.0177, 0.3528, -0.2381, -0.6863, 0.6475, -0.2304],
    #     [0.6546, -0.0256, 0.4220, -0.2490, -0.7158, 0.6174, -0.2106],
    # ]

    # Test waypoints with Sarthak 04/03/2022.
    waypoints = [
        [0.6300, 0.0290, 0.4951, -0.2113, -0.7349, 0.6039, -0.2248],
        [0.6450, -0.0396, 0.3835, -0.1958, -0.7343, 0.6118, -0.2196],
        [0.6347, -0.0732, 0.3621, -0.1361, -0.6620, 0.7006, -0.2289],
        [0.6257, -0.1085, 0.3715, -0.1709, -0.7388, 0.6273, -0.1775],
        [0.6273, -0.0876, 0.4253, -0.1722, -0.7762, 0.5764, -0.1889],
    ]

    # Start up the robot as usual, and record images.
    robot = SawyerRobot()
    rospy.sleep(5)
    robot.DC.start_recording()

    # Get the robot EE pose (w.r.t. world).
    ee_pose = robot.get_ee_pose()
    print('Starting EE: {}'.format(ee_pose))

    # If any point in the waypoint is unsafe, just exit right away because it
    # could impact the location next waypoints.
    for eidx,ee_pose in enumerate(waypoints):
        safe = robot.move_to_ee_pose(ee_pose, bounds='resets')
        if not safe:
            print('Exiting, unsafe!')
            break
        rospy.sleep(amt_sleep)
        print('After action {} of {}, EE: {}\ntool: {}\n'.format(eidx+1,
            len(waypoints), robot.get_ee_pose(), robot.get_tool_pose()))
    rospy.sleep(amt_sleep)
    print('\nAll done!')
    print('EE pose:   {}'.format(robot.get_ee_pose()))
    print('Tool pose: {}'.format(robot.get_tool_pose()))

    # Save videos and point cloud.
    rospy.sleep(1)  # have a little extra to get final pose.
    robot.DC.stop_recording()
    U.make_video(dd, robot.DC.c_image_proc_l, key='color_crop')
    U.make_video(dd, robot.DC.tool_mask_l,    key='tool_mask')
    U.make_video(dd, robot.DC.targ_mask_l,    key='targ_mask')
    U.make_video(dd, robot.DC.dist_mask_l,    key='dist_mask')
    U.save_pcl(dd, robot.DC.pcl_l)
    print('Done!')

# ---------------------------------------------------------------------------------- #
# Algorithmic policy (policies).
# ---------------------------------------------------------------------------------- #

def algorithmic_simple_demonstrator(args):

    # Will do translations only by keeping this as the quaternion.
    QUAT = [-0.1823, -0.724, 0.6477, -0.1518]

    ROBOT_Z = [0.5282]
    T = args.max_T
    PIX_THRESH = 25

    for num_demo in range(args.num_demos):

        robot = SawyerRobot(args.policy)
        rospy.sleep(3)

        robot.DC.eep_pose_l = []
        robot.DC.eep_pose_p = []
        robot.DC.eep_pose_r = []
        robot.DC.pcl_l = []

        # Reset the robot and enter ladle at the 'top' as viewed from the image.
        robot.set_joint_acceleration(joint_accel=0.1)
        robot.reset_to_start(robot_z=ROBOT_Z, robot_y = [0.0113], quat=QUAT, start_recording=True, img_record=False)

        # Algorithmic policy: iterate until pixel distance is under a threshold.
        print('\nRunning the algorithmic policy for up to {} steps.'.format(T))
        epis = defaultdict(list)

        # ------------------------------------------------------------------------------- #
        # Lift and raise gripper to evaluate. Evaluation is going to be trickier when we
        # go beyond 1 target and 1 distractor, as it may be hard to do contour detection.
        # My thinking is to just move the ladle and then check if target is still visible.
        # It may fall out during the movement, though I think it doesn't happen that much.
        # ------------------------------------------------------------------------------- #
        print('\n------------------- Move gripper up to to evaluate -------------------')

        x_setpoint = 0.800
        rospy.loginfo('Heading to {}'.format(x_setpoint))
        ee_pose = robot.get_ee_pose()
        ee_eval_pose_1 = [x_setpoint, ee_pose[1], ee_pose[2]] + QUAT  # straight up
        robot.move_to_ee_pose(ee_eval_pose_1, bounds='policy')
        """The sleep duration has been increased so that we can encode an explicit
        stopping condition in the data"""
        rospy.sleep(3.0)

        # Evaluate depth. Raw distance is in millimeters.
        eval_color_proc = robot.DC.c_image_proc
        eval_depth = robot.DC.d_image
        #U.print_debug(img=eval_depth, imgname='Eval depth (raw)')
        eval_depth = robot.DC.crop_img(eval_depth,
                x=robot.DC.crop_x, y=robot.DC.crop_y, w=robot.DC.crop_w, h=robot.DC.crop_h)

        #U.print_debug(img=eval_depth, imgname='Eval depth (crop)')

        # NOTE(daniel): this cutoff is important to ignore anything in the water!

        eval_depth_cutoff = U.process_depth(eval_depth, cutoff=580)
        eval_depth_mask = ((eval_depth_cutoff > 0) * 255).astype(np.uint8)[:,:,0]  # 1 channel
        eval_targ = robot.DC.targ_mask
        eval_dist = robot.DC.dist_mask
        targ_raised = np.bitwise_and(eval_targ, eval_depth_mask)  # targ. pixels above threshold
        dist_raised = np.bitwise_and(eval_dist, eval_depth_mask)  # dist. pixels above threshold
        targ_raised = U.triplicate(targ_raised, to_int=True)
        dist_raised = U.triplicate(dist_raised, to_int=True)

        rospy.loginfo('Resetting!')
        # Stop recording and reset.
        rospy.sleep(1.0)
        robot.set_joint_acceleration(joint_accel=0.5)
        robot.reset_to_end(robot_z=ROBOT_Z, quat=QUAT, stop_recording=True)

        # Store data from this episode, should be consistent among different policies.
        print('\n---------------------- Now saving videos, etc. -----------------------')
        im_dict = {
            'eval_color_proc': eval_color_proc,
            'eval_depth': eval_depth,
            'eval_depth_cutoff': eval_depth_cutoff,
            'targ_raised': targ_raised,
            'dist_raised': dist_raised,
        }

        robot.save_episode_results(args=args, epis=epis, im_dict=im_dict, num_demo=num_demo)
        del robot.DC.c_image_l
        del robot.DC.pcl_l
        del robot.DC.eep_pose_l
        del robot.DC.eep_pose_p
        del robot.DC.eep_pose_r
        del robot.DC.c_image_proc_l
        del robot.DC.targ_mask_l
        del robot.DC.clr_imgs_l
        del robot.DC.d_image_l
        del robot.DC.dist_mask_l
        del robot.DC.tool_mask_l
        del robot.DC.area_mask_l

def algorithmic_translation_demonstrator(args):
    '''TODO (sarthak): What do I need to implement here?
    1. Save the required data using the data_collector class since that seems to run on a much faster thread than this one.
    2. Required data: LDR, LDP (can we simplifying things with just eep?), PCL, Images
    3. Wrap this in a iterator that loops for n_demo number of times, use the same BC variable here.'''
    """Test the algorithmic policy, can run up to `T = args.max_T` time steps.

    This one operates in pixel space and seems to be slightly more reliable than the
    world space version because the latter seems to not properly compute the world?
    Cropped images are 300x300, which corresponds to roughly 180mm in the real world.

    TODO(daniel): currently only works with 1 target, how to do multiple targets?
    """

    # Will do translations only by keeping this as the quaternion.
    QUAT = [0.18,  0.70, -0.65,  0.16]
    ROBOT_Z = [0.360]
    T = args.max_T
    PIX_THRESH = 25

    for num_demo in range(args.num_demos):
        robot = SawyerRobot(args.policy)
        rospy.sleep(3)

        robot.DC.eep_pose_l = []
        robot.DC.eep_pose_p = []
        robot.DC.eep_pose_r = []
        robot.DC.pcl_l = []

        # Reset the robot and enter ladle at the 'top' as viewed from the image.
        robot.set_joint_acceleration(joint_accel=0.1)
        robot.reset_to_start(robot_z=ROBOT_Z, quat=QUAT, start_recording=True, img_record=False)

        # Algorithmic policy: iterate until pixel distance is under a threshold.
        print('\nRunning the algorithmic policy for up to {} steps.'.format(T))
        epis = defaultdict(list)

        for t in range(T):

            print('\n=========================== Demonstrations: {} Action {}/{} ==========================='.format(num_demo, t+1,T))
            ee_pose_b = robot.get_ee_pose()
            tool_pose_b = robot.get_tool_pose()
            points_world = np.reshape(tool_pose_b[:3], (1,3))
            uu,vv = robot.world_to_pixel(points_world=points_world)
            tool_u = uu[0] - robot.DC.crop_x
            tool_v = vv[0] - robot.DC.crop_y

            # Mask is on the cropped image to save time and to ignore stuff outside container.
            cimg_proc = (robot.DC.c_image_proc).copy()
            targ_mask = (robot.DC.targ_mask).copy()
            dist_mask = (robot.DC.dist_mask).copy()
            tool_mask = (robot.DC.tool_mask).copy()
            area_mask = (robot.DC.area_mask).copy()

            # Get the point cloud data, hopefully roughly time-aligned with images.
            pcl_t = (robot.DC.pcl).copy()

            # This lets us use cv2.circle(targ_mask,(u,v)) to visualize the target.
            mask_idxs = np.where(targ_mask > 0)
            targ_u = int(np.median(mask_idxs[1]))
            targ_v = int(np.median(mask_idxs[0]))

            # Debugging. See other methods for debugging with images, etc.
            print('tool position (world): {}'.format(points_world))
            print('tool u,v: {} {} (on cropped image)'.format(tool_u, tool_v))
            print('targ u,v: {} {} (on cropped image)'.format(targ_u, targ_v))
            draw_cimg = cimg_proc.copy()
            cv2.circle(draw_cimg, (targ_u,targ_v), radius=10, color=(0,255,0), thickness=-1)
            cv2.circle(draw_cimg, (tool_u,tool_v), radius=10, color=(255,0,0), thickness=-1)

            # Compute the appropriate direction. Increasing `dir_y` means going downwards.
            dir_x = targ_u - tool_u
            dir_y = targ_v - tool_v
            dist_pix = np.sqrt(dir_x**2 + dir_y**2)
            dist_pix_mm = dist_pix * PIX_TO_MM
            print('direction (x,y) in pixels: {}, {}'.format(dir_x, dir_y))
            print('distance: {:.3f} pix, ~ {:.3f} mm'.format(dist_pix, dist_pix_mm))

            # Record information. We store images so that we can see the final input.
            epis['cimgs'].append(cimg_proc)
            epis['targs'].append(targ_mask)
            epis['dists'].append(dist_mask)
            epis['tools'].append(tool_mask)
            epis['areas'].append(area_mask)
            epis['cimgs_draw'].append(draw_cimg)
            epis['pcls'].append(pcl_t)
            epis['ee_pose_b'].append(list(ee_pose_b))
            epis['tool_pose_b'].append(list(tool_pose_b))
            epis['tool_u'].append(tool_u)
            epis['tool_v'].append(tool_v)
            epis['targ_u'].append(targ_u)
            epis['targ_v'].append(targ_v)
            epis['dist_pix'].append(dist_pix)
            epis['dist_pix_mm'].append(dist_pix_mm)

            if dist_pix < PIX_THRESH:
                print('Dist_pix under threshold, exit.')
                break

            # Positive y means going 'down' in an image, but that's negative y in the world.
            posi_xy = np.array([dir_x, -dir_y], dtype=np.float64)  # note the negative
            posi_xy = posi_xy / np.linalg.norm(posi_xy)  # norm is 1 meter
            posi_xy = (posi_xy / 1000.0) * dist_pix_mm  # norm is `dist_pix_mm` millimeters

            # Check if safe before moving, and if not, try decreasing the posi_xy.
            new_ee_pose_b = ee_pose_b.copy()
            new_ee_pose_b[0] = ee_pose_b[0] + posi_xy[0]
            new_ee_pose_b[1] = ee_pose_b[1] + posi_xy[1]
            new_tool_pose_b = robot.ee_to_tool(new_ee_pose_b)
            attempts = 1
            max_attempts = 10
            while not (robot.is_safe(
                    ee_pose=new_ee_pose_b, tool_pose=new_tool_pose_b, bounds='policy')):
                print('Moving unsafe with xy change: {}.'.format(posi_xy))
                posi_xy = posi_xy * 0.9
                new_ee_pose_b[0] = ee_pose_b[0] + posi_xy[0]
                new_ee_pose_b[1] = ee_pose_b[1] + posi_xy[1]
                new_tool_pose_b = robot.ee_to_tool(new_ee_pose_b)
                attempts += 1
                if attempts == max_attempts:
                    print('Exiting, unable to make any safe movements.')
                    break

            if attempts < max_attempts:
                # CAREFUL! This is where the closed-loop behavior happens.
                print('Position change: {} (norm {:0.2f} mm) after {} attempts'.format(
                        posi_xy, np.linalg.norm(posi_xy) * 1000.0, attempts))
                print('old EE pose: {}'.format(ee_pose_b))
                print('new EE pose: {}'.format(new_ee_pose_b))
                robot.move_to_ee_pose(new_ee_pose_b, bounds='policy')
            else:
                posi_xy = np.array([0., 0.])

            # Record more info. Note: this may have 1 item fewer than the others.
            epis['posi_xy'].append(list(posi_xy))  # action in meters
            epis['attempts'].append(attempts)
            epis['new_ee_pose_b'].append(list(new_ee_pose_b))
            epis['new_tool_pose_b'].append(list(new_tool_pose_b))

            # Wait for the next action. This value needs to be tuned.
            rospy.sleep(args.rospy_action_wait)

        # ------------------------------------------------------------------------------- #
        # Lift and raise gripper to evaluate. Evaluation is going to be trickier when we
        # go beyond 1 target and 1 distractor, as it may be hard to do contour detection.
        # My thinking is to just move the ladle and then check if target is still visible.
        # It may fall out during the movement, though I think it doesn't happen that much.
        # ------------------------------------------------------------------------------- #
        rospy.sleep(1)
        print('\n------------------- Move gripper up to to evaluate -------------------')
        ee_pose = robot.get_ee_pose()
        ee_eval_pose_1 = [ee_pose[0], ee_pose[1], 0.410] + QUAT  # straight up
        ee_eval_pose_2 = [     0.655,     -0.063, 0.470] + QUAT  # move to center
        robot.move_to_ee_pose(ee_eval_pose_1, bounds='policy')
        rospy.sleep(0.5)
        robot.move_to_ee_pose(ee_eval_pose_2, bounds='policy')

        # Evaluate depth. Raw distance is in millimeters.
        eval_color_proc = robot.DC.c_image_proc
        eval_depth = robot.DC.d_image
        #U.print_debug(img=eval_depth, imgname='Eval depth (raw)')
        eval_depth = robot.DC.crop_img(eval_depth,
                x=robot.DC.crop_x, y=robot.DC.crop_y, w=robot.DC.crop_w, h=robot.DC.crop_h)
        #U.print_debug(img=eval_depth, imgname='Eval depth (crop)')
        # NOTE(daniel): this cutoff is important to ignore anything in the water!
        eval_depth_cutoff = U.process_depth(eval_depth, cutoff=580)
        eval_depth_mask = ((eval_depth_cutoff > 0) * 255).astype(np.uint8)[:,:,0]  # 1 channel
        eval_targ = robot.DC.targ_mask
        eval_dist = robot.DC.dist_mask
        targ_raised = np.bitwise_and(eval_targ, eval_depth_mask)  # targ. pixels above threshold
        dist_raised = np.bitwise_and(eval_dist, eval_depth_mask)  # dist. pixels above threshold
        targ_raised = U.triplicate(targ_raised, to_int=True)
        dist_raised = U.triplicate(dist_raised, to_int=True)

        # Stop recording and reset.
        rospy.sleep(1.0)
        robot.set_joint_acceleration(joint_accel=0.5)
        robot.reset_to_end(robot_z=ROBOT_Z, quat=QUAT, stop_recording=True)

        # Store data from this episode, should be consistent among different policies.
        print('\n---------------------- Now saving videos, etc. -----------------------')
        im_dict = {
            'eval_color_proc': eval_color_proc,
            'eval_depth': eval_depth,
            'eval_depth_cutoff': eval_depth_cutoff,
            'targ_raised': targ_raised,
            'dist_raised': dist_raised,
        }
        robot.save_episode_results(args=args, epis=epis, im_dict=im_dict, num_demo=num_demo)
        del robot.DC.c_image_l
        del robot.DC.pcl_l
        del robot.DC.eep_pose_l
        del robot.DC.eep_pose_p
        del robot.DC.eep_pose_r
        del robot.DC.c_image_proc_l
        del robot.DC.targ_mask_l
        del robot.DC.clr_imgs_l
        del robot.DC.d_image_l
        del robot.DC.dist_mask_l
        del robot.DC.tool_mask_l
        del robot.DC.area_mask_l

def test_algorithmic_policy_pix(args):
    """Test the algorithmic policy, can run up to `T = args.max_T` time steps.

    This one operates in pixel space and seems to be slightly more reliable than the
    world space version because the latter seems to not properly compute the world?
    Cropped images are 300x300, which corresponds to roughly 180mm in the real world.

    TODO(daniel): currently only works with 1 target, how to do multiple targets?
    """
    robot = SawyerRobot()
    rospy.sleep(3)

    # Will do translations only by keeping this as the quaternion.
    QUAT = [0.18,  0.70, -0.65,  0.16]
    ROBOT_Z = [0.360]
    T = args.max_T
    PIX_THRESH = 25

    # Reset the robot and enter ladle at the 'top' as viewed from the image.
    robot.reset_to_start(robot_z=ROBOT_Z, quat=QUAT, start_recording=True)

    # Algorithmic policy: iterate until pixel distance is under a threshold.
    print('\nRunning the algorithmic policy for up to {} steps.'.format(T))
    epis = defaultdict(list)
    for t in range(T):
        print('\n=========================== Action {}/{} ==========================='.format(t+1,T))
        ee_pose_b = robot.get_ee_pose()
        tool_pose_b = robot.get_tool_pose()
        points_world = np.reshape(tool_pose_b[:3], (1,3))
        uu,vv = robot.world_to_pixel(points_world=points_world)
        tool_u = uu[0] - robot.DC.crop_x
        tool_v = vv[0] - robot.DC.crop_y

        # Mask is on the cropped image to save time and to ignore stuff outside container.
        cimg_proc = (robot.DC.c_image_proc).copy()
        targ_mask = (robot.DC.targ_mask).copy()
        dist_mask = (robot.DC.dist_mask).copy()
        tool_mask = (robot.DC.tool_mask).copy()
        area_mask = (robot.DC.area_mask).copy()

        # Get the point cloud data, hopefully roughly time-aligned with images.
        pcl_t = (robot.DC.pcl).copy()

        # This lets us use cv2.circle(targ_mask,(u,v)) to visualize the target.
        mask_idxs = np.where(targ_mask > 0)
        targ_u = int(np.median(mask_idxs[1]))
        targ_v = int(np.median(mask_idxs[0]))

        # Debugging. See other methods for debugging with images, etc.
        print('tool position (world): {}'.format(points_world))
        print('tool u,v: {} {} (on cropped image)'.format(tool_u, tool_v))
        print('targ u,v: {} {} (on cropped image)'.format(targ_u, targ_v))
        draw_cimg = cimg_proc.copy()
        cv2.circle(draw_cimg, (targ_u,targ_v), radius=10, color=(0,255,0), thickness=-1)
        cv2.circle(draw_cimg, (tool_u,tool_v), radius=10, color=(255,0,0), thickness=-1)

        # Compute the appropriate direction. Increasing `dir_y` means going downwards.
        dir_x = targ_u - tool_u
        dir_y = targ_v - tool_v
        dist_pix = np.sqrt(dir_x**2 + dir_y**2)
        dist_pix_mm = dist_pix * PIX_TO_MM
        print('direction (x,y) in pixels: {}, {}'.format(dir_x, dir_y))
        print('distance: {:.3f} pix, ~ {:.3f} mm'.format(dist_pix, dist_pix_mm))

        # Record information. We store images so that we can see the final input.
        epis['cimgs'].append(cimg_proc)
        epis['targs'].append(targ_mask)
        epis['dists'].append(dist_mask)
        epis['tools'].append(tool_mask)
        epis['areas'].append(area_mask)
        epis['cimgs_draw'].append(draw_cimg)
        epis['pcls'].append(pcl_t)
        epis['ee_pose_b'].append(list(ee_pose_b))
        epis['tool_pose_b'].append(list(tool_pose_b))
        epis['tool_u'].append(tool_u)
        epis['tool_v'].append(tool_v)
        epis['targ_u'].append(targ_u)
        epis['targ_v'].append(targ_v)
        epis['dist_pix'].append(dist_pix)
        epis['dist_pix_mm'].append(dist_pix_mm)

        if dist_pix < PIX_THRESH:
            print('Dist_pix under threshold, exit.')
            break

        # Positive y means going 'down' in an image, but that's negative y in the world.
        posi_xy = np.array([dir_x, -dir_y], dtype=np.float64)  # note the negative
        posi_xy = posi_xy / np.linalg.norm(posi_xy)  # norm is 1 meter
        posi_xy = (posi_xy / 1000.0) * dist_pix_mm  # norm is `dist_pix_mm` millimeters

        # Check if safe before moving, and if not, try decreasing the posi_xy.
        new_ee_pose_b = ee_pose_b.copy()
        new_ee_pose_b[0] = ee_pose_b[0] + posi_xy[0]
        new_ee_pose_b[1] = ee_pose_b[1] + posi_xy[1]
        new_tool_pose_b = robot.ee_to_tool(new_ee_pose_b)
        attempts = 1
        max_attempts = 10
        while not (robot.is_safe(
                ee_pose=new_ee_pose_b, tool_pose=new_tool_pose_b, bounds='policy')):
            print('Moving unsafe with xy change: {}.'.format(posi_xy))
            posi_xy = posi_xy * 0.9
            new_ee_pose_b[0] = ee_pose_b[0] + posi_xy[0]
            new_ee_pose_b[1] = ee_pose_b[1] + posi_xy[1]
            new_tool_pose_b = robot.ee_to_tool(new_ee_pose_b)
            attempts += 1
            if attempts == max_attempts:
                print('Exiting, unable to make any safe movements.')
                break

        if attempts < max_attempts:
            # CAREFUL! This is where the closed-loop behavior happens.
            print('Position change: {} (norm {:0.2f} mm) after {} attempts'.format(
                    posi_xy, np.linalg.norm(posi_xy) * 1000.0, attempts))
            print('old EE pose: {}'.format(ee_pose_b))
            print('new EE pose: {}'.format(new_ee_pose_b))
            robot.move_to_ee_pose(new_ee_pose_b, bounds='policy')
        else:
            posi_xy = np.array([0., 0.])

        # Record more info. Note: this may have 1 item fewer than the others.
        epis['posi_xy'].append(list(posi_xy))  # action in meters
        epis['attempts'].append(attempts)
        epis['new_ee_pose_b'].append(list(new_ee_pose_b))
        epis['new_tool_pose_b'].append(list(new_tool_pose_b))

        # Wait for the next action. This value needs to be tuned.
        rospy.sleep(args.rospy_action_wait)

    # ------------------------------------------------------------------------------- #
    # Lift and raise gripper to evaluate. Evaluation is going to be trickier when we
    # go beyond 1 target and 1 distractor, as it may be hard to do contour detection.
    # My thinking is to just move the ladle and then check if target is still visible.
    # It may fall out during the movement, though I think it doesn't happen that much.
    # ------------------------------------------------------------------------------- #
    rospy.sleep(2)
    print('\n------------------- Move gripper up to to evaluate -------------------')
    ee_pose = robot.get_ee_pose()
    ee_eval_pose_1 = [ee_pose[0], ee_pose[1], 0.410] + QUAT  # straight up
    ee_eval_pose_2 = [     0.655,     -0.063, 0.470] + QUAT  # move to center
    robot.move_to_ee_pose(ee_eval_pose_1, bounds='policy')
    rospy.sleep(0.5)
    robot.move_to_ee_pose(ee_eval_pose_2, bounds='policy')

    # Evaluate depth. Raw distance is in millimeters.
    eval_color_proc = robot.DC.c_image_proc
    eval_depth = robot.DC.d_image
    #U.print_debug(img=eval_depth, imgname='Eval depth (raw)')
    eval_depth = robot.DC.crop_img(eval_depth,
            x=robot.DC.crop_x, y=robot.DC.crop_y, w=robot.DC.crop_w, h=robot.DC.crop_h)
    #U.print_debug(img=eval_depth, imgname='Eval depth (crop)')
    # NOTE(daniel): this cutoff is important to ignore anything in the water!
    eval_depth_cutoff = U.process_depth(eval_depth, cutoff=580)
    eval_depth_mask = ((eval_depth_cutoff > 0) * 255).astype(np.uint8)[:,:,0]  # 1 channel
    eval_targ = robot.DC.targ_mask
    eval_dist = robot.DC.dist_mask
    targ_raised = np.bitwise_and(eval_targ, eval_depth_mask)  # targ. pixels above threshold
    dist_raised = np.bitwise_and(eval_dist, eval_depth_mask)  # dist. pixels above threshold
    targ_raised = U.triplicate(targ_raised, to_int=True)
    dist_raised = U.triplicate(dist_raised, to_int=True)

    # Stop recording and reset.
    rospy.sleep(2.0)
    robot.reset_to_end(robot_z=ROBOT_Z, quat=QUAT, stop_recording=True)

    # Store data from this episode, should be consistent among different policies.
    print('\n---------------------- Now saving videos, etc. -----------------------')
    im_dict = {
        'eval_color_proc': eval_color_proc,
        'eval_depth': eval_depth,
        'eval_depth_cutoff': eval_depth_cutoff,
        'targ_raised': targ_raised,
        'dist_raised': dist_raised,
    }
    robot.save_episode_results(args=args, epis=epis, im_dict=im_dict)

def rotator(pc_file):
    #* Function to rotate the ladel model intob the upright position, so that it can then be translated and transformed into the end-effector frame
    pcd_og = o3d.io.read_point_cloud(pc_file)
    #* The 0.0941 is the scale that we computed from CloudCompare. Check this block of updates in my Notion: https://www.notion.so/Tool-Flow-Experiments-1e3be6c2a51b470f88abf4d1934b93dc#0c2d84a5ee8a4281a143672adc8c7402
    pcd_og = np.asarray(pcd_og.points) * 0.0941
    pcd_np = pcd_og.copy()
    #*Rotation to get it turn it
    rot_matrix_1 = np.array([[1, 0, 0], [0, np.cos(120 * np.pi/180), -np.sin(120 * np.pi/180)], [0, np.sin(120 * np.pi/180), np.cos(120 * np.pi/180)]])
    #* Rotation to get it upright 
    rot_matrix_2 = np.array([[np.cos(90 * np.pi/180), -np.sin(90 * np.pi/180), 0], [np.sin(90 * np.pi/180), np.cos(90 * np.pi/180), 0 ], [0, 0, 1]])

    pcd_np = np.matmul(pcd_np.copy(), rot_matrix_1)
    pcd_np = np.matmul(pcd_np.copy(), rot_matrix_2)

    #* It seems like the center of the ladle is offset from its tip. Just eyeballing it and it seems about right
    #! NOTE(sarthak): This may be one of the reasons why the ladel looks shifted. If you look at the ladle on the robot
    #! it isn't gripped at the end, but rather at some distance through the tool holder.
    pcd_np[:, 2] += 0.35
    return pcd_np

def run_inference(args):

    # pco_dir = 'data/policy_hum_t_demo_ntarg_01_ndist_00_maxT_10/data/policy_data_125/'
    # pco_files = glob(join(pco_dir, 'pco_0_*'))
    # pco_files.sort(key = lambda x: (int(basename(x).split('.')[0][4:].split('_')[0]), int(basename(x).split('.')[0][4:].split('_')[1])))

    # print(pco_files[pco_files[0][:, 3] == 1].shape)

    if args.dslr:
        camera = VideoRecorder(save_dir=args.data_dir)
    if args.side_cam:
        side_cam = DoneCam(save_dir=args.data_dir)

    robot = SawyerRobot(policy=args.policy)
    rospy.sleep(2)

    height_idx = 0

    ROBOT_Z_CHOICES = [0.360, 0.450]
    ROBOT_Z = [ROBOT_Z_CHOICES[height_idx]]

    y_idx = 0

    ROBOT_Y_CHOICES = [-0.0207, -0.0409]
    ROBOT_Y = [ROBOT_Y_CHOICES[y_idx]]

    QUAT = [-0.1823, -0.724, 0.6477, -0.1518]

    context = zmq.Context()
    obs_socket = context.socket(zmq.PUB)
    obs_socket.setsockopt(zmq.SNDHWM, 0)
    obs_socket.bind("tcp://127.0.0.1:2024")

    print('Established output socket')

    if args.dslr:
        camera.start_movie()

    # robot.reset_to_rotated_start(robot_z=ROBOT_Z, quat=QUAT, start_recording=True, rotations = True)
    robot.reset_to_start(robot_z=ROBOT_Z, robot_y = ROBOT_Y, quat=QUAT, start_recording=True)
    act_socket = context.socket(zmq.SUB)
    act_socket.subscribe("")
    tunnel = ssh.tunnel_connection(act_socket, "tcp://127.0.0.1:5698", "sarthak@omega.rpad.cs.cmu.edu")

    print('Connected to input socket')

    obs_socket.send_pyobj("wakeup")
    time.sleep(1)

    #* This will scale and rotate the model in place
    model_ladle = rotator('dense_ladel.pcd')

    start = time.time()
    t = 0
    num_demos = 0
    while num_demos < 5:
        while True:
            try:
                if args.side_cam:
                    side_cam.capture()
                obs = (robot.DC.pcl).copy()
                # I have some concerns about this eepose
                # info = robot.get_ee_pose()
                info = robot.DC.eep_pose_l[-1]

                pts_targ = np.where(obs[:,3] == 0.0)[0]
                pts_dist = np.where(obs[:,3] == 1.0)[0]
                pts_tool = np.where(obs[:,3] == 2.0)[0]

                #! ---- Starting analytical ladle integration -----

                ladle_transform = robot.buffer.lookup_transform('base', 'right_gripper_base', rospy.Time(0))

                ladle_position = np.array([ladle_transform.transform.translation.x, ladle_transform.transform.translation.y, ladle_transform.transform.translation.z])
                ladle_rotation = quat(ladle_transform.transform.rotation.w, ladle_transform.transform.rotation.x, ladle_transform.transform.rotation.y, ladle_transform.transform.rotation.z)

                #* Converting to a rotation matrix from the quaternion
                ladle_rotation_matrix = ladle_rotation.rotation_matrix

                pcd_np = np.matmul(model_ladle, ladle_rotation_matrix.transpose()) + ladle_position

                if args.max_points == 1400:
                    # If the max_points is set to 1400, then the model was trained on the algorithmic/newer datasets
                    choice = np.random.choice(len(pcd_np), size=1400, replace=False)
                    new_pcl = np.zeros((1400, 4))
                    new_pcl[:, :3] = pcd_np[choice]
                    new_pcl[:,  3] = 2.0
                    new_pcl[:min(300, len(pts_targ))] = obs[pts_targ][:min(300, len(pts_targ))]
                    new_pcl[min(300, len(pts_targ)):min(300, len(pts_targ)) + min(300, len(pts_dist))] = obs[pts_dist][:min(300, len(pts_dist))]
                elif args.max_points == 1200:
                    # If the max_points is set to 1400, then the model was trained on the older human demonstrator datasets
                    choice = np.random.choice(len(pcd_np), size=1200, replace=False)
                    new_pcl = np.zeros((1200, 4))
                    new_pcl[:, :3] = pcd_np[choice]
                    new_pcl[:,  3] = 2.0
                    new_pcl[:min(100, len(pts_targ))] = obs[pts_targ][:min(100, len(pts_targ))]
                    new_pcl[min(100, len(pts_targ)):min(100, len(pts_targ)) + min(100, len(pts_dist))] = obs[pts_dist][:min(100, len(pts_dist))]
                else:
                    print('Check your arguments. max_points is not 1200 or 1400. It is: {}'.format(args.max_points))
                    return False

                one_hot_encoding = np.eye(3)[new_pcl[:,3].astype(int)]
                '''This below slicing operation for the column just returns the columnn that contains the tool-encoding'''
                if args.obs_dim == 4:
                    new_obs = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3]])
                elif args.obs_dim == 5:
                    new_obs = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3], one_hot_encoding[:, 0:1]])
                elif args.obs_dim == 6:
                    new_obs = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3], one_hot_encoding[:, 0:1], one_hot_encoding[:, 1:2]])

                # Checking what actually gets conveyed to the tool, we can plot these files later in the pcer.py function
                np.save(join(args.data_dir, 'obs_{}_{}.npy'.format(num_demos, t)).format(t), new_obs)

                #! ---- Ending analytical ladle integration -----

                # unicode_obs = np.char.decode(new_obs.astype(np.bytes_), 'UTF-8')
                # unicode_info = np.char.decode(info.astype(np.bytes_), 'UTF-8')

                # unicode_t = str(t).decode("utf-8")

                # obs_dict = {
                #     "id": unicode_t,
                #     "obs": unicode_obs,
                #     "info": unicode_info,
                # }

                # obs_socket.send_pyobj(obs_dict)

                # Convert each thing into bytes
                def np_to_bytes(arr):
                    buf = io.BytesIO()
                    np.lib.format.write_array(buf, arr, allow_pickle=False)
                    return buf.getvalue()

                # loaded_pco = np.load(pco_files[t])

                # print('loaded_pco_shape: {}'.format(loaded_pco.shape))
                # print('loaded_pco_shape tools', loaded_pco[loaded_pco[:, 3] == 1].shape)

                # print('observed observation obs_shape: {}'.format(new_obs.shape))
                # print('observed tools', new_obs[new_obs[:, 3] == 1].shape)

                # obs_bytes = np_to_bytes(loaded_pco)
                obs_bytes = np_to_bytes(new_obs)
                info_bytes = np_to_bytes(info)
                t_bytes = struct.pack('q', t)

                obs_socket.send_multipart([obs_bytes, info_bytes, t_bytes])

                # action = act_socket.recv_pyobj()
                def bytes_to_np(raw):
                    buf = io.BytesIO(raw)
                    arr = np.lib.format.read_array(buf, allow_pickle=False)
                    buf.close()
                    return arr

                in_raw = act_socket.recv_multipart(copy=True)
                in_t = struct.unpack('q', in_raw[0])[0]
                in_action = bytes_to_np(in_raw[1])
                in_model = split(in_raw[2])
                action = {
                    'id': in_t,
                    'action': in_action,
                    'model': in_model
                }

                delta_translation = (action['action'][:3]).copy()

                apply_translation = delta_translation.copy()
                current_xyz = info[:3].copy()

                next_xyz = current_xyz + apply_translation

                # print(action['action'][3:])

                # delta_quat = quat(axis = [action['action'][3], action['action'][4], action['action'][5]], angle = 1.0)

                # info_quat = quat(info[3], info[4], info[5], info[6])

                # next_quat = delta_quat * info_quat

                # complete_pose = np.append(next_xyz, [next_quat[0], next_quat[1], next_quat[2], next_quat[3]])

                # print(delta_quat, info_quat)

                complete_pose = np.append(next_xyz, np.array(QUAT).copy())

                # print('number of actions: ', t)
                robot.move_to_ee_pose(complete_pose, bounds='policy')

                ''''Setting this parameter to 3 seconds, since it seems that the latest pointclouds
                are not the same one that the machine runs inference on, so there's a mismatch between
                the actual scene and the generated action'''
                rospy.sleep(2.0)

                t+=1

            except KeyboardInterrupt:

                if args.dslr:
                    camera.stop_movie()
                    camera.exit()

                if args.side_cam:
                    side_cam.done()

                status = str(input('Demo status: '))
                np.save(join(args.data_dir, 'final_pcl_actions_status_{}_{}_height_{}_y_{}_model_{}.npy').format(status, t, height_idx, y_idx, action['model'][1:]), new_obs.copy())

                num_demos+=1

                t = 0

                #* Moving on to the next demonstration
                rospy.loginfo('Moving to next demonstration!')
                #* Borrowed this block of code from another one of Daniel's function where we want the tool to rotate about 
                #* the axis as well once the reset has been triggered
                ee_poses_reset = [[0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], [0.65, -0.0707, 0.450, 0.18, 0.70, -0.65, 0.16]]

                for pose in ee_poses_reset:
                    #* Executing the rotation to drop the ladle into the bowl again. Way too hard to do this kinesthetically
                    robot.move_to_ee_pose(pose, bounds='resets')
                    rospy.sleep(1)

                robot.reset_to_start(ROBOT_Z, QUAT, robot_y = ROBOT_Y, start_recording=False)
                rospy.loginfo('Robot is done with reset.')
                rospy.sleep(1)
                break

    return 0

def kinesthetic_demonstrator(args):
    rospy.loginfo('Inside BC Collection Function')
    #* Sarthak: What needs to go in here?
    # You need to accept a key from the disc, if this key says stop then 
    # stop the data collection, otherwise continue with recording the point cloud data
    # Breakdown of what this function actually does?
    # 1. Once the code starts run a while loop until the recieved key 0xFF is not q.
    # 2. Collect the point cloud at each instance and save it to disc.
    # This is the basic framework
    rospy.loginfo('Collecting data for Behavior Cloning')
    #* This is the height that Daniel has fixed for the default start position for the ladle height
    ROBOT_Z = [0.450]
    QUAT = [0.18, 0.70, -0.65, 0.16]
    # UPRIGHT_QUAT = [0.005, -0.739, 0.674, -0.004]

    #* Taking the number of demonstrations for which the data has to be collected. Set in argparse. Defaults to 5
    num_demos = args.num_demos
    robot = SawyerRobot(args.policy)
    rospy.sleep(2)
    
    if args.rotations_demo == True:
        ROBOT_Z = [0.360]
        '''Support for the rotation included translation'''
        robot.reset_to_rotated_start(robot_z=ROBOT_Z, quat=QUAT, start_recording=True, rotations = True)
    else:
        robot.reset_to_start(robot_z=ROBOT_Z, quat=QUAT, start_recording=True)

    for num_demo in range(num_demos):
        rospy.loginfo('Size of PCL: {} Size of EEP: {} Size of LDP: {} Size of LDR: {} Size of depth: {} Size of img: {}'.format(len(robot.DC.pcl_l), len(robot.DC.eep_pose_l), len(robot.DC.eep_pose_p), len(robot.DC.eep_pose_r), len(robot.DC.d_image_l), len(robot.DC.clr_imgs_l)))

        '''Making policy folder here so that it's compatible with the rest of the visualization.py code'''
        policy_data_dir = join(args.data_dir, 'policy_data')
        if num_demo == 0:
            os.makedirs(policy_data_dir)

        if not args.rotations_demo:
            robot.DC.pcl_l = []
            robot.DC.eep_pose_l = []
            robot.DC.eep_pose_p = []
            robot.DC.eep_pose_r = []
            robot.DC.clr_imgs_l = []
            robot.DC.d_image_l = []
            robot.DC.pcl_l_header = []
            robot.DC.eep_pose_l_header = []
            robot.DC.clr_imgs_header_l = []
            robot.DC.d_image_header_l = []

        #* Iterating through the demos, whos value we've set in the argparse
        rospy.loginfo('Recording Demo: {} Size of PCL: {} Size of EEP: {} Size of LDP: {} Size of LDR: {}'.format(num_demo, len(robot.DC.pcl_l), len(robot.DC.eep_pose_l), len(robot.DC.eep_pose_p), len(robot.DC.eep_pose_r)))
        #* iterations keeps track of the number of actions being collected for each demonstration
        iterations = 0
        #! This variable keeps track of whether or not we've been rotated by pi/2
        # acheived = 0
        # start_joint_angle = robot.limb.joint_angle('right_j6')
        while True:
            #* Setting the impedance to either all unconstrained or traslation only
            try:
                #! Commented code below is to get 4DoF working. Commented because we might use it and I spent way too much time figuring it out
                # if (-2.0 < robot.limb.joint_angle('right_j6') < 0.70) and acheived == 0:
                #     print('here', robot.limb.joint_angle('right_j6'), start_joint_angle, abs(robot.limb.joint_angle('right_j6') - start_joint_angle))
                #     # robot.limb.set_joint_positions({'right_j6':start_joint_angle + np.pi/2})
                #     robot.set_impedence(1)
                    #* Do the rotation about the z axis
                #* As Daniel also mentioned in his functions, hopefully these are time synced with the pointcloud observations
                # elif abs(robot.limb.joint_angle('right_j6') - start_joint_angle) < np.pi/2 and acheived == 0:
                #     print('here', robot.limb.joint_angle('right_j6'), start_joint_angle, abs(robot.limb.joint_angle('right_j6') - start_joint_angle))
                #     # robot.limb.set_joint_positions({'right_j6':start_joint_angle + np.pi/2})
                #     robot.set_impedence(1)
                # else:
                #     acheived = 1
                    # robot.set_impedence()

                # robot.set_impedence()
                #* Here we look up the transformation between the robot base and the end-effector reference point on the base plate using ROS' TF functionality
                ladle_transform = robot.buffer.lookup_transform('base', 'right_gripper_base', rospy.Time(0))

                #* Extracting the ladle position and rotation from the transformation that we just looked up
                ladle_position = np.array([ladle_transform.transform.translation.x, ladle_transform.transform.translation.y, ladle_transform.transform.translation.z])
                ladle_rotation = quat(ladle_transform.transform.rotation.w, ladle_transform.transform.rotation.x, ladle_transform.transform.rotation.y, ladle_transform.transform.rotation.z)

                #* Converting to a rotation matrix from the quaternion
                ladle_rotation_matrix = ladle_rotation.rotation_matrix

                #* Here we collect the pointcloud and the RGB image from the DataCollector class
                # img = (robot.DC.get_color_image()).copy()

                #! Checking to see if removing any print statements might reduce ROS lag
                # rospy.loginfo('Demo: {} Iteration Number: {}'.format(num_demo, iterations))

                # cv2.imwrite(join(args.data_dir, 'img_{}_{}.png').format(num_demo, iterations), img)

                #! This might be causing the issue in the shifting ladle. I will check if I can swap this out for the pose that we collected earlier
                # np.save(join(args.data_dir, 'eep_{}_{}.npy').format(num_demo, iterations), robot.get_ee_pose())

                iterations+=1

                #* We set this to 0 so that we can collect "finely spaced" data
                rospy.sleep(0.1)
            except KeyboardInterrupt:
                assert len(robot.DC.eep_pose_r) == len(robot.DC.pcl_l) == len(robot.DC.eep_pose_p) == len(robot.DC.eep_pose_l) == len(robot.DC.clr_imgs_l) == len(robot.DC.d_image_l), 'Check your callback functions. Some lists are not equally shaped. EEP: {} LDR: {} LDP: {} PCL: {} IMG: {} DEPTH: {}'.format(len(robot.DC.eep_pose_l), len(robot.DC.eep_pose_r), len(robot.DC.eep_pose_p), len(robot.DC.pcl_l), len(robot.DC.clr_imgs_l), len(robot.DC.d_image_l))
                #* Moving on to the next demonstration
                #* Saving everything that we just collected
                rospy.loginfo('Moving to next demonstration! Size of PCL: {} Size of EEP: {} Size of LDP: {} Size of LDR: {} Size of depth: {} Size of img: {}'.format(len(robot.DC.pcl_l), len(robot.DC.eep_pose_l), len(robot.DC.eep_pose_p), len(robot.DC.eep_pose_r), len(robot.DC.d_image_l), len(robot.DC.clr_imgs_l)))
                # Using the len of the robot.DC.pcl_l here for the iterators of all the other lists as well since between the saving of the pcl's to disc
                # and the saving of the next list, the sync code can actually add another set of img/ldr/ldp/eeps. So we fix the value of the pcl_l list at the 
                # start of the for loop and then use that as a fixed list length to save the rest of the lists as well. Even though the previous assert will pass, 
                #  a new callback can lead to the sizes of the lists not being equal. This will trip additional checks in the visualization code.
                for pcl in range(len(robot.DC.pcl_l)):
                    np.save(join(policy_data_dir, 'pcl_{}_{}_{}_{}.npy').format(num_demo, pcl, robot.DC.pcl_l_header[pcl]['secs'], robot.DC.pcl_l_header[pcl]['nsecs']), robot.DC.pcl_l[pcl])
                    np.save(join(policy_data_dir, 'eep_{}_{}_{}_{}.npy').format(num_demo, pcl, robot.DC.eep_pose_l_header[pcl]['secs'], robot.DC.eep_pose_l_header[pcl]['nsecs']), robot.DC.eep_pose_l[pcl])
                    np.save(join(policy_data_dir, 'ldp_{}_{}.npy').format(num_demo, pcl), robot.DC.eep_pose_p[pcl])
                    np.save(join(policy_data_dir, 'ldr_{}_{}.npy').format(num_demo, pcl), robot.DC.eep_pose_r[pcl])
                    cv2.imwrite(join(policy_data_dir, 'img_{}_{}_{}_{}.png').format(num_demo, pcl, robot.DC.clr_imgs_header_l[pcl]['secs'], robot.DC.clr_imgs_header_l[pcl]['nsecs']), robot.DC.clr_imgs_l[pcl])

                #* Borrowed this block of code from another one of Daniel's function where we want the tool to rotate about 
                #* the axis as well once the reset has been triggered
                ee_poses_reset = [[0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998]]
                # ee_poses_reset = [[0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], [0.6443, -0.0366, 0.360, -0.2201, -0.7134,  0.6346, -0.1998]]
                # ee_poses_reset = [[0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], [0.65, -0.0707, 0.450, 0.18, 0.70, -0.65, 0.16]]

                for pose in ee_poses_reset:
                    #* Executing the rotation to drop the ladle into the bowl again. Way too hard to do this kinesthetically
                    robot.move_to_ee_pose(pose, bounds='resets')
                    rospy.sleep(1)

                robot.reset_to_start(ROBOT_Z, QUAT, start_recording=False)
                rospy.loginfo('Robot is done with reset.')
                rospy.sleep(1)
                break
    return 0

# TODO(daniel)
def test_algorithmic_policy_world(args):
    """Test the algorithmic policy.

    Now by using the depth images and computing world position of the ball, then we
    estimate an offset for the robot to go underneath, etc.

    To interpret this: u and v represent image pixels, so u is the x coordinate, v
    is the y coordinate, but the pixels are centered in the upper left in the image,
    so when we visualize these, increasing u goes rightward (as expected) but
    increasing v goes downwards. If we index into a numpy array, we need img[v,u]
    and not the other way around.
    """
    robot = SawyerRobot()
    rospy.sleep(3)

    # Will do translations only by keeping this as the quaternion.
    quat = [0.18,  0.70, -0.65,  0.16]
    robot_z = [0.360]

    # NEW: Iterate until distance is below some threshold.
    # This is where the 'algorithmic policy' is in action!!!
    print('\nNow the algorithmic policy ...')
    ee_pose_b = robot.get_ee_pose()
    tool_pose_b = robot.get_tool_pose()
    points_world = np.reshape(tool_pose_b[:3], (1,3))
    uu,vv = robot.world_to_pixel(points_world=points_world)
    tool_u = uu[0]
    tool_v = vv[0]

    # Note: this is done on the cropped image so that (a) it saves time, and (b) more
    # importantly, the mask isn't obscured by things that could be outside the range.
    cimg_full = robot.DC.c_image
    targ_mask = robot.DC.targ_mask
    mask_idxs = np.where(targ_mask > 0)
    # This is from the cropped version, and set to be (1,0) so we use (medi_u,medi_v)
    # on an image and it has a consistent interpretation as with my earlier usage.
    medi_u = int(np.median(mask_idxs[1]))
    medi_v = int(np.median(mask_idxs[0]))

    # NOW we should find the corresponding world coordinate of that mask.
    # Due to noise maybe we should empirically find a range and exclude NaNs or 0s?
    # NOTE: with the current ROS topics we use, I get values that must be in mm. I think
    # this has to be converted into meters?
    depth_raw = robot.DC.d_image
    depth_proc = robot.DC.d_image_proc
    print('depth_raw min/max/mean/medi: {:0.1f} {:0.1f} {:0.1f} {:0.1f}'.format(
        np.min(depth_raw), np.max(depth_raw), np.mean(depth_raw), np.median(depth_raw)
    ))
    depth_raw = depth_raw / 1000.0
    # Now let's try and get the mask pixels w.r.t. original image.
    mask_u = medi_u + robot.DC.crop_x
    mask_v = medi_v + robot.DC.crop_y

    # Index into the array using (v,u) ordering.
    depth_targ_arr = depth_raw[mask_v-5:mask_v+5, mask_u-5:mask_u+5]
    depth_tool_arr = depth_raw[tool_v-5:tool_v+5, tool_u-5:tool_u+5]
    d_targ = np.median(depth_targ_arr)
    d_tool = np.median(depth_tool_arr)
    print(depth_tool_arr)

    # Debugging. See other methods for debugging with images, etc.
    print('tool position (world): {}'.format(points_world))
    print('tool: (u,v) {} {}, depth {:0.3f}'.format(tool_u, tool_v, d_tool))
    print('mask: (u,v) {} {}, depth {:0.3f}'.format(mask_u, mask_v, d_targ))
    #print('medi: (u,v) {} {}'.format(medi_u, medi_v))
    u_pix = np.array([mask_u, tool_u])
    v_pix = np.array([mask_v, tool_v])
    z_dep = np.array([d_targ, d_tool])
    # Why do I need to swap v and u here? A bit misleading.
    world_pos = robot.pixel_to_world(u=v_pix, v=u_pix, z=z_dep)
    print('world_pos:\n{}'.format(world_pos))

    # Try getting world coordinate from pixels that we converted, to see if world
    # to pixel to world works? TODO.

    cv2.circle(cimg_full, (mask_u,mask_v), radius=15, color=(0,255,0), thickness=-1)
    cv2.circle(cimg_full, (tool_u,tool_v), radius=15, color=(255,0,0), thickness=-1)
    cv2.imwrite('test_cimg_full.png', cimg_full)
    cv2.circle(depth_proc, (mask_u,mask_v), radius=15, color=(0,255,0), thickness=-1)
    cv2.circle(depth_proc, (tool_u,tool_v), radius=15, color=(255,0,0), thickness=-1)
    cv2.imwrite('test_depth_proc.png', depth_proc)

    # Return now, we just want to get world coordinates of the balls and the
    # tool here (debugging), before we get to actual robot movement.
    # It seems like depth is poorly approximated.
    return

    # Initialize by entering the ladle anywhere (assume position doesn't really matter).
    # Same but what if we keep quaternions the same? Actually this isn't too bad!
    # This is promising as we can keep translations simpler.
    ee_poses_init = [
        MM_HOME_POSE_EE,  # home position
        [0.65, -0.0207, 0.475,  0.18,  0.70, -0.65,  0.16],  # close to water
        [0.65, -0.0409] + robot_z + quat,  # enter!
    ]
    print('Initializing!')
    for bpose in ee_poses_init:
        rospy.sleep(1)
        print('Now about to go to: {}'.format(bpose))
        robot.move_to_ee_pose(bpose)


    ## # From this compute the direction. Then I think it's a matter of either moving
    ## # a fixed (small) distance, or we just go until we get to some target.
    ## dir_x = med_u - tool_u
    ## dir_y = med_v - tool_v
    ## dist_pix = np.sqrt(dir_x**2 + dir_y**2)
    ## print('direction (x,y) in pixels: {}, {}'.format(dir_x, dir_y))
    ## print('if thinking of image, increasing x = rightward, increasing y = downward')
    ## print('pixel dist (norm): {:.3f}'.format(dist_pix))  # have to tune

    ## # Wait ... for y, positive y means going 'down' in the image (or going left in
    ## # code) but that actually means going negative y in code. I think we need to adjust.
    ## posi_xy = np.array([dir_x, -dir_y], dtype=np.float32)  # don't negate here
    ## posi_xy = posi_xy / np.linalg.norm(posi_xy)
    ## print('norm direction / translation in EE space: {}'.format(posi_xy))

    ## # Now divide so the actual motion is roughly on the order of 1 cm?
    ## # Remember, posi_xy is 'normed' to be 1 METER! This would actually be easier
    ## # if we just got the 3D world position of the ball, actually ...
    ## posi_xy = posi_xy / 50.0  # means norm is 2 cm
    ## new_ee_pose_b = ee_pose_b.copy()
    ## new_ee_pose_b[0] = ee_pose_b[0] + posi_xy[0]
    ## new_ee_pose_b[1] = ee_pose_b[1] + posi_xy[1]
    ## print('old EE pose: {}'.format(ee_pose_b))
    ## print('new EE pose: {}'.format(new_ee_pose_b))
    ## # Keep commented out until we can confirm
    ## robot.move_to_ee_pose(new_ee_pose_b)

    ## # Lift and raise gripper to evaluate.
    ## rospy.sleep(2)
    ## print('Now moving gripper to evaluate!')
    ## ee_eval_pose = [0.6559, -0.0628, 0.4236] + quat
    ## robot.move_to_ee_pose(ee_eval_pose)
    ## print('TODO: actually have to evaluate!')  # TODO(daniel) can we automate this?
    ## # My thinking is to just move the ladle and then check if the ball is still visible.
    ## # It may fall out during the movement, though I think it doesn't happen that much.

    # Perform the reset procedure to initialize for the next configuration.
    poses_reset = [
        [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # lift more (better for rot, updated 03/20)
        [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation
        [0.5914,  0.07  , 0.5202,  0.3514,  0.2479,  0.7746, -0.4638], # heavy rotation, more extreme
        [0.6077,  0.0506, 0.4815, -0.2355, -0.207 , -0.8495,  0.4243], # heavy rotation (duplicate)
        [0.6443, -0.0366, 0.5143, -0.2201, -0.7134,  0.6346, -0.1998], # return to this duplicate
        MM_HOME_POSE_EE,
    ]

    print('Now resetting!')
    for bpose in poses_reset:
        rospy.sleep(1)
        robot.move_to_ee_pose(bpose)
    print('DONE!')


# ---------------------------------------------------------------------------------- #
# Command line arguments.
# ---------------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/debug')
    p.add_argument('--n_targ', type=int, default=1)
    p.add_argument('--n_dist', type=int, default=0)
    p.add_argument('--num_demos', type=int, default=10)
    p.add_argument('--max_T', type=int, default=10)
    p.add_argument('--obs_dim', type=int, default=5)
    p.add_argument('--max_points', type=int, default=1400)
    p.add_argument('--dslr', action='store_true')
    p.add_argument('--side_cam', action='store_true')
    p.add_argument('--rospy_action_wait', type=float, default=0.25,
        help='Delay between two actions, this is a bit subtle...')
    p.add_argument('--policy', type=str, default='alg_pix')
    args = p.parse_args()

    assert args.policy in ['alg_pix', 'alg_t_demo', 'hum_t_demo', 'run_inference', 'alg_t_simple', 'scripted_rotation'], \
        'Error, policy {} not supported'.format(args.policy)

    # Manage data directory.
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.data_dir = join(os.getcwd(), args.data_dir, date)
    if (args.n_targ > 0) or (args.n_dist > 0):
        newstr = 'policy_{}_ntarg_{}_ndist_{}_maxT_{}'.format(
            args.policy, str(args.n_targ).zfill(2), str(args.n_dist).zfill(2),
            str(args.max_T).zfill(2))
        args.data_dir = args.data_dir.replace('debug', newstr)

    args.rotations_demo = False
    if args.policy == 'scripted_rotation':
        '''If using the scripted_rotation then trigger this condition, to change the 
        rotation start condition'''
        args.rotations_demo = True

    args.save_json = join(args.data_dir, 'args.json')
    return args


if __name__ == "__main__":
    # Parse arguments, make data dir, and put the arguments there (important!).
    args = parse_args()
    os.makedirs(args.data_dir)
    with open(args.save_json, 'w') as fh:
        json.dump(vars(args), fh, indent=4)

    # See the method documentations above for intended use cases of these.
    # Note: only one of the below methods should be un-commented at any time.
    #test_world_to_camera(args)
    #test_EE_tracking(args)

    # ------------ Safety / diagnostics each time we change physical setup ------------ #

    # CHECK 01: image cropping and the color/depth alignment.
    # test_image_crops() ; sys.exit()

    # CHECK 02: Sequentially move to test workspace bounds (obtained beforehand).
    #test_workspace_bounds() ; sys.exit()

    # ------------- First manipulation examples with the physical setup --------------- #

    #test_random_stuff(args) ; sys.exit() # just get a single pose.
    #test_action_params(args); sys.exit()  # test different action parameterizations

    # These go together, first to get poses, then to test going to them.
    #test_kinesthetic_teaching() ; sys.exit() # more scalable kinesthetic teaching
    # test_waypoints(args); sys.exit()  # test going to a sequence of poses.

    # Also kinesthetic teaching, but this is more 'continuous' where we just move
    # and continually get poses, images, etc., without pausing.
    # test_kinesthetic_continuous(args) ; sys.exit()

    # ----------------- 'Official' algorithmic policies ------------------ #

    # # use world distances, not working (debugging).
    # test_algorithmic_policy_world(args); sys.exit()

    # The main experiment we are now using.
    print("To confirm, ntarg, ndist: {}, {}, {}".format(args.n_targ, args.n_dist, args.policy))
    if args.dslr == True:
        raw_input("Check DSLR focus!\nPress Enter after confirmed...")
    else:
        raw_input("Press Enter to continue, CTRL+C to exit...")
    if args.policy == 'hum_t_demo' or args.policy == 'scripted_rotation':
        kinesthetic_demonstrator(args)
    elif args.policy == 'run_inference':
        print('here!')
        run_inference(args)
    elif args.policy == 'alg_t_simple':
        algorithmic_simple_demonstrator(args)
    elif args.policy == 'alg_pix':
        test_algorithmic_policy_pix(args)
    elif args.policy == 'alg_t_demo':
        algorithmic_translation_demonstrator(args)
    else:
        raise NotImplementedError()
