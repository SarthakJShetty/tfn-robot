"""
Based on Ben/Harry's `mp_flow_pred.py` script and redone for MM.
"""
import numpy as np
np.set_printoptions(suppress=True, precision=5, linewidth=150)
from tokenize import group
import argparse
import os
from cv2 import norm
import rospy
import math
import tf
from sensor_msgs.msg import PointCloud2
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Point, PoseStamped
import sys
from scipy.spatial.transform import Rotation as R
from intera_core_msgs.msg import DigitalOutputCommand
import pickle
import tf2_ros
import tf2_py as tf2
import ros_numpy
import shutil
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import copy
import json
from motion_primitives import FlowbotController
# NOTE(daniel): can include these from Ben / Harry later if needed.
#from test_record_vid import VideoRecorder
#from vision_primitives import select_contact_point


def begin_suction(pub):
    cmd_1a = DigitalOutputCommand()
    cmd_1b = DigitalOutputCommand()
    cmd_1a.name = "right_valve_1a"
    cmd_1b.name = "right_valve_1b"
    cmd_1a.value = 0
    cmd_1b.value = 1
    pub.publish(cmd_1a)
    pub.publish(cmd_1b)
    cmd_1a.value = 1
    cmd_1b.value = 0
    pub.publish(cmd_1a)
    pub.publish(cmd_1b)


def end_suction(pub):
    cmd_1a = DigitalOutputCommand()
    cmd_1b = DigitalOutputCommand()
    cmd_1a.name = "right_valve_1a"
    cmd_1b.name = "right_valve_1b"
    cmd_1a.value = 1
    cmd_1b.value = 0
    pub.publish(cmd_1a)
    pub.publish(cmd_1b)
    cmd_1a.value = 0
    cmd_1b.value = 1
    pub.publish(cmd_1a)
    pub.publish(cmd_1b)


def set_goal_with_flow_phase_1(flow_pt, tip_rot, flow_vec, tip_hand_axis):
    v1 = tip_hand_axis / np.linalg.norm(tip_hand_axis)
    v2 = -flow_vec / np.linalg.norm(flow_vec)
    quat = calc_rot(v1, v2)
    hand_rot = R.from_quat(tip_rot).as_dcm()
    flow_pt_rot = R.from_dcm(np.matmul(R.from_quat(quat).as_dcm(), hand_rot)).as_quat()

    T_world_tip = np.vstack([np.hstack([R.from_quat(flow_pt_rot).as_dcm(), np.array(flow_pt).reshape(3, 1)]), np.array([0, 0, 0, 1]).reshape(1, 4)])
    T_tip_hand = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.13], [0, 0, 0, 1]])
    T_goal = np.matmul(T_world_tip, T_tip_hand)
    flow_pt_trans = T_goal[:3, -1]

    fixed_yaw = R.from_quat(flow_pt_rot).as_euler('xyz')[2]
    return flow_pt_trans, flow_pt_rot, fixed_yaw


def set_goal_with_flow_phase_2(tip_trans, tip_rot, flow_vec, tip_hand_axis, fixed_yaw, scale=1):
    goal_pt = np.array(tip_trans) + scale*0.02*flow_vec / np.linalg.norm(flow_vec)
    v1 = tip_hand_axis / np.linalg.norm(tip_hand_axis)
    v2 = -flow_vec / np.linalg.norm(flow_vec)
    quat = calc_rot(v1, v2)
    hand_rot = R.from_quat(tip_rot).as_dcm()
    flow_pt_rot = R.from_dcm(np.matmul(R.from_quat(quat).as_dcm(), hand_rot)).as_quat()

    T_world_tip = np.vstack([np.hstack([R.from_quat(tip_rot).as_dcm(), np.array(goal_pt).reshape(3, 1)]), np.array([0, 0, 0, 1]).reshape(1, 4)])
    T_tip_hand = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.13], [0, 0, 0, 1]])
    T_goal = np.matmul(T_world_tip, T_tip_hand)
    flow_pt_trans = T_goal[:3, -1]

    return flow_pt_trans, flow_pt_rot


def calc_rot(v1, v2):
    angle = np.arccos(np.dot(v1, v2))
    axis = np.cross(v1, v2)
    axis = axis/np.linalg.norm(axis)
    qx = axis[0] * np.sin(angle/2)
    qy = axis[1] * np.sin(angle/2)
    qz = axis[2] * np.sin(angle/2)
    qw = np.cos(angle/2)
    quat = np.array([qx, qy, qz, qw])
    quat = quat / np.linalg.norm(quat)

    return quat


def get_ee_pose(listener):
    while not rospy.is_shutdown():
        try:
            # NOTE(daniel) vacuum gripper seems to be hanging for me
            # (tip_trans, tip_rot) = listener.lookupTransform('/base', '/right_vacuum_gripper_tip_hack', rospy.Time(0))
            (tip_trans, tip_rot) = listener.lookupTransform('/base', '/right_gripper', rospy.Time(0))
            tip_trans = [tip_trans[0], tip_trans[1], tip_trans[2]]
            #print("EE translation: ", tip_trans)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    return tip_trans, tip_rot


def get_hand_pose(listener):
    while not rospy.is_shutdown():
        try:
            (hand_trans, hand_rot) = listener.lookupTransform('/base', '/right_hand', rospy.Time(0))
            hand_trans = [hand_trans[0], hand_trans[1], hand_trans[2]]
            #print("Hand base translation: ", hand_trans)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    return hand_trans, hand_rot


def evaluate_plan(plan):
    # Evaluate the plan by calculating the final pose distance to the initial pose.
    # We want to minimize this distance to do avoid crazy motions
    start_config = np.array(plan.joint_trajectory.points[0].positions)
    planned_end_config = np.array(plan.joint_trajectory.points[-1].positions)
    plan_joint_dist = np.linalg.norm(planned_end_config - start_config)
    return plan_joint_dist


def max_flow_selection():
    # This is used for selecting the max flow point for grasping

    # Due to discrepency between py2 and py3, the flow predication is done in a different
    # Python3 env. The results will be saved into a file and read here.

    pred_result = pickle.load(open('flow_pred_res/flow_pred_result.pkl', 'rb'))
    max_flow_pt = pred_result['max_flow_pt']
    max_flow_vector = pred_result['max_flow_vec']
    return max_flow_pt, max_flow_vector


def max_flow_selection_with_heuristic(T_gripper):
    # This is used for selecting the max flow point for grasping

    # Due to discrepency between py2 and py3, the flow predication is done in a different
    # Python3 env. The results will be saved into a file and read here.

    preds = pickle.load(open('flow_pred_res/all_pred_flows.pkl', 'rb'))
    pcd = preds['pcd']
    raw_pcd = preds['raw_pcd']
    valid_pick_points = preds["valid_pts"]
    flows = preds['flows']

    max_flow_pt, max_flow_vector = select_contact_point(pcd, flows, raw_pcd, valid_pick_points, 10)
    return max_flow_pt, max_flow_vector


def pcd_callback(data, prev_pcd=None):
    # CB function for pointcloud recording
    new_data = do_transform_cloud(data, world_transform)

    pc = ros_numpy.numpify(new_data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']

    nonnan_pts = points[~np.isnan(points).any(axis=1)]
    # nonnan_pts = nonnan_pts[np.logical_and(nonnan_pts[:, 0] > -1.0, nonnan_pts[:, 0] < -0.5)]
    # nonnan_pts = nonnan_pts[np.logical_and(nonnan_pts[:, 1] > -.6, nonnan_pts[:, 1] < 0.6)]
    # cleaned_pcd = nonnan_pts[np.logical_and(nonnan_pts[:, 2] > -0.01, nonnan_pts[:, 2] < 0.8)]
    nonnan_pts = nonnan_pts[np.logical_and(nonnan_pts[:, 0] > -.5, nonnan_pts[:, 0] < 0.6)]
    nonnan_pts = nonnan_pts[np.logical_and(nonnan_pts[:, 1] > -1.0, nonnan_pts[:, 1] < -0.4)]
    cleaned_pcd = nonnan_pts[np.logical_and(nonnan_pts[:, 2] > -0.02, nonnan_pts[:, 2] < 0.7)]
    # cleaned_pcd = clean_pointcloud(nonnan_pts)

    if prev_pcd is not None:

        # Next, if there is a previous pointcloud, filter by its bounding box plus a bit.
        mins, maxs = prev_pcd.min(axis=0), prev_pcd.max(axis=0)
        lowers, uppers = mins - 0.1, maxs + 0.1

        cleaned_pcd = cleaned_pcd[np.logical_and(cleaned_pcd[:, 0] > lowers[0], cleaned_pcd[:, 0] < uppers[0])]
        cleaned_pcd = cleaned_pcd[np.logical_and(cleaned_pcd[:, 1] > lowers[1], cleaned_pcd[:, 1] < uppers[1])]
        cleaned_pcd = cleaned_pcd[np.logical_and(cleaned_pcd[:, 2] > lowers[2], cleaned_pcd[:, 2] < uppers[2])]

    np.save("flow_pred/filtered_pts.npy", cleaned_pcd)

    print("Filtered a pointcloud with {} points".format(len(cleaned_pcd)))

    cleaned_pcd_arr = np.zeros((len(cleaned_pcd),), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
    ])
    cleaned_pcd_arr['x'] = cleaned_pcd[:, 0]
    cleaned_pcd_arr['y'] = cleaned_pcd[:, 1]
    cleaned_pcd_arr['z'] = cleaned_pcd[:, 2]

    msg = ros_numpy.msgify(PointCloud2, cleaned_pcd_arr)
    msg.header = new_data.header
    global publisher
    publisher.publish(msg)

    return cleaned_pcd

publisher = None
world_transform =  None
prev_pcd = None


def pcd_collection(publishVacuumCommand):
    # usr_input = raw_input("Start pcd collection? ( y for yes )")
    # if usr_input.lower() != "y":
    #     exit(0)

    print('Collecting pointcloud from K4A.....')
    data = rospy.wait_for_message("/k4a/depth_filtered/points_filtered", PointCloud2)
    global publisher, world_transform
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    while not rospy.is_shutdown():
        try:
            world_transform = buffer.lookup_transform('base', 'depth_camera_link', rospy.Time(0))
            print("World transformation: ", world_transform)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    publisher = rospy.Publisher('points2_nonnan_filtered', PointCloud2, queue_size=10)
    global prev_pcd
    np_pc = pcd_callback(data, prev_pcd=prev_pcd)
    prev_pcd = np_pc
    print('Finishing up pcd collection.....')
    rospy.sleep(2)
    print('Run Flow Prediction Neural Model now....')
    usr_input = raw_input("Finished running Flow Prediction Neural Model? ( y for yes )")
    while usr_input.lower() != "y" and usr_input.lower() != "n":
        usr_input = raw_input("Please enter a valid option. (y/n)").lower()

    if usr_input == 'n':
        end_suction(publishVacuumCommand)
        return False
    else:
        return True


def record_trial_meta_data(result_dir):
    valid = raw_input("Valid trial (1) / Invalid trial (0)? ")
    while valid != '1' and valid != '0':
        valid = raw_input('Please input a valid result 1/0... ')

    if valid == "0":
        reason = raw_input("Reason for invalid run: ")
        success = "N/A"
        good_contact = "N/A"
        dist = "N/A"
    else:
        reason = "N/A"
        success = raw_input("Success (1) / Failure (0)? ")
        while success != '1' and success != '0':
            success = raw_input('Please input a valid result 1/0... ')
        good_contact = raw_input("Contact Point Success (1) / Failure (0)? ")
        while good_contact != '1' and good_contact != '0':
            good_contact = raw_input('Please input a valid result 1/0...')
        dist = raw_input("Measured distance (cm for prismatic, deg for revolute): ")
    result = {'valid': valid, 'succ': success, 'gc': good_contact, 'dist': dist, "reason": reason,}
    with open(os.path.join(result_dir, 'metdadata.json'), 'w') as fp:
        json.dump(result, fp)
    if valid == '0':
        os.rename(result_dir, result_dir+"_invalid")


def motion_plan(result_dir, shoot_video):
    # MOVEIT COMMANDER MUST HAPPEN BEFORE THE ROSNODE
    joint_state_topic = ["joint_states:=/robot/joint_states"]
    moveit_commander.roscpp_initialize(joint_state_topic)
    rospy.init_node('mp_pred_flow')

    listener = tf.TransformListener()

    robot = FlowbotController()

     # We can get the name of the reference frame for this robot:
    planning_frame = robot.group.get_planning_frame()
    print("============ Reference frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = robot.group.get_end_effector_link()
    print("============ End effector: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.robot.get_group_names()
    print("============ Robot Groups:", group_names)

    # Sometimes for debugging it is useful to print the entire state of the robot:
    print("============ Printing robot state")
    print(robot.robot.get_current_state())
    print("============ Printing robot joint angles")
    print(np.array(robot.group.get_current_joint_values()))
    print("============ Printing home joint angles")
    print(np.array(robot.home_jas))
    print("============ Printing robot EE pose")
    tip_trans, tip_rot = get_ee_pose(listener)
    hand_trans, hand_rot = get_hand_pose(listener)
    print("tip trans:  ", np.array(tip_trans))
    print("tip rot:    ", np.array(tip_rot))
    print("hand trans: ", np.array(hand_trans))
    print("hand rot:   ", np.array(hand_rot))

    # FIRST, RESET THE ROBOT:
    robot.go_to_experiment_home()

    ## Create the camera. Fail fast if we can't claim.
    #if shoot_video:
    #    recorder = VideoRecorder(result_dir)

    """
    PHASE I:
    Here we get the desired grasping point from the point cloud by choosing the point with the maximum flo vector magnitude

    Step 1: Take a point cloud observation
    Step 2: Pass it into the flow prediction module
    """
    if not pcd_collection(robot.publishVacuumCommand):
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("we don't want to continue")
        return

    # Set goal for grasping phase
    # grasp_pt_goal, flow = max_flow_selection()

    tip_trans, tip_rot = get_ee_pose(listener)
    hand_trans, hand_rot = get_hand_pose(listener)
    tip_hand_axis = np.array(tip_trans) - np.array(hand_trans)

    grasp_pt_goal, flow = max_flow_selection_with_heuristic(tip_trans)
    print(grasp_pt_goal)
    print(flow)

    grasp_goal_trans, grasp_goal_rot, fixed_yaw = set_goal_with_flow_phase_1(grasp_pt_goal, hand_rot, flow, tip_hand_axis)

    normflow = flow / np.linalg.norm(flow)

    if shoot_video:
        recorder.start_movie()

    robot.moveit_to_pose(grasp_goal_trans + normflow * 0.1, grasp_goal_rot)

    # Save the step's flow prediction to working directory...
    shutil.copyfile("/home/beisner/catkin_ws/src/flowbot/flow_pred_res/all_pred_flows.pkl", os.path.join(result_dir, "p1_pred.pickle"))
    print("Phase 1 Predicted Flow saved...")

    usr_input = raw_input("Go to phase 2? ( y for yes )")
    while usr_input.lower() != "y" and usr_input.lower() != "n":
        usr_input = raw_input("Please enter a valid option. (y/n)").lower()

    if usr_input == 'n':

        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("we don't want to continue")

        if shoot_video:
            recorder.stop_movie()
            recorder.exit()

        record_trial_meta_data(result_dir)
        return

    robot.move_until_contact(-normflow, 0.03, 10.0)

    robot.begin_suction()
    tip_trans, tip_rot = get_ee_pose(listener)
    hand_trans, hand_rot = get_hand_pose(listener)
    tip_hand_axis = np.array(tip_trans) - np.array(hand_trans)
    grasp_goal_trans, grasp_goal_rot, fixed_yaw = set_goal_with_flow_phase_1(grasp_pt_goal+0.01*flow/np.linalg.norm(flow), hand_rot, flow, tip_hand_axis)

    rospy.sleep(3)

    robot.move_with_compliance(flow, 0.03, 0.01, 2.0)

    """
    PHASE II
    """
    done = False
    step = 0
    while not done:

        # Obtain Current EE pose
        tip_trans, tip_rot = get_ee_pose(listener)
        hand_trans, hand_rot = get_hand_pose(listener)
        tip_hand_axis = np.array(tip_trans) - np.array(hand_trans)

        pose_target = geometry_msgs.msg.Pose()
        min_plan, min_score = None, float('inf')
        print("Step: {}".format(step))

        # Run flow prediction and record point cloud
        if not pcd_collection(robot.publishVacuumCommand):
            robot.end_suction()
            if shoot_video:
                recorder.stop_movie()
                recorder.exit()
            moveit_commander.roscpp_shutdown()
            print('Failure Detected! Exiting prematurely...')
            rospy.signal_shutdown("we don't want to continue")

            record_trial_meta_data(result_dir)


            return
        _, flow = max_flow_selection()
        shutil.copyfile("/home/beisner/catkin_ws/src/flowbot/flow_pred_res/all_pred_flows.pkl", os.path.join(result_dir, "p2_pred_%03d.pickle"%step))
        print("Phase 2 Predicted Flow saved...")
        if True:
            # robot.move_with_compliance(flow, 0.03, 0.1, 2.0)
            robot.move_with_flow(flow, tip_hand_axis, tip_rot, 0.03, 0.1, 2.0)
            usr_input = raw_input("Done? ( y for yes)")
            while usr_input.lower() != "y" and usr_input.lower() != "n":
                usr_input = raw_input("Please enter a valid option. (y/n)").lower()
            if usr_input == "y":
                done = True
            else:
                done = False
                step += 1

    robot.end_suction()

    if shoot_video:
        recorder.stop_movie()
        recorder.exit()
    moveit_commander.roscpp_shutdown()
    print(('The current trial has run to the end. Exiting gracefully...'))
    rospy.signal_shutdown('The current trial has run to the end. Exiting gracefully...')
    record_trial_meta_data(result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str)
    args = parser.parse_args()

    # NOTE(daniel): from their code, ignoring for now.
    #obj_name = args.obj
    #if obj_name is None:
    #    print("Usage: python mp_flow_pred.py --obj <object>")
    #    exit(0)

    #trialname = "umpmetric_results_2022_01_26"
    #if not os.path.exists(os.path.join(os.getcwd(), trialname, obj_name)):
    #    os.makedirs(os.path.join(os.getcwd(), trialname, obj_name))
    #trialnames = os.listdir(os.path.join(os.getcwd(), trialname, obj_name))
    #if len(trialnames) == 0:
    #    ind = 0
    #else:
    #    base = max([int(name[:3]) for name in trialnames])
    #    ind = int(base) + 1
    #trial_num = "%03d" % ind
    #result_dir = os.path.join(os.getcwd(), trialname, obj_name, trial_num)
    result_dir = 'test'
    if not os.path.exists(result_dir):
        print("Creating result directory...")
        os.makedirs(result_dir)
        print("Created result directory")

    #shoot_video = raw_input("Record a video of this trial? (Y/n)") != "n"

    motion_plan(result_dir, shoot_video=False)
