import pickle
import os, datetime
import rospy
import numpy as np
np.set_printoptions(suppress=True, linewidth=120)
import rospy
import intera_interface as ii
from sensor_msgs.msg import Image, PointCloud2
import time
import argparse
from cv_bridge import CvBridge
import cv2
import ros_numpy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from roll_utils import (
    create_goal_overlay,
    quaternion_rotation_matrix,
    roller_sim_forward,
    get_interaction_args,
    int2bool,
    generate_goal_locations
)
from video_recorder import VideoRecorder as VR
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)
from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest
)
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions,
    InteractionOptions,
    InteractionPublisher
)
from intera_core_msgs.msg import InteractionControlCommand
from intera_motion_msgs.msg import TrajectoryOptions
from std_msgs.msg import Header

# From Carl, not using.
#from dough.srv import *

# --------------------- Various constants. -------------------------- #
centre = np.array([0.567, -0.325, 0.0])
centre_sim = np.array([0.5, 0.06,  0.5])
ROBOT_INIT_POS = np.array([0.58, -0.05, 0.3265])
TOOL_REL_POS = np.array([0.015, 0, 0.215])
# FIXED_EE_Q = Quaternion(
#                     x=-0.641,
#                     y=0.641,
#                     z=0.299,
#                     w=0.299
#                     )
# FIXED_EE_Q = Quaternion(
#                     x=-0.664,
#                     y=0.664,
#                     z=0.242,
#                     w=0.242
#                     )
FIXED_EE_Q = Quaternion(
                    x=-0.707,
                    y=0.707,
                    z=0.,
                    w=0.
                    )
FIXED_EE_Q_SCIPY = R.from_quat([FIXED_EE_Q.x, FIXED_EE_Q.y, FIXED_EE_Q.z, FIXED_EE_Q.w])

EE_RESET_POSE = np.array([0.56, -0.05, 0.3, FIXED_EE_Q.w, FIXED_EE_Q.x, FIXED_EE_Q.y, FIXED_EE_Q.z])


def solver(pose):
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


def request_action(reqs):
    rospy.wait_for_service('policy_action')
    try:
        get_action_sim = rospy.ServiceProxy('policy_action', PolicyAct, persistent=True)
        resp = get_action_sim(reqs)

        return (
                np.array(resp.action),
                resp.done
        )
    except rospy.ServiceException as e:
        print(e)


def request_chamfer_distance(reqs):
    rospy.wait_for_service('calc_performance')
    try:
        get_cd = rospy.ServiceProxy('calc_performance', CalcPerformance, persistent=True)
        resp = get_cd(reqs)

        return (
                resp.performance
        )
    except rospy.ServiceException as e:
        print(e)


def get_cd(dough_points, goal_points):
    reqs = CalcPerformanceRequest()
    reqs.dough_x, reqs.dough_y, reqs.dough_z = dough_points[:, 0], dough_points[:, 1], dough_points[:, 2]
    reqs.goal_x, reqs.goal_y, reqs.goal_z = goal_points[:, 0], goal_points[:, 1], goal_points[:, 2]
    init_cd = request_chamfer_distance(reqs)
    return init_cd


def sample_init_tool_particles():
    n = 100
    n_sqrt = int(np.sqrt(n))
    r, h = 0.02 ,0.057
    linsp1 = np.linspace(-h, h, n_sqrt)
    linsp2 = np.linspace(0, np.pi * 2, n_sqrt)
    pos = np.empty((n, 3))
    i = 0
    for k in range(n_sqrt):
        for l in range(n_sqrt):
            pos[i] = np.array([r*np.cos(linsp2[k]), linsp1[l], r*np.sin(linsp2[k])])
            i+=1
    return pos


def sample_goal_particles(init_pos, radius=0.1):
    n_particles = 30000
    r = radius * np.sqrt(np.random.random([n_particles, 1]))
    theta = np.random.random([n_particles, 1]) * 2 * np.pi
    x, y = (np.cos(theta) * r).reshape(-1, 1), (np.sin(theta) * r).reshape(-1, 1)
    p = np.hstack([x, y, np.zeros_like(x)]) + init_pos
    return p


def get_dough_observation(camera_ns, subsample=True):
    # Reset robot arm, generate goal
    dough_xyz = camera_ns+'/filtered_dough_world_xyz'
    dough_xyz_data = rospy.wait_for_message(dough_xyz, PointCloud2)

    pc = ros_numpy.numpify(dough_xyz_data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']

    if subsample and len(points) > 1000:
        choices = np.random.choice(len(points), size=1000, replace=False)
        points = points[choices]
    return points


def capture_image(dir, camera_ns, filename, goal_points=None):
    color_data = rospy.wait_for_message(camera_ns+'/rgb/image_rect_color', Image)
    rgb_im = bridge.imgmsg_to_cv2(color_data).copy()[:,:,:3]
    img_path = os.path.join(dir, camera_ns+'_'+filename+'.jpg')
    if goal_points is not None:
        overlay = create_goal_overlay(buffer, goal_points, rgb_im.copy(), camera_ns)
        rgb_im = 0.3*rgb_im[:,:,:] + 0.7*overlay[:,:,:]
    cv2.imwrite(img_path, rgb_im)
    return img_path


def get_ee_pose(buffer):
    while not rospy.is_shutdown():
        try:
            ee_transform = buffer.lookup_transform('base', 'reference/right_connector_plate_base', rospy.Time(0))
            ee_pose = np.array([ee_transform.transform.translation.x,
            ee_transform.transform.translation.y,
            ee_transform.transform.translation.z,
            ee_transform.transform.rotation.w,
            ee_transform.transform.rotation.x,
            ee_transform.transform.rotation.y,
            ee_transform.transform.rotation.z])
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            continue
    ee_pose[2] -=0.004
    return ee_pose


def get_tool_pose(buffer):
    while not rospy.is_shutdown():
        try:
            ee_transform = buffer.lookup_transform('base', 'reference/right_connector_plate_base', rospy.Time(0))
            ee_pose = np.array([ee_transform.transform.translation.x,
            ee_transform.transform.translation.y,
            ee_transform.transform.translation.z,
            ee_transform.transform.rotation.w,
            ee_transform.transform.rotation.x,
            ee_transform.transform.rotation.y,
            ee_transform.transform.rotation.z])
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            continue
    ee_pose[2] -=0.004
    return ee_to_tool(ee_pose)


def get_tool_particles(tool_pose, init_tool_points):
    pos, Q = tool_pose[:3], tool_pose[3:]
    r_mat = quaternion_rotation_matrix(Q)
    trans_tool_points = init_tool_points.dot(r_mat.T)
    trans_tool_points += pos
    return trans_tool_points


def ee_to_tool(ee_pose):
    pos, Q = ee_pose[:3], ee_pose[3:]
    r_mat = quaternion_rotation_matrix(Q)
    TOOL_REL_POS_world = r_mat.dot(TOOL_REL_POS)

    tool_pos_world = pos + TOOL_REL_POS_world

    return np.concatenate([tool_pos_world, Q])


def tool_to_ee(tool_pose):
    pos, Q = tool_pose[:3], tool_pose[3:]
    r_mat = quaternion_rotation_matrix(Q)
    ee_rel_pos = -TOOL_REL_POS
    ee_rel_pos_world = r_mat.dot(ee_rel_pos)
    ee_pos_world = pos + ee_rel_pos_world
    return np.concatenate([ee_pos_world, Q])


def real_tool_to_sim(pose):
    ret_pose = np.zeros(7)

    curr_rot = R.from_quat([pose[4], pose[5], pose[6], pose[3]])
    angle = (curr_rot * FIXED_EE_Q_SCIPY.inv()).as_euler('zyx', degrees=True)[0]
    print("rotated angle:", angle)
    # import pdb; pdb.set_trace()
    rot = R.from_euler('zyx', [0, angle, 0], degrees=True)
    sim_init_rot = R.from_quat([0.707, 0, 0, 0.707])
    q = (rot * sim_init_rot).as_quat()
    ret_pose[3:] = [q[-1], q[0], q[1], q[2]]

    ret_pose[:3] = real_points_to_sim(pose[:3])

    return ret_pose


def sim_tool_to_real(pose):
    ret_pose = np.zeros(7)

    curr_rot = R.from_quat([pose[4], pose[5], pose[6], pose[3]])
    init_rot = R.from_quat([0.707, 0, 0, 0.707])
    rotated_angle = (curr_rot * init_rot.inv()).as_euler('zyx', degrees=True)[1]
    print("rotated angle:", rotated_angle)

    real_init_rot = FIXED_EE_Q_SCIPY
    rot = R.from_euler('zyx', [rotated_angle, 0, 0], degrees=True)
    q = (rot * real_init_rot).as_quat()
    ret_pose[3:] = [q[-1], q[0], q[1], q[2]]

    ret_pose[:3] = sim_points_to_real(pose[:3])
    return ret_pose


def real_points_to_sim(points):
    rotation = R.from_euler('ZYX', [np.pi / 2, 0, np.pi / 2]).inv()
    ret_points = points - centre
    ret_points = rotation.apply(ret_points)

    ret_points += centre_sim
    return ret_points


def sim_points_to_real(points):
    rotation = R.from_euler('ZYX', [np.pi / 2, 0, np.pi / 2])
    ret_points = points - centre_sim
    # ret_points = points
    ret_points = rotation.apply(ret_points)
    ret_points += centre
    return ret_points


def reset_robot(pose):
    print("Resetting robot to position: {}".format(pose[:3]))
    # execute_solver_resp(solver(pose=pose))
    execute_waypoint(solver(pose=pose))
    # motion_traj_plan(pose)

def turn_along_z(tool_pose, radian):
    new_tool_pose = np.zeros(7)
    new_tool_pose[:3] = tool_pose[:3]
    transform = R.from_rotvec(radian * np.array([0, 0, -1]))
    new_quat = (transform * R.from_quat([tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]])).as_quat()
    new_tool_pose[3:] = [new_quat[-1], new_quat[0], new_quat[1], new_quat[2]]
    print('policy action: ', new_tool_pose[:3] - tool_pose[:3])
    target_pose = tool_to_ee(new_tool_pose)
    resp = solver(target_pose)
    # execute_solver_resp(resp)
    execute_waypoint(resp)
    return target_pose


def execute_solver_resp(resp):
    print('IK Response: ', resp.joints[0].position)

    temp_positions = {}

    for joint in range(0, 7):
        temp_positions['right_j'+str(joint)] = resp.joints[0].position[joint]

    motion_executed = 0
    while not rospy.is_shutdown() and motion_executed != 1:
        if motion_executed!=1:
            '''If the motion has been executed then increment the value and
            exit the loop'''
            limb.move_to_joint_positions(temp_positions)
            motion_executed = 1


def execute_waypoint(resp):
    try:
        traj.clear_waypoints()
        joint_angles = limb.joint_ordered_angles()
        waypoint.set_joint_angles(joint_angles = joint_angles)
        traj.append_waypoint(waypoint.to_msg())
        temp_positions = {}
        joint_angles = []
        for joint in range(0, 7):
            joint_angles.append(resp.joints[0].position[joint])

        waypoint.set_joint_angles(joint_angles = joint_angles)
        traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory()
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


def generate_gif(filenames):
    gif_req = GenerateGifRequest(filenames)
    rospy.wait_for_service('generate_gif')

    try:
        generate = rospy.ServiceProxy('generate_gif', GenerateGif, persistent=True)
        generate(gif_req)
        print("generated rolling gif")
    except rospy.ServiceException as e:
        print(e)


def shutdown_func():
        print("Exit rolling")
        # ic_pub.send_position_mode_cmd()


def roll(args):
    # Logging
    if args.save_photo:
        os.makedirs(args.data_dir)
    dir = args.data_dir

    global limb, motion_executed, traj, waypoint, bridge, buffer

    # Rospy setups
    rospy.init_node('thing1', anonymous=True)
    rospy.Rate(100)
    rospy.on_shutdown(shutdown_func)
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)

    limb = ii.Limb('right')

    traj = MotionTrajectory(limb = limb)
    wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.1, max_joint_accel=0.1)
    waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)
    bridge = CvBridge()


    # Go back to initial position
    reset_robot(np.concatenate([ROBOT_INIT_POS, EE_RESET_POSE[3:]]))
    time.sleep(2)
    # i = 0
    # for r in [0.08, 0.1]:
    #     for _, goal in enumerate(generate_goal_locations()):
    #         points = sample_goal_particles(goal, r)
    #         capture_image(dir, 'k4a_top', 'goal_{}'.format(i), goal_points=points)
    #         capture_image(dir, 'k4a', 'goal_{}'.format(i), goal_points=points)
    #         i += 1
    # exit(0)

    init_dough_points = get_dough_observation('k4a', subsample=False)
    init_tool_points = sample_init_tool_particles()
    all_goal_points = sample_goal_particles(generate_goal_locations()[args.goal_id], args.radius)
    choices = np.random.choice(len(all_goal_points), size=1000, replace=False)
    goal_points = all_goal_points[choices]

    init_cd = get_cd(init_dough_points, all_goal_points)
    print("init chamfer dist:", init_cd)

    if args.save_photo:
        capture_image(dir, 'k4a_top', 'init_dough', goal_points=all_goal_points)
        capture_image(dir, 'k4a', 'init_dough', goal_points=all_goal_points)
        if args.save_video:
            vr = VR(dir, buffer, goal_points=all_goal_points)
            vr.start_recording('k4a/rgb/image_rect_color')

    if args.heuristic:
        roll_heuristic(args)
        T = 0
    else:
        T = 102
        idx_max = np.argmax(init_dough_points[:, 2])
        init_position = init_dough_points[idx_max]
        print("init position:", init_position)
        init_position[2] += 0.02
        EE_RESET_POSE[:3] = tool_to_ee(np.concatenate([init_position, EE_RESET_POSE[3:]]))[:3]
        reset_robot(EE_RESET_POSE)
        done = False
        # raw_input("Press Enter to continue...")

    obs, actions, poses = [], [], []
    for i in range(T):
        print(i)
        if True and i == 50:
            reset_robot(EE_RESET_POSE)
            time.sleep(0.1)
            continue
        elif True and i == 51:
            tool_pose = get_tool_pose(buffer)
            turn_along_z(tool_pose, np.pi/4)
            time.sleep(0.1)
            continue
        else:
            # get observation of the dough point
            if i == 0 or not args.open_loop:
                dough_points = get_dough_observation('k4a')
                print(dough_points.shape)

                # generate tool points
                tool_pose = get_tool_pose(buffer)
                tool_points = get_tool_particles(tool_pose, init_tool_points)
                # pass into the policy, get action
                sim_tool_pose = real_tool_to_sim(tool_pose)
                sim_tool_points, sim_dough_points, sim_goal_points = real_points_to_sim(tool_points), real_points_to_sim(dough_points), real_points_to_sim(goal_points)

                # rescaling
                scale = 2.0
                sim_tool_points -= centre_sim
                sim_dough_points -= centre_sim
                sim_goal_points -= centre_sim
                sim_tool_points, sim_dough_points, sim_goal_points = sim_tool_points * scale, sim_dough_points * scale, sim_goal_points * scale
                sim_tool_points += centre_sim
                sim_dough_points += centre_sim
                sim_goal_points += centre_sim
                # scene_points = np.concatenate([sim_dough_points, sim_tool_points, sim_goal_points], axis=0)
                # scene_min, scene_max = np.min(scene_points, axis=0), np.max(scene_points, axis=0)
                # sim_tool_points -= scene_min + (scene_max - scene_min) / 2
                # sim_dough_points -= scene_min + (scene_max - scene_min) / 2
                # sim_goal_points -= scene_min + (scene_max - scene_min) / 2
                # sim_tool_points, sim_dough_points, sim_goal_points = sim_tool_points * scale, sim_dough_points * scale, sim_goal_points * scale
                # sim_tool_points += scene_min + (scene_max - scene_min) / 2
                # sim_dough_points += scene_min + (scene_max - scene_min) / 2
                # sim_goal_points += scene_min + (scene_max - scene_min) / 2



                reqs = PolicyActRequest()
                reqs.dough_x, reqs.dough_y, reqs.dough_z = sim_dough_points[:, 0], sim_dough_points[:, 1], sim_dough_points[:, 2]
                reqs.tool_x, reqs.tool_y, reqs.tool_z = sim_tool_points[:, 0], sim_tool_points[:, 1], sim_tool_points[:, 2]
                reqs.goal_x, reqs.goal_y, reqs.goal_z = sim_goal_points[:, 0], sim_goal_points[:, 1], sim_goal_points[:, 2]
                reqs.tool_xyz = sim_tool_pose[:3]

                action, pol_done = request_action(reqs)
                if args.open_loop:
                    actions = np.array(action).reshape(-1 ,6)
                    action = actions[0]
                else:
                    actions.append(action)

            else:
                t = 0 if i < 50 else 2
                action = actions[i - t]
                # generate tool points
                tool_pose = get_tool_pose(buffer)
                tool_points = get_tool_particles(tool_pose, init_tool_points)
                # pass into the policy, get action
                sim_tool_pose = real_tool_to_sim(tool_pose)

            #adding
            poses.append(tool_pose)
            obs.append(np.concatenate([sim_dough_points, sim_tool_points, sim_goal_points], axis=0))

            # map action to robot ee position, use ik to solve
            new_sim_tool_pose = roller_sim_forward(sim_tool_pose, action)
            new_sim_tool_points = get_tool_particles(new_sim_tool_pose, init_tool_points)

            new_sim_tool_points -= centre_sim
            new_sim_tool_points *= scale
            new_sim_tool_points += centre_sim
            # new_sim_tool_points -= scene_min + (scene_max - scene_min) / 2
            # new_sim_tool_points *= scale
            # new_sim_tool_points += scene_min + (scene_max - scene_min) / 2
            # import pdb; pdb.set_trace()
            new_tool_pose = sim_tool_to_real(new_sim_tool_pose)
            new_tool_pose[:3] = (new_tool_pose[:3] - tool_pose[:3]) / scale  + tool_pose[:3]

            # import open3d as o3d
            # pcl = np.concatenate([sim_dough_points, sim_tool_points, sim_goal_points, new_sim_tool_points], axis=0)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
            # o3d.visualization.draw_geometries([pcd])

            # if args.frame == 'tool':
            #     tool_xyz = sim_tool_pose[:3].reshape(1, 3)
            #     sim_dough_points -= tool_xyz
            #     sim_tool_points -= tool_xyz
            #     sim_goal_points -= tool_xyz
            #     new_sim_tool_points -= tool_xyz
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # from mpl_toolkits.mplot3d import Axes3D

            # ax = fig.add_subplot(111, projection='3d')
            # # ax.scatter(dough_points[:, 0], dough_points[:, 1], dough_points[:, 2], alpha=0.5, color='r',label='dough')
            # # ax.scatter(tool_points[:, 0], tool_points[:, 1], tool_points[:, 2], alpha=0.5, color='yellow', label='tool')
            # # ax.scatter(goal_points[:, 0], goal_points[:, 1], goal_points[:, 2], alpha=0.5, label='goal')
            # ax.set_xlim(0.3, 0.8)
            # ax.set_ylim(0., 0.5)
            # ax.set_zlim(0.3, 0.8)
            # ax.scatter(sim_dough_points[:, 0], sim_dough_points[:, 1], sim_dough_points[:, 2], alpha=0.5, color='r',label='dough')
            # ax.scatter(sim_tool_points[:, 0], sim_tool_points[:, 1], sim_tool_points[:, 2], alpha=0.5, color='yellow', label='tool')
            # ax.scatter(sim_goal_points[:, 0], sim_goal_points[:, 1], sim_goal_points[:, 2], alpha=0.5, label='goal')
            # ax.scatter(new_sim_tool_points[:, 0], new_sim_tool_points[:, 1], new_sim_tool_points[:, 2], alpha=0.5, color='green', label='new_tool')
            # plt.legend()
            # plt.show()
            # exit(0)

            print('policy action: ', new_tool_pose[:3] - tool_pose[:3])
            target_pose = tool_to_ee(new_tool_pose)
            # break
            resp = solver(target_pose)
            # execute_solver_resp(resp)
            execute_waypoint(resp)

    reset_robot(EE_RESET_POSE)
    reset_robot(np.concatenate([ROBOT_INIT_POS, EE_RESET_POSE[3:]]))

    final_dough_points = get_dough_observation('k4a_top', subsample=False)
    if args.save_photo:
        capture_image(dir, 'k4a_top', 'final_dough', goal_points=all_goal_points)
        capture_image(dir, 'k4a', 'final_dough', goal_points=all_goal_points)
        if args.save_video:
            vr.stop_recording()
            vr.get_video()

    final_cd = get_cd(final_dough_points, all_goal_points)
    print("final chamfer dist:", final_cd)
    print("normalized performance:", (init_cd - final_cd) / init_cd)
    with open(os.path.join(dir, 'traj.pkl'), 'wb') as handle:
        pickle.dump({'normalized_performance':(init_cd - final_cd) / init_cd,
                    'goal_id': args.goal_id,
                    'radius': args.radius,
                    'obs':np.array(obs),
                    'actions':np.array(actions),
                    'tool_poses':np.array(poses),
                    'all_goal_points':all_goal_points,
                    'init_dough_points':init_dough_points,
                    'final_dough_points':final_dough_points}, handle)


def roll_heuristic(args):
    dir = args.data_dir
    n_roll = 2

    for r in range(n_roll):
        i = 0
        dough_points = get_dough_observation('k4a_top')
        idx_max = np.argmax(dough_points[:, 2])
        target_position = dough_points[idx_max]
        target_position[2] += 0.02

        target_position = tool_to_ee(np.concatenate([target_position, EE_RESET_POSE[3:]]))[:3]

        reset_robot(np.concatenate([target_position, EE_RESET_POSE[3:]]))
        i += 1

        radian = np.pi/4 * r
        tool_pose = get_tool_pose(buffer)
        tool_pose = turn_along_z(tool_pose, radian)
        dough_points = R.from_rotvec(radian * np.array([0, 0, -1])).apply(dough_points)
        i += 1

        roll_length = (np.max(dough_points[:, 1]) - np.min(dough_points[:, 1])) / 2
        roll_depth = (np.max(dough_points[:, 2]) - np.min(dough_points[:, 2]) ) / 2
        print(roll_depth, roll_length)

        waypts = np.array([[target_position[0], target_position[1], target_position[2] -z] for z in [roll_depth]])
        for wp in waypts:
            reset_robot(np.concatenate([wp, tool_pose[3:]]))
            i += 1

        target_position = waypts[-1]
        waypts = np.array([[target_position[0] -l*np.sin(radian), target_position[1] -l*np.cos(radian), target_position[2]] for l in [roll_length]])
        for wp in waypts:
            reset_robot(np.concatenate([wp, tool_pose[3:]]))
            i += 1

        target_position = waypts[-1]
        waypts = np.array([[target_position[0]+2*l*np.sin(radian), target_position[1]+2*l*np.cos(radian), target_position[2]] for l in [roll_length]])
        for wp in waypts:
            reset_robot(np.concatenate([wp, tool_pose[3:]]))
            i += 1

        reset_robot(np.concatenate([ROBOT_INIT_POS, EE_RESET_POSE[3:]]))
        i += 1
        # time.sleep(1)

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Rolling parameters')
    parser.add_argument('--save_photo', dest='save_photo', action='store_true')
    parser.add_argument('--no-save_photo', dest='save_photo', action='store_false')
    parser.set_defaults(save_photo=True)
    parser.add_argument('--save_video', dest='save_video', action='store_true')
    parser.add_argument('--no-save_video', dest='save_video', action='store_false')
    parser.set_defaults(save_video=False)
    parser.add_argument('--heuristic', dest='heuristic', action='store_true')
    parser.add_argument('--no-heuristic', dest='heuristic', action='store_false')
    parser.set_defaults(heuristic=False)
    parser.add_argument('--open_loop', dest='open_loop', action='store_true')
    parser.add_argument('--no-open_loop', dest='open_loop', action='store_false')
    parser.set_defaults(open_loop=False)
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'data/debug'))
    parser.add_argument('--frame', type=str, default='world')
    parser.add_argument('--goal_id', type=int, default=-1)
    parser.add_argument('--radius', type=float, default=0.1)
    args = parser.parse_args()
    if args.goal_id == -1:
        print("no valid goal specified !!!!!!!!!!")
        args.data_dir = os.path.join(args.data_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        args.data_dir = os.path.join(args.data_dir,  'goal_{}'.format(args.goal_id), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return args


def debug_robot_movement():
    """Debugging robot movement."""
    pass


if __name__ == "__main__":
    try:
        # From Carl
        args = parse_args()
        roll(args)

        # From me
        #test()
    except rospy.ROSInterruptException as e:
        print(e)