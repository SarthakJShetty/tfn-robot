from turtle import position
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from copy import deepcopy
import tf2_ros
import rospy


def generate_goal_locations():
    init_pos = np.array([0.567, -0.325, 0.0])
    xs = np.linspace(-0.05, 0.05, 4)
    ys = np.linspace(-0.05, 0.05, 4)
    positions = []
    for x in xs:
        for y in ys:
            positions.append(init_pos + np.array([x, y, 0.0]))
    return positions


def create_goal_overlay(buffer, goal_points, rgb_im, camera_ns):
    Ks = {
    'k4a': np.array([[977.870910644531, 0.0, 1022.4010620117188],
                  [0.0, 977.8651123046875, 780.697998046875],
                  [0.0, 0.0, 1.0]]),
    'k4a_top': np.array([[977.0050659179688, 0.0, 1020.2879638671875],
                  [0.0, 976.642578125, 782.8642578125],
                  [0.0, 0.0, 1.0]])
        }
    while not rospy.is_shutdown():
        try:
            if camera_ns == "k4a":
                camera_transform = buffer.lookup_transform('base', 'rgb_camera_link', rospy.Time(0))
            else:
                camera_transform = buffer.lookup_transform('base', 'top_rgb_camera_link', rospy.Time(0))
            # print("Camera to base transformation: ", camera_transform)
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            continue
    camera_pos = np.array([camera_transform.transform.translation.x,
                camera_transform.transform.translation.y,
                camera_transform.transform.translation.z])

    camera_rot = R.from_quat([camera_transform.transform.rotation.x,
                camera_transform.transform.rotation.y,
                camera_transform.transform.rotation.z,
                camera_transform.transform.rotation.w])
    camera_points = camera_rot.inv().apply(goal_points - camera_pos)
    camera_coordinate = np.concatenate([camera_points, np.ones((len(camera_points), 1))], axis=1)

    K = Ks[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    for i in range(len(u)):
        rgb_im[v[i], u[i]] = [0, 0, 255]
    return rgb_im


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def roller_sim_forward(roller_pose, action):
    substep = 19
    action_scale = np.array([0.7, 0.05, 0.005, 0.005, 0., 0.])
    action = action * action_scale / substep

    final_pose = np.copy(roller_pose)
    for s in range(substep):
        dw = action[0]  # rotate about object y
        dth = action[1]  # rotate about the world w
        dy = action[2]  # decrease in y coord...
        w = action[3:6]

        roller_rot = R.from_quat([final_pose[4], final_pose[5], final_pose[6], final_pose[3]])

        y_dir = roller_rot.apply(np.array([0., -1., 0.]))

        x_dir = np.cross(np.array([0., 1., 0.]), y_dir) * (dw * 0.03 + w[0]) #0.03 radius, so dw*0.03 is roughly rolling length
        x_dir[1] = dy  # direction

        rot1 = R.from_rotvec([0., -dth, 0.])
        rot2 = R.from_rotvec([0., dw, 0.])
        new_roller_rot_quat = (rot1 * roller_rot * rot2).as_quat()
        final_pose[3:] = np.array([new_roller_rot_quat[3], new_roller_rot_quat[0], new_roller_rot_quat[1], new_roller_rot_quat[2]])

        # TODO: add safety box here?
        final_pose[:3] = final_pose[:3] + x_dir
        final_pose[1] = max(0.03, final_pose[1])

    return final_pose


def int2bool(var):
    """
    Convert integer value/list to bool value/list
    """
    var_out = deepcopy(var)

    if isinstance(var, int):
        var_out = bool(var)
    elif len(var) >= 1:
        for i in range(0, len(var)):
            var_out[i] = bool(var[i])

    return var_out


def get_interaction_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",  "--interaction_active", type=int, default=1, choices = [0, 1],
        help="Activate (1) or Deactivate (0) interaction controller")
    parser.add_argument(
        "-k", "--K_impedance", type=float,
        nargs='+', default=[1300.0, 1300.0, 1300.0, 30.0, 30.0, 30.0],
        help="A list of desired stiffnesses, one for each of the 6 directions -- stiffness units are (N/m) for first 3 and (Nm/rad) for second 3 values")
    parser.add_argument(
        "-m", "--max_impedance", type=int,
        nargs='+', default=[1, 1, 1, 1, 1, 1], choices = [0, 1],
        help="A list of maximum stiffness behavior state, one for each of the 6 directions (a single value can be provided to apply the same value to all the directions) -- 0 for False, 1 for True")
    parser.add_argument(
        "-md", "--interaction_control_mode", type=int,
        nargs='+', default=[1, 1, 1, 1, 1, 1], choices = [1,2,3,4],
        help="A list of desired interaction control mode (1: impedance, 2: force, 3: impedance with force limit, 4: force with motion limit), one for each of the 6 directions")
    parser.add_argument(
        "-fr", "--interaction_frame", type=float,
        nargs='+', default=[0, 0, 0, 1, 0, 0, 0],
        help="Specify the reference frame for the interaction controller -- first 3 values are positions [m] and last 4 values are orientation in quaternion (w, x, y, z) which has to be normalized values")
    parser.add_argument(
        "-ef",  "--in_endpoint_frame", action='store_true', default=False,
        help="Set the desired reference frame to endpoint frame; otherwise, it is base frame by default")
    parser.add_argument(
        "-en",  "--endpoint_name", type=str, default='right_hand',
        help="Set the desired endpoint frame by its name; otherwise, it is right_hand frame by default")
    parser.add_argument(
        "-f", "--force_command", type=float,
        nargs='+', default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="A list of desired force commands, one for each of the 6 directions -- in force control mode this is the vector of desired forces/torques to be regulated in (N) and (Nm), in impedance with force limit mode this vector specifies the magnitude of forces/torques (N and Nm) that the command will not exceed")
    parser.add_argument(
        "-kn", "--K_nullspace", type=float,
        nargs='+', default=[5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
        help="A list of desired nullspace stiffnesses, one for each of the 7 joints (a single value can be provided to apply the same value to all the directions) -- units are in (Nm/rad)")
    parser.add_argument(
        "-dd",  "--disable_damping_in_force_control", action='store_true', default=False,
        help="Disable damping in force control")
    parser.add_argument(
        "-dr",  "--disable_reference_resetting", action='store_true', default=False,
        help="The reference signal is reset to actual position to avoid jerks/jumps when interaction parameters are changed. This option allows the user to disable this feature.")
    parser.add_argument(
        "-rc",  "--rotations_for_constrained_zeroG", action='store_true', default=False,
        help="Allow arbitrary rotational displacements from the current orientation for constrained zero-G (works only with a stationary reference orientation)")
    parser.add_argument(
        "-r",  "--rate", type=int, default=10,
        help="A desired publish rate for updating interaction control commands (10Hz by default) -- a rate 0 publish once and exits which can cause the arm to remain in interaction control.")

    args = parser.parse_args()
    return args