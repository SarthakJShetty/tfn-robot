"""Various utility files, for now a lot of it based on rotations and images."""
import os
from os.path import join
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from config import K_matrices

import rospy
import tf2_ros
import tf.transformations as tr
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3

# ---------------------------------------------------------------------------------------- #
# https://answers.ros.org/question/332407/transformstamped-to-transformation-matrix-python/
# ---------------------------------------------------------------------------------------- #

def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #

def uv_to_world_pos(buffer, u, v, z, camera_ns, debug_print=False):
    """Transform from image coordinates and depth to world coordinates.

    Parameters
    ----------
    u, v: image coordinates
    z: depth value
    camera_params:

    Returns
    -------
    world coordinates at pixels (u,v) and depth z.
    """

    # We name this as T_BC so that we go from camera to base.
    while not rospy.is_shutdown():
        try:
            if camera_ns == "k4a":
                T_BC = buffer.lookup_transform(
                        'base', 'rgb_camera_link', rospy.Time(0))
            elif camera_ns == "k4a_top":
                T_BC = buffer.lookup_transform(
                        'base', 'top_rgb_camera_link', rospy.Time(0))
            #print("Transformation, Camera -> Base:\n{}".format(T_BC))
            break
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            continue

    # Convert this to a 4x4 homogeneous matrix (borrowed code from ROS answers).
    matrix_camera_to_world = msg_to_se3(T_BC)

    # Get 4x4 camera intrinsics matrix.
    K = K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Will this work? Need to check. From SoftGym, and also they flip u,v here...
    one = np.ones(u.shape)
    x = (v - u0) * z / fx
    y = (u - v0) * z / fy
    # If x,y,z came from scalars, makes (1,4) matrix. Need to test for others.
    cam_coords = np.stack([x, y, z, one], axis=1)
    #world_coords = matrix_camera_to_world.dot(cam_coords.T)  # (4,4) x (4,1)
    #print(world_coords)
    # TODO(daniel) check filter_points, see if this is equivalent.
    world_coords = cam_coords.dot(matrix_camera_to_world.T)

    if debug_print:
        # Camera axis has +x pointing to me, +y to wall, +z downwards.
        #print('\nMatrix camera to world')
        #print(matrix_camera_to_world)
        #print('\n(inverse of that matrix)')
        #print(np.linalg.inv(matrix_camera_to_world))
        print('\n(cam_coords before converting to world)')
        print(cam_coords)
        print('')

    return world_coords  # (n,4) but ignore last row


def world_to_uv(buffer, world_coordinate, camera_ns, debug_print=False):
    """Transform from world coordinates to image pixels.

    From a combination of Carl Qi and SoftGym code.

    See also the `project_to_image` method and my SoftGym code:
    https://github.com/Xingyu-Lin/softagent_rpad/blob/master/VCD/camera_utils.py
    https://github.com/mooey5775/softgym_MM/blob/dev_daniel/softgym/utils/camera_projections.py

    TODO(daniel): should be faster to pre-compute this, right? The transformation
    should not change.

    Parameters
    ----------
    buffer: from ROS, so that we can look up camera transformations.
    world_coordinate: np.array, shape (n x 3), specifying world coordinates, i.e.,
        we might get from querying the tool EE Position.

    Returns
    -------
    (u,v): specifies (x,y) coords, `u` and `v` are each np.arrays, shape (n,).
        To use it directly with a numpy array such as img[uv], we might have to
        transpose it. Unfortunately I always get confused about the right way.
    """

    # We name this as T_CB so that we go from base to camera.
    while not rospy.is_shutdown():
        try:
            if camera_ns == "k4a":
                T_CB = buffer.lookup_transform(
                        'rgb_camera_link', 'base', rospy.Time(0))
            elif camera_ns == "k4a_top":
                T_CB = buffer.lookup_transform(
                        'top_rgb_camera_link', 'base', rospy.Time(0))
            #print("Transformation, Base -> Camera:\n{}".format(T_CB))
            break
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            continue

    # Convert this to a 4x4 homogeneous matrix (borrowed code from ROS answers).
    matrix_world_to_camera = msg_to_se3(T_CB)

    # NOTE(daniel) rest of this is from SoftGym, minus how we get the K matrix.
    world_coordinate = np.concatenate(
        [world_coordinate, np.ones((len(world_coordinate), 1))], axis=1)  # n x 4
    camera_coordinate = np.dot(matrix_world_to_camera, world_coordinate.T)  # 4 x n
    camera_coordinate = camera_coordinate.T  # n x 4  (ignore the last col of 1s)

    if debug_print:
        #print('\nMatrix world to camera')
        #print(matrix_world_to_camera)
        #print('\n(inverse of that matrix)')
        #print(np.linalg.inv(matrix_world_to_camera))
        print('\nWorld coords (source)')
        print(world_coordinate)
        print('\nCamera coords (target), but these need to be converted to pixels')
        print(camera_coordinate)
        print("")

    # Get 4x4 camera intrinsics matrix.
    K = K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Convert to ints because we want the pixel coordinates.
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    return (u,v)


def quaternion_rotation_matrix(Q):
    """Covert a quaternion into a full three-dimensional rotation matrix.

    (Borrowed from Carl's code.)

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


def print_debug(img, imgname):
    """Common method I use for debugging depth."""
    print('Eval {}: min/max/mean/medi: {:0.1f} {:0.1f} {:0.1f} {:0.1f}'.format(
        imgname, np.min(img), np.max(img), np.mean(img), np.median(img)
    ))


def triplicate(img, to_int=False):
    """Stand-alone `triplicate` method."""
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    for i in range(3):
        new_img[:,:,i] = img
    if to_int:
        new_img = new_img.astype(np.uint8)
    return new_img


def process_depth(orig_img, cutoff=2000):
    """Make a raw depth image human-readable by putting values in [0,255).

    Careful if the cutoff is in meters or millimeters!
    This might depend on the ROS topic. If using:
        rospy.Subscriber('k4a_top/depth_to_rgb/image_raw', ...)
    then it seems like it is in millimeters.
    """
    img = orig_img.copy()

    # Useful to turn the background into black into the depth images.
    def depth_to_3ch(img, cutoff):
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.flatten()
        img[img>cutoff] = 0.0
        img = img.reshape([w,h])
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def depth_scaled_to_255(img):
        if np.max(img) <= 0.0:
            print('Warning, np.max: {:0.3f}'.format(np.max(img)))
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img

    img = depth_to_3ch(img, cutoff)  # all values above 255 turned to white
    img = depth_scaled_to_255(img)   # correct scaling to be in [0,255) now
    return img


def make_video(data_dir, images_l, key):
    """Read images, write to video, in Python 2.7 code.

    Meant to be run at the end of a single robot trial.

    As a subprocedure, it saves the images to the target data directory, then loads
    them back. Note: remember that OpenCV uses BGR mode to represent images. Saving
    all the images is not strictly necessary but can be helpful for debugging and to
    check precise time alignment, which is why we do this.

    I was originally getting strange videos (screen was basically green) but this
    seems to be resolved with using the provided codec I use.

    Parameters
    ----------
    data_dir: place to store video and the intermediate images. Actually we'll use
        `all_imgs` as a sub-directory and just keep the videos in `data_dir`.
    images_l: the list of images we want to save.
    key: A string used to indicate the type (for us to understand file names). Make
        this unique if we are saving multiple videos in one directory! CAREFUL.
    """
    N = len(images_l)
    img_all_dir = join(data_dir, 'all_imgs')
    if not os.path.exists(img_all_dir):
        os.makedirs(img_all_dir)

    # Store the images.
    for idx in range(N):
        if np.isnan( np.sum(images_l[idx]) ):
            print('Warning, {} has NaN, skipping.'.format(idx))
            continue
        tail = 'img_{}_{}.png'.format(key, str(idx).zfill(4))
        fname = join(img_all_dir, tail)
        cv2.imwrite(fname, images_l[idx])
    print('Done saving {} images.'.format(N))

    # Get (width,height) which turns to (height,width) for frameSize argument.
    print('Img shape, dtype: {}, {}'.format(images_l[0].shape, images_l[0].dtype))

    if len(images_l[0].shape) == 2:
        # In this case we might have grayscale from segmentation.
        W, H = images_l[0].shape

        # Here we have to triplicate but this can also be done by loading. :)
        images_l = []  # override this with loaded triplicated grayscale images
        images_fnames = sorted(
            [img for img in os.listdir(img_all_dir)
                if img.endswith(".png") and key in img])
        for image in images_fnames:
            img = cv2.imread(join(img_all_dir, image))
            images_l.append(img)
    else:
        W, H, _ = images_l[0].shape

    # Make video, adding images from the provided `images_l`.
    video_name = join(data_dir, 'video_{}.avi'.format(key))
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(video_name, codec, fps=10, frameSize=(H,W))
    for img_t in images_l:
        video.write(img_t)
    cv2.destroyAllWindows()
    video.release()
    print('See video: {}, H,W are {},{}'.format(video_name, H, W))


def save_pcl(data_dir, pcl_l):
    """Save point clouds, then later we can try making a video from them.

    Meant to be run at the end of a single robot trial.

    For now we just save, I think making a video will be easier in Python3.

    Parameters
    ----------
    data_dir: place to store data.
    pcl_l: the list of point clouds we want to save.
    """
    N = len(pcl_l)
    pcl_all_dir = join(data_dir, 'all_pcls')
    if not os.path.exists(pcl_all_dir):
        os.makedirs(pcl_all_dir)

    # Store the individual point clouds.
    for idx in range(N):
        tail = 'pcl_{}.npy'.format(str(idx).zfill(4))
        fname = join(pcl_all_dir, tail)
        np.save(fname, pcl_l[idx])
    print('Done saving {} point clouds.'.format(N))
