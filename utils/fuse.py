"""
Post-processing script to fuse out of sync depth and image files. Not utilized anymore since we
fixed most of the runtime sync issues
"""

from config import K_matrices
# from filter_points import CROP_X, CROP_H, CROP_W, CROP_Y
import cv2
import numpy as np
from os.path import basename, dirname, join, split
from glob import glob
import argparse
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.distance import cdist

ns = 'k4a_top'

CROP_X = 840
CROP_Y = 450
CROP_W = 300
CROP_H = 300

K_matrices = {
    'k4a': np.array([
        [977.870,     0.0, 1022.401],
        [    0.0, 977.865,  780.697],
        [    0.0,     0.0,      1.0]
    ]),
    'k4a_top': np.array([
        [977.005,     0.0, 1020.287],
        [    0.0, 976.642,  782.864],
        [    0.0,     0.0,      1.0]
    ]),
}

camera_transform = TransformStamped()
camera_transform.transform.translation.x = 0.667
camera_transform.transform.translation.y = -0.364
camera_transform.transform.translation.z = 0.668

camera_transform.transform.rotation.x = 0.999982007787
camera_transform.transform.rotation.y = 0.00255910116634
camera_transform.transform.rotation.z = 0.00131301532838
camera_transform.transform.rotation.w = 0.00526281172579

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

def rgb_depth_merge(bgr_im, depth_im):
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

    # print(len(pcl))

    # choices = np.random.choice(len(pcl), size=5000, replace=False)

    # pcl = pcl[choices]

    return pcl

def time_sync(pre_sync_img_files, pre_sync_dpt_files):
    '''This is the function that we will use to time-sync the depth to the RGB images'''
    pre_sync_img_files.sort(key = lambda x: (int(basename(x).split('_')[3]), int(basename(x).split('_')[4][:-4])))
    pre_sync_dpt_files.sort(key = lambda x: (int(basename(x).split('_')[3]), int(basename(x).split('_')[4][:-4])))

    imgs_time = np.zeros((len(pre_sync_img_files), 1))
    dpts_time = np.zeros((len(pre_sync_img_files), 1))

    for t in range(len(imgs_time)):
        imgs_time[t, 0] = int(basename(pre_sync_img_files[t]).split('_')[3]) + 1e-9 * int(basename(pre_sync_img_files[t]).split('_')[4][:-4])
        dpts_time[t, 0] = int(basename(pre_sync_dpt_files[t]).split('_')[3]) + 1e-9 * int(basename(pre_sync_dpt_files[t]).split('_')[4][:-4])

    dist_idx = np.argmin(cdist(imgs_time, dpts_time, 'euclidean'), axis = 1)

    pre_sync_dpt_files_np = np.array(pre_sync_dpt_files)


    synced_dpt_files = pre_sync_dpt_files_np[dist_idx]

    print('Depth image indices, corresponding to the RGB images: \n', dist_idx)

    return pre_sync_img_files, synced_dpt_files

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir')
    p.add_argument('--time_sync', type=bool, default=True)
    args = p.parse_args()

    '''Calling this pre_sync_* because we haven't sorted them according to time-stamp or demo, iteration'''
    pre_sync_img_files = glob(join(args.dir, 'img_*'))
    pre_sync_dpt_files = glob(join(args.dir, 'dpt_*'))

    if args.time_sync == False:
        '''If there's no time-sync then just sort them according to num_demo_iteration, otherwise
        send them to time_sync'''
        img_files = pre_sync_img_files.sort(key = lambda x: (int(basename(x).split('.')[0][4:].split('_')[0]), int(basename(x).split('.')[0][4:].split('_')[1])))
        dpt_files = pre_sync_dpt_files.sort(key = lambda x: (int(basename(x).split('.')[0][4:].split('_')[0]), int(basename(x).split('.')[0][4:].split('_')[1])))
    else:
        img_files, dpt_files = time_sync(pre_sync_img_files, pre_sync_dpt_files)

    assert len(img_files) == len(dpt_files), 'Mismatch in size of img {} and depth image list {}'.format(len(img_files), len(dpt_files))

    for idx in range(len(img_files)):

        demo = basename(img_files[idx]).split('.')[0][4:].split('_')[0]
        iteration = basename(img_files[idx]).split('.')[0][4:].split('_')[1]

        color_img = cv2.imread(img_files[idx])
        depth_img = cv2.imread(dpt_files[idx], cv2.COLOR_BGR2GRAY)

        print('Generating PCL from image: {} depth: {}'.format(basename(img_files[idx]), basename(dpt_files[idx])))

        pcl = rgb_depth_merge(color_img, depth_img)
        pcl_path = join(dirname(img_files[idx]), 'pcn_{}_{}.npy'.format(demo, iteration))
        np.save(pcl_path, pcl.copy())
