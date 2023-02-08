"""
Put files here for configurations ettings, etc.
Meant to be used for both Python2 and Python3 as needed.
"""
import numpy as np

# Directory to where point cloud data is saved.
PC_HEAD = '/home/sarthak/catkin_ws/src/mixed-media-physical/pcl'

# For cropping images.
CROP_X = 840
CROP_Y = 450
CROP_W = 300
CROP_H = 300

# Get from `rostopic echo /k4a_top/rgb/camera_info`. These are the _intrinsics_,
# which are just given to us (we don't need calibration for these).
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
