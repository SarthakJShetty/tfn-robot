"""Mean to be run in Python3. I'm using open3d==0.6.0 and Python 3.6.

Some basic documentation:
http://www.open3d.org/docs/0.6.0/tutorial/Basic/pointcloud.html
"""
from os.path import join
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import PC_HEAD

# See `filter_points.py` for what gets saved.
PC_HEAD = '/home/sarthak/catkin_ws/src/mixed-media-physical/pcl'
PC_FILE = 'pts_combo_data.npy'

# Load from saved numpy for inspection.
pcl_path = join(PC_HEAD, PC_FILE)
pcl = np.load(pcl_path)
print('Loaded PC at {}, shape: {}'.format(pcl_path, pcl.shape))
print('Mean per axis: {}'.format(np.mean(pcl, axis=0)))

# Try to add color to the point clouds.
pcl_color = np.zeros((pcl.shape[0], 3))
pts_targ = np.where(pcl[:,3] == 0.0)[0]
pts_dist = np.where(pcl[:,3] == 1.0)[0]
pts_tool = np.where(pcl[:,3] == 2.0)[0]
pcl_color[pts_targ, :] = (0,255,0)
pcl_color[pts_dist, :] = (0,0,255)
pcl_color[pts_tool, :] = (0,0,0)

# Visualize the point cloud.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pcl_color / 255)
o3d.visualization.draw_geometries([pcd])
# ???
# pcd = o3d.geometry.PointCloud()
# o3d.visualization.draw_geometries([pcd])

# Next, do matplotlib visualization, less fine-grained but with coordinates.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(0.4, 0.8)
# ax.set_ylim(-0.6, -0.1)
# ax.set_zlim(-0.2, 0.4)
if len(pcl) >= 5000:
    choice = np.random.choice(len(pcl), size=5000, replace=False)
else:
    choice = np.arange(len(pcl))
ax.scatter(pcl[choice, 0], pcl[choice, 1], pcl[choice, 2])

# Can add a different color if desired.
#if len(pcl) >= 5000:
#    choice = np.random.choice(len(pcl), size=5000, replace=False)
#else:
#    choice = np.arange(len(pcl))
#ax.scatter(pcl[choice, 0], pcl[choice, 1], pcl[choice, 2], color="red")
plt.show()