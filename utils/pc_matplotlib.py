from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

# pcl = np.load('pcl_3_10.npy')
# ldp = np.load('ldp_3_10.npy')
# ldr = np.load('ldr_3_10.npy')

# new_pcl = np.zeros((1200, 4))

# targ_points = np.where(pcl[:, 3] == 0.0)[0]
# dist_points = np.where(pcl[:, 3] == 1.0)[0]
# tool_points = np.where(pcl[:, 3] == 2.0)[0]

# def rotator(pc_file):
#     #* Function to rotate the ladel model into the upright position, so that it can then be translated and transformed into the end-effector frame
#     pcd_og = o3d.io.read_point_cloud(pc_file)
#     #* The 0.0941 is the scale that we computed from CloudCompare. Check this block of updates in my Notion: https://www.notion.so/Tool-Flow-Experiments-1e3be6c2a51b470f88abf4d1934b93dc#0c2d84a5ee8a4281a143672adc8c7402
#     pcd_og = np.asarray(pcd_og.points) * 0.0941
#     pcd_np = pcd_og.copy()
#     #*Rotation to get it turn it
#     rot_matrix_1 = np.array([[1, 0, 0], [0, np.cos(120 * np.pi/180), -np.sin(120 * np.pi/180)], [0, np.sin(120 * np.pi/180), np.cos(120 * np.pi/180)]])
#     #* Rotation to get it upright 
#     rot_matrix_2 = np.array([[np.cos(90 * np.pi/180), -np.sin(90 * np.pi/180), 0], [np.sin(90 * np.pi/180), np.cos(90 * np.pi/180), 0 ], [0, 0, 1]])

#     pcd_np = np.matmul(pcd_np.copy(), rot_matrix_1)
#     pcd_np = np.matmul(pcd_np.copy(), rot_matrix_2)

#     #* It seems like the center of the ladle is offset from its tip. Just eyeballing it and it seems about right
#     #! NOTE(sarthak): This may be one of the reasons why the ladel looks shifted. If you look at the ladle on the robot, it isn't gripped at the end, but rather at some distance through the tool holder.
#     pcd_np[:, 2] += 0.35
#     return pcd_np

# model_ladle = rotator('dense_ladel.pcd')
# pcd_np = np.matmul(model_ladle, ldr.transpose()) + ldp
# choice = np.random.choice(len(pcd_np), size=1200, replace=False)

# new_pcl[:, :3] = pcd_np[choice]
# new_pcl[:,  3] = 2.0
# new_pcl[:min(100, len(targ_points))] = pcl[targ_points][:min(100, len(targ_points))]
# new_pcl[min(100, len(targ_points)):min(100, len(targ_points)) + min(100, len(dist_points))] = pcl[dist_points][:min(100, len(dist_points))]

# from pdb import set_trace
# set_trace()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')

# print('obs: {} tools: {} targets: {} distractors: {}'.format(len(pcl), len(tool_points), len(targ_points), len(dist_points)))

# ax.scatter(pcl[dist_points, 0], pcl[dist_points, 1], pcl[dist_points, 2], color = 'r', label = 'distractor')
# ax.scatter(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2], color = 'orange', label = 'model tool')
# ax.scatter(pcl[targ_points, 0], pcl[targ_points, 1], pcl[targ_points, 2], color = 'g', label = 'target')
# ax.scatter(pcl[tool_points, 0], pcl[tool_points, 1], pcl[tool_points, 2], color = 'b', label = 'tool')

# ax.set_xlim(0.2, 0.8)
# ax.set_ylim(-0.6, 0.0)
# ax.set_zlim(-0.2, 0.4)

# plt.show()

from glob import glob

frames = glob('obs_*')
eep = glob('eep_*')
pos = glob('pos_*')

frames.sort(key = lambda x: int(x.split('_')[1].split('.npy')[0]))
# eep.sort(key = lambda x: int(x.split('_')[1].split('.npy')[0]))
# pos.sort(key = lambda x: int(x.split('_')[1].split('.npy')[0]))

print(frames, '\n')
print(eep, '\n')
print(pos, '\n')

for idx in range(len(frames)):

    new_pcl = np.load('obs_{}.npy'.format(idx))
    # eep = np.load('eep_{}.npy'.format(idx))
    # pos = np.load('pos_{}.npy'.format(idx))

    print(eep[:3], pos[:3], '\n')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    # print('obs: {} tools: {} targets: {} distractors: {}'.format(len(new_pcl), len(new_pcl[new_pcl[:, 5]==1]), len(new_pcl[new_pcl[:, 3]==1]), len(new_pcl[new_pcl[:, 4]==1])))

    targ_points = np.where(new_pcl[:, 3] == 1.0)[0]
    dist_points = np.where(new_pcl[:, 4] == 1.0)[0]
    tool_points = np.where(new_pcl[:, 5] == 1.0)[0]

    # ax.scatter(eep[0], eep[1], eep[2], label = 'ee', color = 'b')
    # ax.scatter(pos[0], pos[1], pos[2], label = 'pred', color = 'y')
    ax.scatter(new_pcl[dist_points, 0], new_pcl[dist_points, 1], new_pcl[dist_points, 2], color = 'r', label = 'distractor')
    ax.scatter(new_pcl[targ_points, 0], new_pcl[targ_points, 1], new_pcl[targ_points, 2], color = 'g', label = 'target')
    # ax.scatter(new_pcl[tool_points, 0], new_pcl[tool_points, 1], new_pcl[tool_points, 2], color = 'b', label = 'tool')

    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.6, 0.0)
    ax.set_zlim(-0.2, 0.4)

    plt.legend(loc = 'best')

    plt.savefig('demos/yes_{}.png'.format(idx))

    plt.close()

    # plt.show()