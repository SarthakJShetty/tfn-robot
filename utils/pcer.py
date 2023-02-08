"""Stand-alone script to visualize pointclouds given some folder and the
.npy filename pattern that we're interested in"""

from glob import glob
from os.path import split, join
import numpy as np
import argparse
from matplotlib import pyplot as plt

def pcer(pc_files, encoding_type):
    '''Barebones script to check the observations being sent to the model'''
    dir_name = split(pc_files[0])[0]

    # pc_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))

    counter = 0


    for pc_file in pc_files:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # demo = split(pc_file)[1].split('.')[0][4:].split('_')[0]
        # iteration = split(pc_file)[1].split('.')[0][4:].split('_')[1]

        pcl = np.load(pc_file)

        if encoding_type == 'tool':
            tool_points = np.where(pcl[:, 3] == 1.0)[0]
            targ_points = np.where(pcl[:, 4] == 1.0)[0]
        elif encoding_type == 'targ':
            targ_points = np.where(pcl[:, 3] == 1.0)[0]
            tool_points = np.where(pcl[:, 4] == 1.0)[0]
        elif encoding_type == 'full':
            targ_points = np.where(pcl[:, 3] == 1.0)[0]
            tool_points = np.where(pcl[:, 5] == 1.0)[0]

        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(-0.6, 0.0)
        ax.set_zlim(-0.2, 0.4)

        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')

        ax.scatter(pcl[tool_points, :][Ellipsis, 0], pcl[tool_points, :][Ellipsis, 1], pcl[tool_points, :][Ellipsis, 2], color="b", label='tool points', s=2)
        ax.scatter(pcl[targ_points, :][Ellipsis, 0], pcl[targ_points, :][Ellipsis, 1], pcl[targ_points, :][Ellipsis, 2], color="g", label='target points', s=5)

        plt.legend(loc='best')
        plt.savefig(join(dir_name, 'pc_{}.png'.format(counter)))
        plt.close()
        counter += 1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_loc', type=str, default = './')
    ap.add_argument('--encoding_type', type = str, default = 'tool')
    args = ap.parse_args()
    pc_files = glob(join(args.dataset_loc, 'start_obs_*.npy'))
    pcer(pc_files, args.encoding_type)
