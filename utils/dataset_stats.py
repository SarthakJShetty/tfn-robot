from matplotlib import pyplot as plt
from os.path import basename, join, split, dirname
from glob import glob
import numpy as np
import argparse
import pickle as pkl

def numpify_data(pkls):
    '''What needs to go in here?
    1. Get the PKLs listed in this location
    2. Extract them and get the flow for each observation. 
    3. Run standard numpy/pandas functions to get the mean and range'''
    all_pkl_stats = np.empty((0, 6))
    for pkl_idx in range(len(pkls)):
        with open(pkls[pkl_idx], 'rb') as f:
            data = pkl.load(f)
            pkl_stats = np.zeros((len(data['act_raw']), 6))
            for t in range(len(data['act_raw'])):
                pkl_stats[t, :] = data['act_raw'][t]
        all_pkl_stats = np.append(all_pkl_stats, pkl_stats, axis = 0)
    print('Number of PKLs: {}\nNumber of data points: {}'.format(len(pkls), len(all_pkl_stats)))
    return all_pkl_stats

def histogram(numpy_data, dataset_loc):
    '''Plot the range of values along X, Y and Z'''
    fig, axs  = plt.subplots(ncols = 3, figsize = (20, 7))

    axes = ['X', 'Y', 'Z']

    for t in range(3):
        axs[t].hist(numpy_data[:, t], bins = 50)
        axs[t].set_box_aspect(1)
        axs[t].set_title('Stats along {} directon\nMean: {} m\nMedian: {} m\nRange of action: {} to {}'.format(axes[t], np.mean(numpy_data[:, t], axis = 0).round(decimals = 6), np.median(numpy_data[:, t], axis = 0).round(decimals = 6), np.min(numpy_data[:, t], axis = 0).round(decimals = 6), np.max(numpy_data[:, t], axis = 0).round(decimals = 6)))

    plt.savefig(join(dataset_loc, 'dataset_stats.png'))

    return np.mean(numpy_data[:, :3], axis = 0), np.min(numpy_data[:, :3], axis = 0), np.max(numpy_data[:, :3], axis = 0)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_loc', type=str, required = True)
    args = p.parse_args()
    pkls = glob(join(args.dataset_loc, '*.pkl'))
    numpified_data = numpify_data(pkls)
    axis_avgs, axis_mins, axis_maxs = histogram(numpified_data, args.dataset_loc)
    print('Mean of action along:\nX axis: {} m\nY axis: {} m\nZ axis: {} m\nRange of actions: X axis: {} to {}\nRange of actions: Y axis: {} to {}\nRange of actions: Z axis: {} to {}'.format(axis_avgs[0], axis_avgs[1], axis_avgs[2], axis_mins[0], axis_maxs[0], axis_mins[1], axis_maxs[1], axis_mins[2], axis_maxs[2]))