from collections import defaultdict
from email.policy import default
from matplotlib import pyplot as plt
from os import makedirs, listdir
from os.path import basename, join, split, dirname, isdir
from glob import glob
import numpy as np
import argparse
import pickle as pkl
from pdb import set_trace

def numpify_data(pkls, data_type = 'algorithmic'):
    for pkl_idx in range(len(pkls)):
        print('Processing: {}'.format(pkls[pkl_idx]))
        with open(pkls[pkl_idx], 'rb') as f:
            '''Checking the indivisual PKLs'''
            data = pkl.load(f)
            '''Recreating the dataset starts from here'''
            new_data = defaultdict(list)
            new_data['act_raw'] = data['act_raw'].copy()
            for t in range(len(data['obs'])):
                '''Swapping the ordering here, from the previous encoding to sim encoding'''
                data['obs'][t][3][:, [3, 4, 5]] = data['obs'][t][3][:, [5, 3, 4]]
                '''Deleting the distractor column'''
                pco = np.delete(data['obs'][t][3], 5, 1).copy()
                '''Similar structure as the original data'''
                new_data['obs'].append((data['obs'][t][0].copy(), data['obs'][t][1].copy(), data['obs'][t][2], pco, data['obs'][t][4]))
                if data_type == 'algorithmic':
                    '''Sanity check to make sure we're actually checking the algorithmic dataset, will add more checks here, pretty redundant conditional'''
                    assert new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape[1] == 5 and new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape[0] > new_data['obs'][t][3][new_data['obs'][t][3][:, 4] == 1].shape[0], 'Looks like there are {} and {} instead of 1100 and 300 for this non-dense human dataset'.format(new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape, new_data['obs'][t][3][new_data['obs'][t][3][:, 4] == 1].shape)
                    # assert new_data['obs'][t][3].shape == (1400, 5), 'Looks like this is algorithmic data that does not have 1400 points, but instead has {}'.format(new_data['obs'][t][3].shape)
                elif data_type == 'human':
                    assert new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape[1] == 5 and new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape[0] > new_data['obs'][t][3][new_data['obs'][t][3][:, 4] == 1].shape[0], 'Looks like there are {} and {} instead of 1100 and 100 for this non-dense human dataset'.format(new_data['obs'][t][3][new_data['obs'][t][3][:, 3] == 1].shape, new_data['obs'][t][3][new_data['obs'][t][3][:, 4] == 1].shape)
                    # assert(new_data['obs'][t][3].shape == (1200, 5)), 'Looks like this is human data that does not have 1200 points, but instead has {}. This can be denser data that you are re-formatting or algorithmic'.format(new_data['obs'][t][3].shape)

        pkl_dir_name_save = join(dirname(pkls[pkl_idx]), 'n_by_5')
        '''This will be where the PKLs will be saved'''
        if isdir(pkl_dir_name_save) == False:
            makedirs(pkl_dir_name_save)
        with open(join(pkl_dir_name_save, 'n_by_5_'+basename(pkls[pkl_idx])), 'wb') as w:
            pkl.dump(new_data, w)
        print('Saved: {}'.format(join(pkl_dir_name_save, 'n_by_5_'+basename(pkls[pkl_idx]))))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_loc', type=str, required = True)
    '''Human data had just 1200 points, whereas algorithmic has 1400'''
    p.add_argument('--data_type', type=str, default = 'algorithmic')
    args = p.parse_args()
    pkls = glob(join(args.dataset_loc, '*.pkl'))
    numpified_data = numpify_data(pkls, args.data_type)