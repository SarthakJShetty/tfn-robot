import argparse
from shutil import copy, move
from glob import glob
from os import makedirs
from os.path import basename, join, isdir
import tarfile

def collect(src, dst, k_steps):
    '''What does this function do?
    1. Looks at all the files in the given src and gathers all the PKLs
    2. Moves them to dst
    3. TARS them'''

    # Here we grab all the folders containing the PKLs
    folders = glob(join(src, 'policy_data_*'))
    # We make a folder for the dst if it does not exist
    makedirs(dst) if not isdir(dst) else print('{} Exists'.format(dst))
    for folder in folders:
        # Grabbing all the pkls in each of the policy_data_* folders
        pkls = glob(join(folder, 'BC_k_steps_{}_demo_*.pkl').format(k_steps))
        for pkl in pkls:
            # Getting the SESSION from the policy_data_SESSION filename
            session = basename(folder).split('_')[-1]
            # Getting the PKL_EXT from thisOne_k_step_1_BC_PKL_EXT
            pkl_ext = basename(pkl).split('_')[-1]
            print(pkl, session, pkl_ext)
            # Moving the PKL to the dst specified
            move(pkl, join(dst, 'BC_pkl_{}_{}').format(session, pkl_ext))
    # TARing the entire folder to which we copied all the PKLs to
    with tarfile.open(join(dst, 'dataset.tar'), 'w:gz') as tar:
        tar.add(dst, arcname=basename(dst))

    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # SRC folder containing the data from all the demonstration
    p.add_argument('--src', type = str, default='data/')
    # DST folder to which all the PKLs are to be transferred to
    p.add_argument('--dst', type = str, default='pkls')
    # The same src folder might have PKLs from different k_steps, so we specify the k_steps here
    p.add_argument('--k_steps', type = int, required=True)
    args = p.parse_args()
    collect(args.src, args.dst, args.k_steps)