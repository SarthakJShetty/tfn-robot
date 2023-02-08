"""This is an extra script that can be used to rewrite the demonstration folders, prior to running visualization.py
visualization.py expects the folders to be in sequential form, that's why we need this script occasionally.
"""

from shutil import move, copytree
from glob import glob
import argparse
from os.path import join, dirname

def run(folder: str, offset: int):
    files = glob(join(folder, 'policy_data_*'))
    for file in files:
        copytree(file, join(dirname(file), 'policy_folder_{}'.format(offset+files.index(file))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type = str, required=True)
    ap.add_argument('--offset', type = int, required=True)
    args = ap.parse_args()
    run(args.folder, args.offset)