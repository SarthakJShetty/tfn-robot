"""This script removes n number of files from the image, end-effector npy files
and then re-orders them sequentially."""

import argparse
from glob import glob
from os import remove
from os.path import split, dirname, join
from shutil import move

def run(file: int, num_remove: int):
    imgs = glob(join(file, 'img_*'))
    imgs.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
    for _ in range(int(num_remove)):
        remove(imgs.pop(0))
    ldrs = glob(join(file, 'ldr_*'))
    ldrs.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
    for _ in range(int(num_remove)):
        remove(ldrs.pop(0))
    ldps = glob(join(file, 'ldp_*'))
    ldps.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
    for _ in range(int(num_remove)):
        remove(ldps.pop(0))
    pcls = glob(join(file, 'pcl_*'))
    pcls.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
    for _ in range(int(num_remove)):
        remove(pcls.pop(0))
    eeps = glob(join(file, 'eep_*'))
    eeps.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
    for _ in range(int(num_remove)):
        remove(eeps.pop(0))

    counter = 0
    for img in imgs:
        move(img, join(dirname(img), 'img_' + str(split(img)[1].split('.')[0][4:].split('_')[0]) + '_' + str(counter) + '.png'))
        counter += 1

    counter = 0
    for pcl in pcls:
        move(pcl, join(dirname(pcl), 'pcl_' + str(split(pcl)[1].split('.')[0][4:].split('_')[0]) + '_' + str(counter) + '.npy'))
        counter += 1

    counter = 0
    for eep in eeps:
        move(eep, join(dirname(eep), 'eep_' + str(split(eep)[1].split('.')[0][4:].split('_')[0]) + '_' + str(counter) + '.npy'))
        counter += 1

    counter = 0
    for ldr in ldrs:
        move(ldr, join(dirname(ldr), 'ldr_' + str(split(ldr)[1].split('.')[0][4:].split('_')[0]) + '_' + str(counter) + '.npy'))
        counter += 1

    counter = 0
    for ldp in ldps:
        move(ldp, join(dirname(ldp), 'ldp_' + str(split(ldp)[1].split('.')[0][4:].split('_')[0]) + '_' + str(counter) + '.npy'))
        counter += 1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type = str, required=True)
    ap.add_argument('--num_remove', type = int, default=1)
    args = ap.parse_args()
    folder = glob(join(args.folder,'policy_data_*'))
    print(folder)
    for sess in folder:
        run(sess, args.num_remove)