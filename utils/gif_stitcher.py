import imageio
import numpy as np
import argparse
from os.path import dirname, join

def stitch(gif_list):
    '''This function basically horizontally stitches together N number of GIFs
    that are passed to it through the gif_list variable'''

    print('[INFO] Received {} GIFs: {}'.format(len(gif_list), gif_list))

    gifs = []
    for gif in gif_list:
        '''Appending all the GIFs to a list here'''
        gifs.append(imageio.get_reader(gif))
    print('[INFO] Added GIFs to a list')
    ''' Determining the minimum number of frames here so that we can capture the same
    number of frames across all GIFs'''
    number_of_frames = min(x.get_length() for x in gifs)
    print('[INFO] Minimum number of frames: {}'.format(number_of_frames))
    '''Creating the resultant GIF here'''
    stitched_gif = imageio.get_writer(join(dirname(gif_list[0]), 'stitched_{}.gif').format(len(gif_list)))
    print('[INFO] GIF will be stored here: {}/stitched_{}.gif'.format(dirname(gif_list[0]), len(gif_list)))
    for _ in range(number_of_frames):
        '''Iterating through the frames of each GIF'''
        stitched_img = []
        for gif in range(len(gif_list)):
            '''Stacking the individual frames horizontally'''
            stitched_img.append(gifs[gif].get_next_data())
        stacked_img = np.hstack((stitched_img))
        '''Adding the stitched frame to the resultant GIF'''
        stitched_gif.append_data(stacked_img)
    print('[INFO] GIF-ed! {} frames from {} GIFs'.format(number_of_frames, len(gif_list)))
    stitched_gif.close()
    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gif_list', nargs='+')
    args = p.parse_args()
    stitch(args.gif_list)