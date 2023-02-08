"""Data analysis or making GIFS / videos. Easier done with Python3 code.

For example, given the directory like this:
    (mm) seita@twofish:~/catkin_ws/src/mixed-media-physical (main) $ ls -lh data/debug/2022-03-13_16-34-32/
    -rw-rw-r-- 1 seita seita  55M Mar 13 16:35 recording_proc.avi
    -rw-rw-r-- 1 seita seita  55M Mar 13 16:34 recording_raw.avi
    -rw-rw-r-- 1 seita seita  43M Mar 13 16:35 recording_track_tool.avi
We can make a set of GIFs that combines the above.

Example usage:
    (mm) seita@twofish:~/catkin_ws/src/mixed-media-physical (main) $ python analyze.py --data_dir data/debug/2022-03-13_17-19-09/
"""
import os
from os.path import join
import subprocess
import argparse
import cv2
import numpy as np
import glob
import imageio
from moviepy.editor import ImageSequenceClip
from collections import defaultdict

# --------------------------------------------------------------------------------- #
# -------------------------------- various utilities ------------------------------ #
# --------------------------------------------------------------------------------- #

def triplicate(img):
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def read_video(video_path):
    """Read the video and turn to numpy arrays.

    We saved with RGB images, since we have the BGR to RGB conversion. However, I think
    even with that, when we read from cv2 again, it still uses BGR convention, so we need
    to convert things back.
    """
    assert os.path.exists(video_path), video_path
    vcap = cv2.VideoCapture(video_path)
    ret = True
    frames = []
    while ret:
        ret, frame = vcap.read()
        if frame is not None:
            # Convert to RGB mode? I think I need this to get accurate vision.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    X = np.array(frames)
    print(f'From {video_path}, size: {X.shape}')
    return X

# --------------------------------------------------------------------------------------- #
# -------------------------------- various GIF/video stuff ------------------------------ #
# --------------------------------------------------------------------------------------- #

def visualize_pcl(args):
    """Make video from point cloud data.

    Usage is similar to `stack_one_trial()`, just provide a directory from one
    trial, assuming it has point cloud data stored in it! These are normally the
    ones captured as fast as possible (though we can also save point clouds just
    at the level of separate actions).

    Not sure the best way to do this, matplotlib could be doable but I think is
    slow. Open3D might be faster.

    https://stackoverflow.com/questions/34975972/ ?
    """
    dd = args.data_dir
    head = join(dd, 'all_pcls')
    pcl_paths = sorted([
        join(head, pth) for pth in os.listdir(head) if pth[-4:] == '.npy'
    ])
    pcls = [np.load(pcl_path) for pcl_path in pcl_paths]

    # Make 'matplotlib video.'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import shutil
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    tmp_folder = 'tmp_folder'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # The positions of the point cloud should be w.r.t. world / base frame.
    # To find appropriate bounds, can look at the SAFETY_LIMITS in `robot.py`.
    # The x and y ranges should probably match, as well as z (for realism).
    for pidx,pcl in enumerate(pcls):
        print(f'PCL {pidx}, shape: {pcl.shape}')
        fig = plt.figure(figsize=(8,10), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20., azim=-75)
        ax.set_xlim( 0.55,  0.80)
        ax.set_ylim(-0.40, -0.15)
        ax.set_zlim( 0.00,  0.25)

        # Subsample for visual clarity.
        if len(pcl) >= 2000:
            choice = np.random.choice(len(pcl), size=2000, replace=False)
        else:
            choice = np.arange(len(pcl))
        pcl = pcl[choice]

        # Identify segmentation labels (and indices). Careful if we change this!
        i_targ = np.where(pcl[:,3] == 0.0)[0]
        i_dist = np.where(pcl[:,3] == 1.0)[0]
        i_tool = np.where(pcl[:,3] == 2.0)[0]

        # Scater based on color.
        ax.scatter(pcl[i_targ, 0], pcl[i_targ, 1], pcl[i_targ, 2], color='yellow')
        ax.scatter(pcl[i_dist, 0], pcl[i_dist, 1], pcl[i_dist, 2], color='blue')
        ax.scatter(pcl[i_tool, 0], pcl[i_tool, 1], pcl[i_tool, 2], color='black')

        # Save so we can load later.
        savepth = join(tmp_folder, f'scatter_{str(pidx).zfill(3)}.png')
        #plt.tight_layout()  # actually seems worse?
        plt.savefig(savepth, bbox_inches='tight')

    # Load images again to make the video.
    plots = sorted([
        join(tmp_folder,x) for x in os.listdir(tmp_folder) if '.png' in x
    ])
    H, W = 400, 400
    fps = 10
    gif_path = join(dd, f'pcl_{fps}fps_{H}x{W}.gif')
    clip = ImageSequenceClip(plots, fps=fps)
    clip.write_gif(gif_path, fps=fps)

    # Remove the folder.
    shutil.rmtree(tmp_folder)


def stack_one_trial(args):
    """Takes videos from cv2, and turns them to GIF, for ONE trial.

    General use case is if we want to combine a bunch of videos together from
    one directory into side-by-side GIFs for presentation.
    """
    dd = args.data_dir
    VIDS = [
        join(dd, 'video_color_crop.avi'),
        join(dd, 'video_tool_mask.avi'),
        join(dd, 'video_targ_mask.avi'),
    ]
    DATA = [read_video(vid) for vid in VIDS]

    # Actually the rest of this is similar to `stack_multiple_trials`.
    frames = []
    H,W = 200,200

    # If videos have unequal counts of frames, store last frame in case we repeat.
    # Though I don't think this is going to be an issue if we use the same trial.
    last_frame = [None for _ in range(len(DATA))]

    # For each time step `t`, go through the files in DATA, and stack frames.
    max_time_steps = max([d.shape[0] for d in DATA])
    for t in range(max_time_steps):
        fs = []  # Contains 1 frame per video (all at this time step).
        for j in range(len(DATA)):
            if t < len(DATA[j]):
                frame = DATA[j][t].copy()
                frame = cv2.resize(frame, (H,W))
                last_frame[j] = frame
            else:
                frame = last_frame[j]
            fs.append(frame)
        frame = np.hstack(fs)
        frames.append(frame)

    # Actually write to file.
    fps = 10
    gif_path = join(dd, f'stack_vids_{fps}fps_{H}x{W}.gif')
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_gif(gif_path, fps=fps)


def stack_multiple_trials(args):
    """Takes videos from cv2, and turns them to GIF.

    Here we have a directory of trials, and want to stack them together. For example,
    with 4 separate episodes, we might want to merge the processed video recordings.
    """
    # # Algorithmic with 1 yellow ball and 0 distractors:
    #HEAD = 'data/policy_alg_pix_ntarg_01_ndist_00_maxT_10'
    #VIDS = [
    #    join(HEAD, '2022-03-27_18-40-10', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-43-37', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-44-48', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-49-27', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-50-56', 'video_color_crop.avi'),
    #]
    #VIDS = [
    #    join(HEAD, '2022-03-27_18-51-58', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-54-27', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-56-01', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-57-16', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_18-58-38', 'video_color_crop.avi'),
    #]

    # # Algorithmic with 1 yellow ball and 1 distractors:
    #HEAD = 'data/policy_alg_pix_ntarg_01_ndist_01_maxT_10'
    #VIDS = [
    #    join(HEAD, '2022-03-27_19-02-28', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-05-44', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-08-42', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-10-07', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-12-17', 'video_color_crop.avi'),
    #]
    #VIDS = [
    #    join(HEAD, '2022-03-27_19-13-25', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-15-01', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-16-30', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-21-45', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-24-03', 'video_color_crop.avi'),
    #]

    # Algorithmic with 1 yellow ball and 04 distractors:
    #HEAD = 'data/policy_alg_pix_ntarg_01_ndist_04_maxT_10'
    #VIDS = [
    #    join(HEAD, '2022-03-27_19-29-49', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-32-51', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-34-53', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-36-00', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-40-25', 'video_color_crop.avi'),
    #]
    #VIDS = [
    #    join(HEAD, '2022-03-27_19-42-37', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-44-35', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-48-22', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-49-50', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-51-42', 'video_color_crop.avi'),
    #]

    # Algorithmic with 1 yellow ball and 06 distractors:
    #HEAD = 'data/policy_alg_pix_ntarg_01_ndist_06_maxT_10'
    #VIDS = [
    #    join(HEAD, '2022-03-27_20-28-39', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-30-28', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-33-10', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-34-49', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-37-49', 'video_color_crop.avi'),
    #]
    #VIDS = [
    #    join(HEAD, '2022-03-27_20-39-31', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-43-09', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-45-48', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-47-41', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-48-33', 'video_color_crop.avi'),
    #]

    # Algorithmic with 1 yellow ball and 08 distractors:
    #HEAD = 'data/policy_alg_pix_ntarg_01_ndist_08_maxT_10'
    #VIDS = [
    #    join(HEAD, '2022-03-27_19-57-30', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_19-59-18', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-01-40', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-02-34', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-07-53', 'video_color_crop.avi'),
    #]
    #VIDS = [
    #    join(HEAD, '2022-03-27_20-08-46', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-11-01', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-15-09', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-19-36', 'video_color_crop.avi'),
    #    join(HEAD, '2022-03-27_20-20-29', 'video_color_crop.avi'),
    #]

    ## For testing rotations
    VIDS = [
        join('data', 'debug', '2022-04-03_18-07-44', 'video_color_crop.avi'),
        join('data', 'debug', '2022-04-03_18-11-05', 'video_color_crop.avi'),
    ]

    # Read the data from videos. Also, no need to crop!
    DATA = [read_video(VID) for VID in VIDS]
    frames = []
    H,W = 200,200

    # Videos have unequal counts of frames, store last frame in case we repeat.
    last_frame = [None for _ in range(len(DATA))]

    # For each time step `t`, go through the files in DATA, and stack frames.
    max_time_steps = max([d.shape[0] for d in DATA])
    for t in range(max_time_steps):
        fs = []  # Contains 1 frame per video (all at this time step).
        for j in range(len(DATA)):
            if t < len(DATA[j]):
                frame = DATA[j][t].copy()
                frame = cv2.resize(frame, (H,W))
                last_frame[j] = frame
            else:
                frame = last_frame[j]
            fs.append(frame)
        frame = np.hstack(fs)
        frames.append(frame)

    # Actually write to file.
    fps = 10
    gif_path = f'stack_multiple_trials_{fps}fps.gif'
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_gif(gif_path, fps=fps)

# ------------------------------------------------------------------------------------ #
# -------------------------------- testing segmentation ------------------------------ #
# ------------------------------------------------------------------------------------ #

"""
Some possibly useful references but keep in mind if using BGR or RGB mode:
https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
Red, blue, yellow, gray boundaries (for BGR mode I think):
    boundaries = [
        ( [17, 15, 100], [50, 56, 200]),
        (   [86, 31, 4], [220, 88, 50]),
        ([25, 146, 190], [62, 174, 250]),
        ( [103, 86, 65], [145, 133, 128])
    ]

If green colors, see:
https://pyimagesearch.com/2015/09/21/opencv-track-object-movement/

For white, I've tried these bounds:
    bounds = [
        ([150,150,150], [255,255,255]),
        ([180,180,180], [255,255,255]),
        ([200,200,200], [255,255,255]),
        ([220,220,220], [255,255,255]),
        ([240,240,240], [255,255,255]),
    ]
It seems like with food coloring, we can get a reasonable white segmentation using
the threshold of 180 or 200.
"""

SEGM_PARAMS = {
    # Clear box (from myself), black curved ladle, yellow and blue ping-pong
    # balls, red food coloring (7-8 drops). Yellow is the target. These use
    # color ranges, not HSV or other spaces. NOTE! These use RGB, not BGR ordering,
    # so if we find good values for these, we have to swap the 1st and 3rd parts of
    # the lower array (and upper array).
    'set_01_color': {
        'ball_targ': [
            ([190, 146,  25], [255, 200,  62]),
            ([170, 125,  25], [255, 255, 100]),  # lgtm?
            ([190, 146,  25], [255, 255, 100]),  # lgtm?
            ([190, 125,   0], [255, 255, 100]),  # lgtm?
            ([190, 100,  25], [255, 255, 100]),
            ([120,  80,   0], [255, 255, 120]),
        ],
        'ball_dist': [
            ([ 17,  15, 100], [ 50,  56, 200]),
            ([  0,   0,  50], [100, 100, 255]),
            ([  0,   0, 100], [100, 100, 200]),  # lgtm?
            ([  0,   0, 100], [100, 100, 255]),
            ([  0,   0,  50], [100, 100, 200]),
            ([  0,   0,   0], [100, 100, 200]),
        ],
        'water': [
            ([  0,   0,   0], [220,  88,  50]),
            ([ 86,  31,   4], [220,  88,  50]),
            ([ 86,  31,   4], [220, 120, 100]),
            ([ 86,  31,   4], [255, 120, 100]),
            ([ 50,   0,   0], [255, 120, 100]),  # lgtm?
        ],
        'ladle': [
            ([  0,   0,   0], [ 20,  20,  20]),
            ([  0,   0,   0], [ 40,  40,  40]),
            ([  0,   0,   0], [ 60,  60,  60]),  # lgtm?
            ([  0,   0,   0], [ 80,  80,  80]),
            ([  0,   0,   0], [100, 100, 100]),
        ],
    },

    # Clear box (from myself), black curved ladle, yellow and blue ping-pong
    # balls, red food coloring (7-8 drops). Yellow is the target. Uses HSV.
    'set_01_hsv': {
        'ball_targ': [
        ],
        'ball_dist': [
        ],
        'water': [
        ],
        'ladle': [
        ],
    },
}

def test_segm_one_dir_channels(args):
    """Test segmentation.

    Here, let's try and get segmentation as closely related to SoftGym sim as possible.
    Thus, we are still stacking images / GIFs, except these will be to visualize different
    channels instead of tuning the parameters. So here we can just copy and paste the best
    bounds from earlier.

    For sim, the code is here:
        https://github.com/mooey5775/softgym_MM/blob/dev_daniel/softgym/utils/visualization.py
    This assumes we already did the segmentation, which is acutally done here for SoftGym:
        https://github.com/mooey5775/softgym_MM/blob/dev_daniel/softgym/utils/segmentation.py
    But we don't have ground-truth data in sim.
    """
    n_segm = 6

    # Loading videos, etc.
    VID = join(args.data_dir, 'recording_proc.avi')
    GIF_GRAY = join(args.data_dir, f'recording_proc_segm_{args.fps}fps_channels_gray.gif')
    DATA = read_video(VID)

    # Try to see if these are somewhat aligned with other methods. These are for
    # cropping the _processed_ images, watch out! Keep ww=hh please.
    xx = 200
    yy = 170
    ww = 300
    hh = 300
    obs_l_raw = []
    obs_l_seg = []
    print(f'Size of proc. images: {DATA[0].shape}')  # e.g., (720,720)

    # Segment images from each time step, and stack them.
    for t in range(len(DATA)):
        fr_raw = DATA[t]

        # Segmentation. This is from the procesed version, keep in mind when
        # computing bounding boxes, etc. Also that we only have 300x300 images.
        fr_segm = fr_raw.copy()
        crop_segm = fr_segm[yy:yy+hh, xx:xx+ww]

        # Now actually do the segmentation, in the same order that we apply in sim?
        # Doing it this way means we can choose which classes override other classes.
        img_segm = np.zeros((ww,hh)).astype(np.uint8)

        # Segment [black] tool, potentially overriding parts of the glass?
        ll, uu = ([  0,   0,   0], [ 60,  60,  60])
        ladle_lo = np.array(ll, dtype='uint8')
        ladle_up = np.array(uu, dtype='uint8')
        ladle_mask = cv2.inRange(crop_segm, ladle_lo, ladle_up)  # binary mask {0,255}
        img_segm[ ladle_mask > 0 ] = 1  # BTW, we do need the `> 0` here

        # Segment [red] water, potentially overriding parts of the glass or the tool.
        ll, uu = ([ 50,   0,   0], [255, 120, 100])
        water_lo = np.array(ll, dtype='uint8')
        water_up = np.array(uu, dtype='uint8')
        water_mask = cv2.inRange(crop_segm, water_lo, water_up)  # binary mask {0,255}
        img_segm[ water_mask > 0 ] = 2

        # Segment the target [yellow] item(s), potentially overiding any prior stuff.
        ll, uu = ([170, 125,  25], [255, 255, 100])
        targ_lo = np.array(ll, dtype='uint8')
        targ_up = np.array(uu, dtype='uint8')
        targ_mask = cv2.inRange(crop_segm, targ_lo, targ_up)  # binary mask {0,255}
        img_segm[ targ_mask > 0 ] = 3

        # Distractor(s). Blue distractor items.
        ll, uu = ([  0,   0, 100], [100, 100, 200])
        dist_lo = np.array(ll, dtype='uint8')
        dist_up = np.array(uu, dtype='uint8')
        dist_mask = cv2.inRange(crop_segm, dist_lo, dist_up)  # binary mask {0,255}
        img_segm[ dist_mask > 0 ] = 4

        # Actually return a _binary_ segmented image, but to be consistent with RGB,
        # (1) put values in (0,255) and (2) return a uint8. This could be the input
        # to the policy (modulo possibly changes)?
        segmented = np.zeros((ww, hh, n_segm)).astype(np.float32)
        segmented[:, :, 0] = img_segm == 0  # outside
        segmented[:, :, 1] = img_segm == 1  # tool pix, but unlike in sim, have occlusions
        #segmented[:, :, 2] = img_segm == 1  # glass  (NOTE(daniel): SKIPPING for now)
        segmented[:, :, 3] = img_segm == 2  # water
        segmented[:, :, 4] = img_segm == 3  # target item
        segmented[:, :, 5] = img_segm == 4  # distractor (if any)
        segmented = (segmented * 255.0).astype(np.uint8)

        # Add for later.
        obs_l_raw.append(crop_segm)
        obs_l_seg.append(segmented)

    # We have that, now convert it to what we have used for SoftGym visualization.
    # Each list consists of one of the segmentation channels we extract.
    all_segm_frames = []
    print(f'Saving segmentations with {n_segm} channels to {GIF_GRAY}!')
    print('Order: (1) RGB, (2) outside, (3) tool, (4) water, (5) targ, (6) dist.')
    for idx,(obs_r,obs_s) in enumerate(zip(obs_l_raw, obs_l_seg)):
        assert obs_s.shape[2] == n_segm, f'{obs_s.shape} vs {n_segm}'
        current_segms_l = [triplicate(obs_s[:,:,c]) for c in range(n_segm)
                if c != 2]  # NOTE(daniel): only because I skip the glass.
        current_frames = [obs_r] + current_segms_l
        concat_segm = np.concatenate(current_frames, axis=1)
        all_segm_frames.append(concat_segm)

    # Make a `combo` which combines the prior segmentation channels.
    clip = ImageSequenceClip(all_segm_frames, fps=args.fps)
    clip.write_gif(GIF_GRAY, fps=args.fps)


def test_segm_one_dir_params(args):
    """Test segmentation.

    This is meant for testing one specific directory, where we change the different
    segmentation values (e.g., different color thresholds) and compare them to see
    which ones look better. See the `SEGM_PARAMS` dict for a reasonable set of params
    as a function of the food colors, container, etc.

    Also this is meant for testing 1 type of value, such as the color of the target
    balls, instead of segmenting the whole thing.

    NOTE(daniel): testing in Python3 for ease of saving GIFS and various analysis there,
    but please refrain from using any Python 3 specific OpenCV code (or note how it can be
    done in Python2). But also we have to see if we can get this to be fast enough in the
    Python2 ROS code for the policy to be sufficiently closed-loop.

    The way to interpret the resulting GIFs (other than the originals):
    - Gray: these are binary images, where white indicates the thing we are detecting.
        Black is everything else.
    - Color: these are color images. If we are detecting something, then keep its color,
        else turn the color to black. TL;DR keep pixels only if the mask is white (255).
    """

    # Pick the bounds that we want to test, and also if color or hsv.
    overall_key = 'set_01_color'
    #overall_key = 'set_01_hsv'
    #key = 'ball_targ'
    #key = 'ball_dist'
    #key = 'water'
    key = 'ladle'
    BOUNDS = SEGM_PARAMS[overall_key][key]

    # Loading videos, etc.
    VID = join(args.data_dir, 'recording_proc.avi')
    GIF_GRAY  = join(args.data_dir, f'recording_proc_segm_{args.fps}fps_{key}_gray.gif')
    GIF_COLOR = join(args.data_dir, f'recording_proc_segm_{args.fps}fps_{key}_color.gif')
    DATA = read_video(VID)

    # Try to see if these are somewhat aligned with other methods. These are for
    # cropping the _processed_ images, watch out! Keep ww=hh please.
    xx = 200
    yy = 170
    ww = 300
    hh = 300
    frames_gray = []
    frames_color = []
    print(f'Size of proc. images: {DATA[0].shape}')  # e.g., (720,720)

    # Segment images from each time step, and stack them.
    for t in range(len(DATA)):
        fr_raw = DATA[t]

        # Segmentation. This is from the procesed version, keep in mind when
        # computing bounding boxes, etc. Also that we only have 300x300 images.
        # TODO(daniel) deprecated, we might as well do the entire cropping beforehand.
        fr_segm = fr_raw.copy()
        cv2.rectangle(fr_segm, (xx,yy), (xx+ww, yy+hh), (0,255,0), 3)

        # Crop images.
        crop_orig = fr_raw[yy:yy+hh, xx:xx+ww]
        crop_segm = fr_segm[yy:yy+hh, xx:xx+ww]

        # Start by stacking here:
        stacked_t_c = [crop_orig]
        stacked_t_g = [triplicate(cv2.cvtColor(crop_orig, cv2.COLOR_RGB2GRAY))]

        # Color segmentation? I think we have RGB, not BGR. Test different bounds.
        for bb in BOUNDS:
            # find the colors within the specified boundaries and apply # the mask
            lower = np.array(bb[0], dtype='uint8')
            upper = np.array(bb[1], dtype='uint8')
            mask = cv2.inRange(crop_segm, lower, upper) # binary mask, {0,255}
            output = cv2.bitwise_and(crop_segm, crop_segm, mask=mask)
            stacked_t_g.append(triplicate(mask))  # grayscale, triplicated
            stacked_t_c.append(output)  # color, 3 channels

        # Now we can stack.
        frame_gray  = np.hstack(stacked_t_g)
        frame_color = np.hstack(stacked_t_c)
        frames_gray.append(frame_gray)
        frames_color.append(frame_color)

    # Actually write to file, with color and grayscale variants. Do NOT use
    # `np.array(frames_gray)`, it creates an annoying null character error.
    clip_gray  = ImageSequenceClip(frames_gray, fps=args.fps)
    clip_color = ImageSequenceClip(frames_color, fps=args.fps)
    clip_gray.write_gif(GIF_GRAY, fps=args.fps)
    clip_color.write_gif(GIF_COLOR, fps=args.fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--fps', type=int, default=10)
    args = p.parse_args()

    # Combine videos from trials to compare stuff (e.g., for repeatability).
    #visualize_pcl(args)
<<<<<<< HEAD
    # stack_one_trial(args)
    stack_multiple_trials(args)
=======
    stack_one_trial(args)
    #stack_multiple_trials(args)
>>>>>>> 63ce1b2118d77f3757452540feeabd733fb9a9f4

    # Test segmentation. For these please see the `SEGM_PARAMS` dict.
    #test_segm_one_dir_params(args)   # stack GIFs to tune params
    #test_segm_one_dir_channels(args)  # stack GIFs, one per channel
