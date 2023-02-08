import numpy as np
from macpath import dirname
import plotly.graph_objects as go
import pickle as pkl
import imageio
from glob import glob
from os.path import join, split, basename
import argparse
from pdb import set_trace

def pointcloud(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene", name=None) -> go.Scatter3d:
    marker_dict = {"size": 3}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]
    return go.Scatter3d(
        x=-T_chart_points[0, ::downsample],
        y=T_chart_points[2, ::downsample],
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )

def _3d_scene_fixed(x_range, y_range, z_range):
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def create_flow_plot(pts, sizeref=2.0, args=None):
    pts_name = f'pts {pts.shape}'

    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    colors = np.zeros_like(pts[:, :3])
    # tool_pts = pts[]
    # ball_pts = pts[pts[:, 4] == 1.]
    EVAL_COLOR = np.array([255, 0, 0])
    from pdb import set_trace
    # set_trace()
    colors[pts[:, 3] == 1.] += EVAL_COLOR

    TEST_COLOR = np.array([255, 0, 0])
    colors[pts[:, 4] == 1.] += TEST_COLOR

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name)
    )

    f.update_layout(scene1=_3d_scene_fixed([0.2, 1.2], [-0.8, 0.2], [-0.5, 0.5]))
    # f.update_layout(scene1=_3d_scene_fixed([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65]))

    # NOTE(daniel) IDK why this is not working? Would help to show more info.
    f.update_layout(title_text="Pointcloud Plot", title_font_size=10)
    f.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    _adjust_camera_angle(f)
    # f.show()
    return f

def _adjust_camera_angle(f):
    """Adjust default camera angle if desired.
    For default settings: https://plotly.com/python/3d-camera-controls/
    """
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=0.0, z=0.0)
    )
    f.update_layout(scene_camera=camera)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_loc', type=str)
    parser.add_argument('-sampling', action = 'store_true')
    args = parser.parse_args()

    dir_name = args.dataset_loc

    pcl_files = glob(join(dir_name, 'start_obs_*'))

    random_start_eval = np.random.randint(0, len(pcl_files))

    npy_file = 'start_obs_101.npy'

    # test_files = glob(join('test_eval_pc/test_pc', 'obs_*'))

    # random_start_test = np.random.randint(0, len(test_files))

    points = np.empty((0, 5))

    # set_trace()

    for pcl_file in [pcl_files[random_start_eval]]:

        new_points = np.load(join(dir_name, npy_file))
        new_points[:, 0] *= -1
        new_points[:, [1, 2]] = new_points[:, [2, 1]]
        points = np.append(points, new_points, axis = 0)

        from pdb import set_trace
        # set_trace()

        frame = create_flow_plot(points)

        frame.show()

        frame.write_html(join(dir_name, 'example_{}.html'.format(npy_file)))

    # frame.write_html(join(dir_name, 'flp_random_{}_{}.html').format(basename(pcl_files[random_start_test])[:-4], basename(pcl_files[random_start_eval])[:-4], test_files.index(pcl_file)))

if __name__ == "__main__":
    main()