import numpy as np
import plotly.graph_objects as go
import pickle as pkl
import imageio
from glob import glob
from os.path import join, split
import argparse
import cv2
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
        x=-T_chart_points[2, ::downsample],
        y=T_chart_points[0, ::downsample],
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )


def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows][::30]
    n_flows = flows[~nonzero_flows][::30]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        # # From daniel
        x_lines.append(-n_pos[i][2])
        y_lines.append(n_pos[i][0])
        z_lines.append(n_pos[i][1])
        x_lines.append(-n_dest[i][2])
        y_lines.append(n_dest[i][0])
        z_lines.append(n_dest[i][1])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=-n_dest[:, 2],
        y=n_dest[:, 0],
        z=n_dest[:, 1],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


def _3d_scene_fixed(x_range, y_range, z_range):
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def create_flow_plot(pts, flow, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current points + (predicted) flow.
    Note: tried numerous ways to add titles and it strangely seems hard. To
    add more info, I'm adjusting the names we supply to the scatter plot and
    increasing its `hoverlabel`.
    """
    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    # print(args.scale_pcl_flow)
    # if args.scale_pcl_flow:
    #     pts = pts  / args.scale_pcl_val
    #     flow = flow / args.scale_pcl_val
    # elif args.scale_targets:
    #     # PointNet++ averaging but with scaling of the targets. TODO make more general.
    #     flow = flow / 250.

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        xaxis = go.layout.XAxis(visible=False, showticklabels=False),
        yaxis = go.layout.YAxis(visible=False, showticklabels=False)
    )

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)
    # f.update_layout(scene1=_3d_scene(sample['points']))
    #! These are the scales that we usually use
    # f.update_layout(scene1=_3d_scene_fixed([0.40, 1.10], [-0.80, -0.10], [0.10, 0.80]))

    f.update_layout(scene1=_3d_scene_fixed([-0.4, 0.6], [-1, 0], [-0.5, 0.5]))
    # f.update_layout(scene1=_3d_scene_fixed([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65]))

    # NOTE(daniel) IDK why this is not working? Would help to show more info.
    f.update_layout(title_text="Flow Plot", title_font_size=10)

    f.update_yaxes(visible=False, showticklabels=False)
    f.update_xaxes(visible=False, showticklabels=False)

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
        # eye=dict(x=0.75, y=.75, z=1.5)
        eye=dict(x=0., y=0, z=1.5)
    )
    f.update_layout(scene_camera=camera)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str)
    parser.add_argument('--demo', default=2, type=int)
    parser.add_argument('--k_step', default=4, type=int)
    parser.add_argument('--sampling', action = 'store_true')
    args = parser.parse_args()

    dir_name = args.loc

    pklFile = join(dir_name, 'BC_pkl_k_2_1_0.pkl'.format(args.k_step, args.demo))
    with open(pklFile, 'rb') as f:
        data = pkl.load(f)
    # points = data['obs'][13][4]['points'].copy()
    # flow = data['obs'][13][4]['flow'].copy()


    from pdb import set_trace
    # set_trace()

    print('Number of actions: {}'.format(len(data['obs'])))

    for data_points in range(len(data['obs'])):

        points = data['obs'][data_points][4]['points']
        flow = data['obs'][data_points][4]['flow']

        assert len(points) == len(flow), 'Check the PKL file: {} Observation: {} is faulty'.format(pklFile, data_points)

        points[:, 0] *= -1
        flow[:, 0]  *= -1
        points[:, [1, 2]] = points[:, [2, 1]]
        flow[:, [1, 2]] = flow[:, [2, 1]]

        if len(points)>400 and args.sampling:
            print('Check if you wanted sampling!')
            sampling = 50
        else:
            sampling = len(points)

        choice = np.random.choice([i for i in range(len(points))], size = sampling, replace = False)

        frame = create_flow_plot(points[choice], flow[choice])

        frame.write_image(join(dir_name, 'flp_{}.png').format(data_points))
        cv2.imwrite(join(dir_name, "img_{}.png".format(data_points)),  data['obs'][data_points][1])

    from PIL import Image

    plotly_files = glob(join(dir_name, 'flp*'))
    imgs_files = glob(join(dir_name, 'img*'))

    plotly_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0])))
    imgs_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0])))

    plotly_images = [Image.fromarray(imageio.imread(plots)).resize((640, 480)) for plots in plotly_files]
    imgs_images = [Image.fromarray(imageio.imread(imgs)).resize((640, 480)) for imgs in imgs_files]

    imageio.mimsave(join(dir_name, 'plotly_k_step_{}.gif'.format(args.k_step)), plotly_images, fps=7)
    imageio.mimsave(join(dir_name, 'images_k_step_{}.gif'.format(args.k_step)), imgs_images, fps=7)

if __name__ == "__main__":
    main()