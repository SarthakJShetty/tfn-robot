from builtins import IndexError
from os.path import join, split, dirname, isdir, basename
from os import listdir, makedirs
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import imageio
from PIL import Image
from collections import defaultdict, Counter
import cv2
import pickle as pkl
from pyquaternion import Quaternion as quat
import argparse 

def rotator(pc_file):
    #* Function to rotate the ladel model into the upright position, so that it can then be translated and transformed into the end-effector frame
    pcd_og = o3d.io.read_point_cloud(pc_file)
    #* The 0.0941 is the scale that we computed from CloudCompare. Check this block of updates in my Notion: https://www.notion.so/Tool-Flow-Experiments-1e3be6c2a51b470f88abf4d1934b93dc#0c2d84a5ee8a4281a143672adc8c7402
    pcd_og = np.asarray(pcd_og.points) * 0.0941
    pcd_np = pcd_og.copy()
    #*Rotation to get it turn it
    rot_matrix_1 = np.array([[1, 0, 0], [0, np.cos(120 * np.pi/180), -np.sin(120 * np.pi/180)], [0, np.sin(120 * np.pi/180), np.cos(120 * np.pi/180)]])
    #* Rotation to get it upright 
    rot_matrix_2 = np.array([[np.cos(90 * np.pi/180), -np.sin(90 * np.pi/180), 0], [np.sin(90 * np.pi/180), np.cos(90 * np.pi/180), 0 ], [0, 0, 1]])

    pcd_np = np.matmul(pcd_np.copy(), rot_matrix_1)
    pcd_np = np.matmul(pcd_np.copy(), rot_matrix_2)

    #* It seems like the center of the ladle is offset from its tip. Just eyeballing it and it seems about right
    #! NOTE(sarthak): This may be one of the reasons why the ladel looks shifted. If you look at the ladle on the robot, it isn't gripped at the end, but rather at some distance through the tool holder.
    pcd_np[:, 2] += 0.33
    return pcd_np

def pcer(pc_files, encoding):

    #* Extarcting this so that we can save the images that we generate from the npy files
    dir_name = split(pc_files[0])[0]
    #! Moving this here so that we don't keep querying the model from disc
    model_ladle = rotator('assets/dense_ladel.pcd')

    for pc_file in pc_files[:]:
        demo = split(pc_file)[1].split('.')[0][4:].split('_')[0]
        iteration = split(pc_file)[1].split('.')[0][4:].split('_')[1]
        # print(demo, iteration)
        pcl = np.load(pc_file)
        pcn = np.load(glob(join(dir_name, 'pcl_{}_{}.npy').format(demo, iteration))[0])
        pose = np.load(glob(join(dir_name, 'eep_{}_{}.npy').format(demo, iteration))[0])
        print(basename(glob(join(dir_name, 'pcl_{}_{}.npy').format(demo, iteration))[0]), basename(glob(join(dir_name, 'eep_{}_{}.npy').format(demo, iteration))[0]))
        ldp = pose[:3]
        ldr = quat(pose[3:]).rotation_matrix

        #* This is the scanned ladle file. There are a bunch of PCD viewer extensions available for VS Code that you can use to view this. The rotor function just rotates it
        #* to the upright position before sampling or applying any transformations to it.
        # print('LDP: {} LDR: {} Ladle: {}'.format(ldp.shape, ldr.shape, model_ladle.shape))

        # Try to add color to the point clouds.
        # pcl_color = np.zeros((pcl.shape[0], 3))
        #! Since we rearranged the index with which the tool pointcloud is saved (now 0 as opposed to 2 earlier)
        pts_targ = np.where(pcl[:,3] == 0.0)[0]
        ptn_targ = np.where(pcn[:,3] == 0.0)[0]
        pts_dist = np.where(pcl[:,3] == 1.0)[0]
        pts_tool = np.where(pcl[:,3] == 2.0)[0]

        #* Moving ladel into end-effector frame
        print('Demo:{} Iteration:{}'.format(demo, iteration))
        pcd_np = np.matmul(model_ladle, ldr.transpose()) + ldp
        # pcd_np_old = np.matmul(model_ladle, ldr_old.transpose()) + ldp_old
        new_pcl = np.zeros((1400, 4))
        # new_pcl_old = np.zeros((1400, 4))
        choice = np.random.choice(len(pcd_np), size=1400, replace=False)

        # new_pcl_old[:, :3] = pcd_np_old[choice]
        # new_pcl_old[:,  3] = 2.0
        # new_pcl_old[:min(300, len(pts_targ))] = pcl[pts_targ][:min(300, len(pts_targ))]
        # new_pcl_old[min(300, len(pts_targ)):min(300, len(pts_targ)) + min(300, len(pts_dist))] = pcl[pts_dist][:min(300, len(pts_dist))]

        new_pcl[:, :3] = pcd_np[choice]
        new_pcl[:,  3] = 2.0
        new_pcl[:min(300, len(pts_targ))] = pcl[pts_targ][:min(300, len(pts_targ))]
        new_pcl[min(300, len(pts_targ)):min(300, len(pts_targ)) + min(300, len(pts_dist))] = pcl[pts_dist][:min(300, len(pts_dist))]

        targ_points = np.where(new_pcl[:, 3] == 0.0)[0]
        dist_points = np.where(new_pcl[:, 3] == 1.0)[0]
        tool_points = np.where(new_pcl[:, 3] == 2.0)[0]

        # tool_points_old = np.where(new_pcl_old[:, 3] == 2.0)[0]

        # pcl_color[pts_targ, :] = (0,255,0)
        # pcl_color[pts_dist, :] = (0,0,255)
        # pcl_color[pts_tool, :] = (0,0,0)

        # Visualize the point cloud.
        #! Commenting this out so that we can speed up the data generation
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(pcl_color / 255)
        # o3d.io.write_point_cloud(join(dir_name, 'pcd_{}_{}.pcd').format(demo, iteration), pcd)
        # o3d.visualization.draw_geometries([pcd])

        # Next, do matplotlib visualization, less fine-grained but with coordinates.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #! Axis limits from Daniel's code
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(-0.6, 0.0)
        ax.set_zlim(-0.2, 0.4)

        #! Equal axis limits
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # ax.set_xlim(-1, 1)

        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')

        # Plotting the location of the end-effector here as well
        ax.scatter(ldp[0], ldp[1], ldp[2], color="red", label='ee loc', s=10)
        # ax.scatter(pcl[pts_tool, :][Ellipsis, 0], pcl[pts_tool, :][Ellipsis, 1], pcl[pts_tool, :][Ellipsis, 2], color="orange", label='pc tool', s=2)
        ax.scatter(pcl[pts_targ, :][Ellipsis, 0], pcl[pts_targ, :][Ellipsis, 1], pcl[pts_targ, :][Ellipsis, 2], color="y", label='target', s=5)
        # ax.scatter(pcn[ptn_targ, :][Ellipsis, 0], pcn[ptn_targ, :][Ellipsis, 1], pcn[ptn_targ, :][Ellipsis, 2], color="g", label='synced_target', s=5)
        ax.scatter(new_pcl[dist_points, :][Ellipsis, 0], new_pcl[dist_points, :][Ellipsis, 1], new_pcl[dist_points, :][Ellipsis, 2], color="red", label='distractor', s = 2)
        # ax.scatter(new_pcl[targ_points, :][Ellipsis, 0], new_pcl[targ_points, :][Ellipsis, 1], new_pcl[targ_points, :][Ellipsis, 2], color="green", label='target', s = 2)

        #* The ladel model has 5000 points. From these 5000 points, we sample the same number of points as the tool points in the Kinect observed pointcloud

        # choice = np.random.choice(5000, size=len(pcl[pcl[:, 3] == 2.0]), replace=False)

        # ax.scatter(pcl[pts_tool, :][Ellipsis, 0], pcl[pts_tool, :][Ellipsis, 1], pcl[pts_tool, :][Ellipsis, 2], color="orange", label='og tool')
        # print('pcl_tool_points: {} pcl_np points: {}'.format(len(new_pcl[new_pcl[:, 3] == 2.0]), pcd_np[choice].shape))


        #* We replace the ground truth tool points with the analytically computed ladel points
        # pcl[pts_tool, :3] = pcd_np[choice]

        ax.scatter(new_pcl[tool_points, :][Ellipsis, 0], new_pcl[tool_points, :][Ellipsis, 1], new_pcl[tool_points, :][Ellipsis, 2], color="blue", label='model tool', s = 2)
        # ax.scatter(pcd_np[choice][:, 0], pcd_np[choice][:, 1], pcd_np[choice][:, 2], label='model tool')
        # ax.scatter(ee_pose[0], ee_pose[1], ee_pose[2], label='gripper', color = 'orange', s=50)

        #* Converting the (N, 4) array into a one-hot encoded matrix, with shape (N, 6). col[3] corresponds to target, col[4] is the distractor and col[5] is the tool
        one_hot_encoding = np.eye(3)[new_pcl[:,3].astype(int)]
        #* h-stacking the positions and encodings together

        """We will be following the structure of the overall tool-flow net architecture from now on, rather
        than a different structure specfically for the physical code"""
        if encoding == 'tool':
            """If we're processing data from the 'simple' demonstrator, then we only have a tool moving in some direction"""
            pco = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3]])
        elif encoding == 'targ':
            pco = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3], one_hot_encoding[:, 0:1]])
        elif encoding == 'full':
            pco = np.hstack([new_pcl[:, :3], one_hot_encoding[:, 2:3], one_hot_encoding[:, 0:1], one_hot_encoding[:, 1:2]])

        #* Explanation of the 2 stages of sampling taking place here:
        #* 1. The first sampling taking place above is specifically to reduce the number of points in the scanned tool point cloud from 5000 to the number of tool points that was observed in the scene.
        #* 2. We then replace all the tool points in the pcl array (that corresponds to the observed scene) with the scanned, downsampled ladle points. This sampling is controlled by the 'choice' variable
        #* 3. Below, we create a new np.choice array that controls the number of points to be retained in the ENTIRE pointcloud. Earlier, we had 5000 points being saved, 
        #* 4. I'm pretty sure I'll be revisiting this block of code in case of any inconsistencies so wanted to make a note of this, both for myself and others reading this

        subsampling = np.random.choice(len(pco), size = 1400, replace = False)

        pco = pco[subsampling, Ellipsis]

        #! Changing the index at which the one-hot encoding of the tool is stored
        # print(np.sum(pco[:, 5] == 1), pcd_np[choice].shape)
        np.save(join(dir_name, 'pco_{}_{}.npy').format(demo, iteration), pco)

        plt.legend(loc='best')

        plt.title('Demo: {} Iteration: {}'.format(demo, iteration))

        # plt.savefig('debug.png')

        plt.savefig(join(dir_name, 'fig_{}_{}.png').format(demo, iteration))
        # plt.show()
        plt.close()


def apply_transformation(ladle_cloud, position, rotation):
    rotated_ladle = rotator(ladle_cloud)
    rotation_matrix = np.load(rotation)
    position_matrix = np.load(position)
    transformed_ladle = np.matmul(rotated_ladle, rotation_matrix.transpose()) + position_matrix
    return transformed_ladle

def compose_actions(demo, iteration, k, eep_list, old_ladel):
    #! demo - is the current demo that whose actions we're looking at.
    #! k - is the slice from current iteration ahead that we're looking at
    #! eep_list is the list of the end-effector positions that we'll be slicing

    '''What do we want to compute here?
    We want to compose the transformations together. Lets say we want to increase the magnitude of the deltas returned by the 
    model. We do this by chaining together some "n" number of actions together

    1. Take the current index and the next index. The assumption here is that the current_demo next_demo would've 
    made sure that both actions are from the same demonstration.'''

    # print(eep_list)

    # current_scene = np.load('pco_{}_{}.npy'.format(demo, iteration))
    # current_ladel_points = np.where(current_scene[:, 5] == 1.0)[0]
    # current_ladel = current_scene[current_ladel_points, :][:, :3]

    dir_name = split(eep_list[0])[0]

    current_ladel = old_ladel.copy()

    #* Load the current pose's index
    current_pose_index = eep_list.index(glob(join(dir_name, 'eep_{}_{}.npy'.format(demo, iteration)))[0])
    #* Load load the iteration + k pose's index
    kth_pose_index = eep_list.index(glob(join(dir_name, 'eep_{}_{}.npy'.format(demo, iteration + k)))[0])
    # print('{} and {}'.format(eep_list[current_pose_index], eep_list[kth_pose_index]))
    #* Collect all the intermediate poses
    intermediate_poses = eep_list[current_pose_index: kth_pose_index + 1]

    #* Setting the tool flow to zeros here, based on the shape of the ladle that we're current reading
    total_flow = np.zeros_like(current_ladel)
    
    for idx in range(len(intermediate_poses) - 1):
        current_pose = np.load(intermediate_poses[idx])
        next_pose = np.load(intermediate_poses[idx + 1])

        current_q = quat(current_pose[3], current_pose[4], current_pose[5], current_pose[6])
        next_q = quat(next_pose[3], next_pose[4], next_pose[5], next_pose[6])

        delta_pos = (next_pose[:3] - current_pose[:3])
        flow = np.zeros_like(current_ladel)
        flow += delta_pos

        #* Here we compute the rotations required for the tool
        current_q = current_pose[3:]
        next_q = next_pose[3:]
        #* Changing the ordering of the quaternion assignment here so that the indexes correctly encode for the w component
        current_quat = quat(current_q[0], current_q[1], current_q[2], current_q[3])
        next_quat = quat(next_q[0], next_q[1], next_q[2], next_q[3])
        delta_quat = next_quat * current_quat.inverse

        number_of_tool_points = current_ladel.shape[0]

        delta_quat._normalise()
        dqp = delta_quat.conjugate.q

        relative = current_ladel - current_pose[:3]

        vec_mat = np.zeros((number_of_tool_points, 4, 4), dtype = current_ladel.dtype)

        vec_mat[:, 0, 1] = -relative[:, 0]
        vec_mat[:, 0, 2] = -relative[:, 1]
        vec_mat[:, 0, 3] = -relative[:, 2]

        vec_mat[:, 1, 0] =  relative[:, 0]
        vec_mat[:, 1, 2] = -relative[:, 2]
        vec_mat[:, 1, 3] =  relative[:, 1]

        vec_mat[:, 2, 0] =  relative[:, 1]
        vec_mat[:, 2, 1] =  relative[:, 2]
        vec_mat[:, 2, 3] = -relative[:, 0]

        vec_mat[:, 3, 0] =  relative[:, 2]
        vec_mat[:, 3, 1] = -relative[:, 1]
        vec_mat[:, 3, 2] =  relative[:, 0]

        mid = np.matmul(vec_mat, dqp)
        mid = np.expand_dims(mid, axis = -1)

        relative_rotation = delta_quat._q_matrix() @ mid
        relative_rotation = relative_rotation[:, 1:, 0]

        flow += relative_rotation - relative

        #* Continuously applying the flow to the ladel pointlcloud
        current_ladel += flow

        #* Continuously appending the flow here
        total_flow += flow

    return current_ladel, total_flow

def flow_extractor(pco_files, eep_files, k_steps = 1, scanned_pc='dense_ladel.pcd'):
    #! From Daniel: Improve the variable naming!

    '''what I need to complete here:
    1. Collect PC data from only translations.
    2. Collect EE data from only translations.
    3. Take the delta of the EE poses and apply.
    4. Apply this delta to the point-cloud
    5. Save as (s, a) pairs, where S is the state and a is the resulting flow'''

    dir_name = dirname(pco_files[0])
    pkl_dir_name_save = join('/data/sarthak/data_demo', dir_name)
    print('Saving PKLs here: {}'.format(pkl_dir_name_save))
    if isdir(pkl_dir_name_save) == False:
        makedirs(pkl_dir_name_save)

    from glob import glob

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(scanned_pc)

    pcd  = np.array(pcd.points)

    #! Need like a sampler here that randomly samples some points from the ladle

    #* Recording the occurances of the different demonstrations in the given session
    eep_list = [basename(eep_file).split('.')[0][4:].split('_')[0] for eep_file in eep_files]

    #* Here we count the number of times the demonstration shows up, which in turn is the number of actions in that demonstration
    eep_dict = Counter(eep_list)

    print('Number of actions found for each demonstration: ', eep_dict)

    for demo in eep_dict.keys():
        #* Here we cycle through the demosntration s collected in the given
        iteration = 0
        epis = defaultdict(list)

        while iteration < eep_dict[demo]:
            #* We cycle through the numnber of actions in the given demonstration. Below we load the scene and then the tool points
            current_scene = np.load(glob(join(dir_name, 'pco_{}_{}.npy'.format(demo, iteration)))[0])
            current_tip = np.load(glob(join(dir_name, 'eep_{}_{}.npy'.format(demo, iteration)))[0])
            current_ladel_points = np.where(current_scene[:, 3] == 1.0)[0]
            current_ladel = current_scene[current_ladel_points, :][:, :3]

            #* This factor keeps track of the number of actions we compose
            compose = 1

            #* Creating a placeholder for the flow here
            flow = np.zeros_like(current_ladel)
            while (np.linalg.norm(flow[0, :])) < 0.01 and (iteration + (compose * k_steps) < int(eep_dict[demo])):
                #* We keep composing actions if the flow isn't in the centimeter scale
                _, flow = compose_actions(demo, iteration, (k_steps * compose), eep_files, current_ladel)
                #* Increasing the compose by 1 when 
                compose += 1
            if (np.linalg.norm(flow[0, :])) > 0.01:
                if iteration + (compose * k_steps) == int(eep_dict[demo]):
                    #* In some instances the action that was used last to compose the action to greater than 0.01, was the last index in the list
                    #* If that happens then the next_tip index is len(eep) + 1. If that is the case, then this conditional reduces the index by 1.
                    #* Hack.
                    compose -= 1

                #* This is basically when the flow has become greater
                next_tip = np.load(glob(join(dir_name, 'eep_{}_{}.npy'.format(demo, iteration + (compose * k_steps))))[0])

                #* Calculate the translation here
                delta_pos = (next_tip[:3] - current_tip[:3])

                #* Current orientation
                current_q = current_tip[3:]

                #* Next orientation
                next_q = next_tip[3:]

                #* Convereting both orientations into quaternions
                current_quat = quat(current_q[0], current_q[1], current_q[2], current_q[3])
                next_quat = quat(next_q[0], next_q[1], next_q[2], next_q[3])

                #* Calculating the orientation delta
                delta_quat = next_quat * current_quat.inverse

                #*Stacking the translation and the orientation
                act_raw = np.hstack([delta_pos, delta_quat.axis * delta_quat.angle])

                #* Adding act_raw into the demo dictionary as well
                epis['act_raw'].append(act_raw)

                #* The actual dictionary we need for TFN
                epis['obs'].append((np.load(glob(join(dir_name, 'eep_{}_{}.npy').format(demo, iteration))[0]), cv2.imread(glob(join(dir_name, 'img_{}_{}.png').format(demo, iteration))[0]), 'segmented_rgb_image', np.load(join(dir_name, 'pco_{}_{}.npy').format(demo, iteration)), {'points': current_ladel, 'flow': flow}))
                print('Raised! Flow: {} Demo: {}: Iteration: {} Accumalation: {}'.format(np.linalg.norm(flow[0, :]), demo, iteration, k_steps*compose))
            else:
                #* The else statement if we keep accumalating actions and still don't manage to make the threshold
                print('No! Flow: {} Demo: {}: Iteration: {} Accumalation: {}'.format(np.linalg.norm(flow[0, :]), demo, iteration, k_steps*compose))
            iteration += 1

        #* Only dump the epis to the disc if there's actual demonstrator data in it.
        if len(epis['obs'])>0:
            with open(join(pkl_dir_name_save, 'BC_k_steps_{}_demo_{}.pkl').format(k_steps, demo), 'wb') as dumper:
                pkl.dump(epis, dumper)

def giffer(pc_files, im_files):
    #* Standard, code to glob together all image, pcl and flow images together to GIF them.
    #! NOTE (sarthak): "flow" files here aren't the plotly flow figures but the ones generated by the matplotlib code.
    #! If you're looking for the plotly figures, run plotly_visualization.py

    dir_name = split(pc_files[0])[0]

    print("GIF-ing {} PC together".format(len(pc_files)))
    print("GIF-ing {} images together".format(len(im_files)))

    pc_images = [Image.fromarray(imageio.imread(pc_file)).resize((640, 480)) for pc_file in pc_files]
    im_images = [Image.fromarray(imageio.imread(im_file)).resize((640, 480)) for im_file in im_files]

    comibined_images = []

    for idx in range(len(pc_images)):
        new_image = Image.new('RGB', (2*640, 480))
        new_image.paste(pc_images[idx], (0, 0))
        new_image.paste(im_images[idx], (640, 0))
        comibined_images.append(new_image)

    imageio.mimsave(join(dir_name, 'both.gif'), comibined_images, fps=10)
    imageio.mimsave(join(dir_name, 'pc.gif'), pc_images, fps=10)
    imageio.mimsave(join(dir_name, 'image.gif'), im_images, fps=10)


def pcd_reader(input_pcd='output.pcd'):
    #! NOTE (sarthak): obsolete function to sample and rotate the ladel PCD. I'd like to still keep this here in case we need to refer to some of the older code

    #! (TODO: sarthak) Not sure if we need a cmd process like how Qiao and Brian do it in Zephyr?

    import open3d as o3d
    import numpy as np
    from matplotlib import pyplot as plt

    ee_pose = np.load('eep_0_0.npy')
    pcd = o3d.io.read_point_cloud(input_pcd)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(pcl_color / 255)
    # o3d.visualization.draw_geometries([pcd])

    pcd_np = np.asarray(pcd.points)

    x_mean = np.mean(pcd_np[:,0])
    y_mean = np.mean(pcd_np[:,1])
    z_mean = np.mean(pcd_np[:,2])

    pcd_np[:,0] = (pcd_np[:,0].copy() - x_mean)/100
    pcd_np[:,1] = (pcd_np[:,1].copy() - y_mean)/100
    pcd_np[:,2] = (pcd_np[:,2].copy() - z_mean)/100

    #* Just a simple rotation matrix to make sure that the frames are algined 
    rot_matrix = np.array([[np.cos(90 * np.pi/180), np.sin(90 * np.pi/180), 0], [-np.sin(90 * np.pi/180), np.cos(90 * np.pi/180), 0], [0, 0, 1]])

    pcd_np = np.matmul(pcd_np, rot_matrix)

    pc_to_scene_transformation = np.array([0.089, 0.005	,-0.031, 0.656], [-0.009, 0.093	,-0.011, -0.200], [0.030, 0.013, 0.088, 0.200], [0.000, 0.000,	0.000, 1.000])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(pcl_color / 255)

    o3d.io.write_point_cloud('ladel.pcd', pcd)

    pcd_np = pcd_np + ee_pose[:3]

    np.save('ladel_ee.npy', pcd_np)

    # print(rot_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcd_np[:,0], pcd_np[:,1], pcd_np[:,2])
    ax.set_box_aspect(aspect=(1,1,1))

    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')

    # ax.set_xlim(0.4, 0.8)
    # ax.set_ylim(-0.6, -0.1)
    # ax.set_zlim(-0.2, 0.4)

    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlim(-1, 1)

    plt.savefig('ladel_ee.png')
    plt.close()
    # set_trace()
    # plt.show()
    return pcd

def transform(ladel_cloud = 'output.pcd'):
    import open3d as o3d
    import numpy as np
    from matplotlib import pyplot as plt

    pcd = o3d.io.read_point_cloud(ladel_cloud)

    pcd_np = np.asarray(pcd.points)

    x_mean = np.mean(pcd_np[:,0])
    y_mean = np.mean(pcd_np[:,1])
    z_mean = np.mean(pcd_np[:,2])

    pcd_np[:,0] = (pcd_np[:,0].copy() - x_mean)/100
    pcd_np[:,1] = (pcd_np[:,1].copy() - y_mean)/100
    pcd_np[:,2] = (pcd_np[:,2].copy() - z_mean)/100

    rot_matrix = np.array([[np.cos(90 * np.pi/180), np.sin(90 * np.pi/180), 0], [-np.sin(90 * np.pi/180), np.cos(90 * np.pi/180), 0], [0, 0, 1]])

    pcd_np = np.matmul(pcd_np, rot_matrix)

    pc_to_scene_transformation = np.array([[0.089, 0.005	,-0.031, 0.656], [-0.009, 0.093	,-0.011, -0.200], [0.030, 0.013, 0.088, 0.200], [0.000, 0.000,	0.000, 1.000]])

    add_column  = np.ones((len(pcd_np), 1))

    new_arr = np.hstack([pcd_np, add_column])

    pcd_np = np.matmul(new_arr, pc_to_scene_transformation)

    # print(add_column.shape, pcd_np.shape, new_arr.shape, rot_np.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcd_np[:,0], pcd_np[:,1], pcd_np[:,2])
    ax.set_box_aspect(aspect=(1,1,1))

    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)

    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.set_xlim(-1, 1)

    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')

    # plt.show()

    return 0

def ee_path():
    from glob import glob
    import numpy as np
    eep_files = glob('eep*')
    eep_files.sort(key = lambda x: (int(x.split('.')[0][4:].split('_')[0]), int(x.split('.')[0][4:].split('_')[1])))
    poses = np.array([np.load(eep_file) for eep_file in eep_files])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(poses[:,0], poses[:,1], poses[:,2], color = 'r', label ='poses')
    plt.legend(loc='best')
    plt.show()
    return 0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--k_steps', type=int, default = 1)
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--encoding', type=str, required=True)
    args = parser.parse_args()

    for dir_incomplete in listdir(args.folder_name):
        print('Accessing session: {}'.format(dir_incomplete))
        dir_name = join(args.folder_name, dir_incomplete)
        # dir_name = args.dir_name

        pc_files = glob(join(dir_name, 'pcl_*'))

        #* This weird looking lambda function first sorts the demos, and then the iterations from those demos
        pc_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        pcer(pc_files, encoding = args.encoding)

        #* These are the one-hot encoded pointclouds
        pco_files = glob(join(dir_name, 'pco_*'))
        #* These are the end-effector pose from robot.get_pose()
        eep_files = glob(join(dir_name, 'eep_*'))
        #* These are the end-effector translation that we query from tf. Hopefully they have the same value as the [:3] of the corresponding eep files above [Check 4.2 "tf_echo" section here: http://wiki.ros.org/tf/Tutorials/Introduction%20to%20tf]
        ldp_files = glob(join(dir_name, 'ldp_*'))
        #* These contain the rotation matrix to transform the ladle from the origin to the end-effector frame. Also queried from tf
        ldr_files = glob(join(dir_name, 'ldr_*'))

        pco_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        eep_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        ldp_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        ldr_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))

        flow_extractor(pco_files, eep_files, k_steps = args.k_steps)

        #* Globbing all the images to generate GIFs
        pc_files = glob(join(dir_name, 'fig*'))
        im_files = glob(join(dir_name, 'img*'))

        pc_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        #! NOTE (sarthak): the below step is very important and might have you guessing what it is. We're basically stripping the pointcloud observations to just their
        #! demo and iterations in the format of demo_iteration. We then keep only those RGB images for which the corresponding demo_iteration pcl files are available. This
        #! does not cause an issue in most cases, but sometimes when exiting the robot.py code, extra frames might be captured, which might mess up the syncing of the GIFs, flow
        #! computation etc.
        pc_demo_iteration_list = [split(pc)[1].split('.png')[0][4:] for pc in pc_files]
        im_files.sort(key = lambda x: (int(split(x)[1].split('.')[0][4:].split('_')[0]), int(split(x)[1].split('.')[0][4:].split('_')[1])))
        im_files = [im for im in im_files if split(im)[1].split('.png')[0][4:]]

        assert len(pc_files) == len(im_files), 'Check if im and pc\'s are correctly loaded'

        if args.slice == True:
            '''Limitting the number frames used to generate the
            GIFs so that we can get the results faster'''
            slice_factor = 120
            print('WARNING! You are slicing the number of frames in the GIF to: {} {}'.format(slice_factor, args.slice))
        else:
            slice_factor = len(im_files)

        if args.gif:
            giffer(pc_files[:slice_factor], im_files[:slice_factor])

if __name__ == '__main__':
    main()