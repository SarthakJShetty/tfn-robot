# mixed-media-physical

Relevant code for physical mixed media experiments. This is a catkin package.

We also need potentially modified versions of the following packages:

- [Sawyer Robot](https://github.com/DanielTakeshi/sawyer_robot)
- [Sawyer MoveIt](https://github.com/DanielTakeshi/sawyer_moveit)

## Installation and Setup

### Python 2.7

We use Python 2.7 (sorry) for ROS and dealing with the physical robot. The setup
should look something like this in your "catkin" directory, which is the place
that has all the ROS packages (you can tell if they have the `package.xml` in
them).

```
seita@twofish:~/catkin_ws $ ls -lh src/
total 40K
drwxrwxr-x 10 seita seita 4.0K Feb 23 11:58 Azure_Kinect_ROS_Driver
lrwxrwxrwx  1 seita seita   50 Feb 23 11:03 CMakeLists.txt -> /opt/ros/kinetic/share/catkin/cmake/toplevel.cmake
drwxrwxr-x  7 seita seita 4.0K Feb 23 12:16 easy_handeye
drwxrwxr-x  6 seita seita 4.0K Feb 26 13:43 flowbot
drwxrwxr-x  7 seita seita 4.0K Feb 23 11:04 intera_common
drwxrwxr-x  6 seita seita 4.0K Feb 23 11:04 intera_sdk
drwxrwxr-x  8 seita seita 4.0K Mar 11 16:33 mixed-media-physical
drwxrwxr-x 11 seita seita 4.0K Feb 23 12:22 realtime_urdf_filter
drwxrwxr-x  5 seita seita 4.0K Feb 27 15:30 sawyer_moveit
drwxrwxr-x  5 seita seita 4.0K Feb 27 15:30 sawyer_robot
drwxrwxr-x  9 seita seita 4.0K Feb 23 11:19 sawyer_utils
```

**Warning**: I made changes to `sawyer_moveit` and `sawyer_robot` based on changes from 
Ben, Harry, and Carl. You should git clone the linked directories above, or (this might
be simpler) just copy the directories from my `catkin_ws/src` folder to yours. In general,
if there are any discrepancies, please make sure the files are exactly the same among
our catkin workspace directories. If you make changes, you need to run `catkin_make` to
"recompile" things. (Do that command in `~/catkin_ws/`).

**In addition** there may be extra stuff to install using Python 2.7 and (an ancient) pip.
In particular, you need to do:

```
pip install --user scipy==1.2.3
```

### Python 3.6

We also use Python 3.6 for any code that doesn't have to do with ROS.  We might
use this for data analysis, or for querying a policy. To get started:

```
conda create --name mm python=3.6 -y
conda activate mm
pip install opencv-python
pip install imageio
pip install moviepy
pip install open3d-python==0.6.0
```

We use this code for:

- `python analysis.py` script to make it easier to generate GIFs from avi files.
- `python view.py` to visualize saved point clouds generated from Python2 code.

For anything else, assume we use Python 2.7.


## Moving the Sawyer Through Python

First, connect to the robot, for example with Thing 1. I use the following alias
to make things easier:

```
alias set_th1="source /opt/ros/kinetic/setup.bash  &&  source ~/catkin_ws/devel/setup.bash  &&  cd ~/catkin_ws  &&  ./thing1.sh"
```

Then run the following launch file, which gets rviz and (more importantly) the
cameras set up:

```
roslaunch mixed_media mixed_media.launch
```

Then run something like:

```
python robot.py
```

for various tests. See the arguments in `robot.py` for parmeters, etc. 
I recommend testing (a) image crops first, and (b) moving the EE to the workspace bounds.

For Mixed Media, the main files to know are `robot.py`, `utils_robot.py`, and
`data_collector.py`. Other files are copied from other projects or are just
needed to make this a "catkin" package.

The `filter_points.py` is used to obtain point clouds. It is called once we start
the `roslaunch` command.

For testing segmentation, please run `python segmentor.py` using Python 2.7 code.
That will bring up a GUI where you can drag and test different HSV values.
