#! /usr/bin/env python

import rospy
import math
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
import sys
from scipy.spatial.transform import Rotation as R
import numpy as np

# 
def init_scene(scene):
    # listener = tf.TransformListener()

    # while not rospy.is_shutdown():
    #     try:
    #         timestamp = listener.getLatestCommonTime('/base', '/right_hand')
    #         break
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         continue
    # Creating Collision Objects for Motion Planning
    scene.remove_world_object('table')
    scene.remove_world_object('beam_1')
    scene.remove_world_object('beam_2')
    scene.remove_world_object('beam_3')
    scene.remove_world_object('beam_4')
    scene.remove_world_object('beam_5')
    scene.remove_world_object('beam_6')
    scene.remove_world_object('fridge')
    scene.remove_world_object('beam_7')
    rospy.sleep(2)
    timestamp = rospy.get_rostime()

    table_size = [1.1, 1.1, 0.01]
    table_pose = PoseStamped()
    table_pose.header.frame_id = '/base'
    table_pose.header.stamp = timestamp
    table_pose.pose.orientation.w = 1
    table_pose.pose.position.y = -0.6
    table_pose.pose.position.x = -0.07
    table_pose.pose.position.z = -0.015

    beam1_pose = PoseStamped()
    beam1_size = [0.08, 0.08, 0.29]
    beam1_pose.header.frame_id = '/base'
    beam1_pose.header.stamp = timestamp
    beam1_pose.pose.orientation.w = 1
    beam1_pose.pose.position.x = 0.42
    beam1_pose.pose.position.y = -0.16
    beam1_pose.pose.position.z = 0.08

    cam_pose = PoseStamped()
    cam_size = [0.12, 0.08, 0.59]
    cam_pose.header.frame_id = '/base'
    cam_pose.header.stamp = timestamp
    cam_pose.pose.orientation.w = 1
    cam_pose.pose.position.x = -0.35
    cam_pose.pose.position.y = -0.04
    cam_pose.pose.position.z = 0.08


    beam2_pose = PoseStamped()
    beam2_size = [0.05, 0.05, 1.1]
    beam2_pose.header.frame_id = '/base'
    beam2_pose.header.stamp = timestamp
    beam2_pose.pose.orientation.w = 1
    beam2_pose.pose.position.x = -0.4
    beam2_pose.pose.position.y = -0.16
    beam2_pose.pose.position.z = 0.55

    beam3_pose = PoseStamped()
    beam3_size = [ 0.05, 0.85, 0.05]
    beam3_pose.header.frame_id = '/base'
    beam3_pose.header.stamp = timestamp
    beam3_pose.pose.orientation.w = 1
    beam3_pose.pose.position.x = -0.4
    beam3_pose.pose.position.y = -0.26
    beam3_pose.pose.position.z = 0.75

    beam4_pose = PoseStamped()
    beam4_size = [0.3, 0.1,  0.05]
    beam4_pose.header.frame_id = '/base'
    beam4_pose.header.stamp = timestamp
    beam4_pose.pose.orientation.w = 1
    beam4_pose.pose.position.x = -0.25
    beam4_pose.pose.position.y = 0.14
    beam4_pose.pose.position.z = 0.75

    beam5_pose = PoseStamped()
    beam5_size = [0.3, 0.1,  0.05]
    beam5_pose.header.frame_id = '/base'
    beam5_pose.header.stamp = timestamp
    beam5_pose.pose.orientation.w = 1
    beam5_pose.pose.position.x = 0.57
    beam5_pose.pose.position.y = -0.6
    beam5_pose.pose.position.z = -0.10

    beam6_pose = PoseStamped()
    beam6_size = [ 0.25, 0.40,0.8]
    beam6_pose.header.frame_id = '/base'
    beam6_pose.header.stamp = timestamp
    beam6_pose.pose.orientation.w = 1
    beam6_pose.pose.position.x = 0.55
    beam6_pose.pose.position.y = -0.6
    beam6_pose.pose.position.z = 0.3

    beam7_pose = PoseStamped()
    beam7_size = [0.35,0.2,  0.12]
    beam7_pose.header.frame_id = '/base'
    beam7_pose.header.stamp = timestamp
    beam7_pose.pose.orientation.w = 1
    beam7_pose.pose.position.x = 0.60
    beam7_pose.pose.position.y = -0.6
    beam7_pose.pose.position.z = 0.65

    # fridge_pose = PoseStamped()
    # fridge_size = [ 0.25, 0.40,0.8]
    # fridge_pose.header.frame_id = '/base'
    # fridge_pose.header.stamp = timestamp
    # fridge_pose.pose.orientation.w = 1
    # fridge_pose.pose.position.x = -0.3
    # fridge_pose.pose.position.y = -0.7
    # fridge_pose.pose.position.z = 0.3

    rospy.sleep(0.5)
    scene.add_box('table_new', table_pose, table_size)
    scene.add_box('beam_1', beam1_pose, beam1_size)
    scene.add_box('beam_2', beam2_pose, beam2_size)
    scene.add_box('beam_3', beam3_pose, beam3_size)
    scene.add_box('beam_4', beam4_pose, beam4_size)
    scene.add_box('beam_5', beam5_pose, beam5_size)
    scene.add_box('beam_6', beam6_pose, beam6_size)
    scene.add_box('beam_7', beam7_pose, beam7_size)
    scene.add_box('cam', cam_pose, cam_size)
    # scene.add_box('fridge', fridge_pose, fridge_size)


def main():
    joint_state_topic = ['joint_states:=/robot/joint_states']
    moveit_commander.roscpp_initialize(joint_state_topic)
    rospy.init_node('init_scene')

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    init_scene(scene)

    moveit_commander.roscpp_shutdown()
    rospy.signal_shutdown("we don't want to continue")
if __name__ == '__main__':
    main()