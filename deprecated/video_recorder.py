#!/usr/bin/env python
import sys
import os
from roll_utils import create_goal_overlay
import rospy
from sensor_msgs.msg import Image
import cv2
import time
import thread
import numpy as np
import ros_numpy
from cv_bridge import CvBridge
import os
import ffmpeg
# I don't think Carl is using.
#from sawyer_control.srv import *


class VideoRecorder(object):
    def __init__(self, dir, buffer, goal_points=None):
        self.bridge = CvBridge()
        self.buffer = buffer
        self.filenames = []
        self.dir = os.path.join(dir, 'frames')
        self.i = 0
        self.goal_points = goal_points


    def store_latest_image(self, data):
        img = self.bridge.imgmsg_to_cv2(data).copy()[:,:,:3]
        if self.goal_points is not None:
            overlay_img = create_goal_overlay(self.buffer, self.goal_points, img.copy(), 'k4a')
            img = 0.3 * img[:,:,:] + 0.7 * overlay_img[:,:,:3]
        self.filenames.append(os.path.join(self.dir, 'frame%04d'%self.i+'.jpg'))
        cv2.imwrite(self.filenames[-1], img)
        self.i += 1
        # rospy.loginfo("I have updated the lasest image")

    def start_recording(self, topic):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.image_sub = rospy.Subscriber(topic, Image, self.store_latest_image)
        def spin_thread():
            rospy.spin()
        thread.start_new(spin_thread, ())

    def stop_recording(self):
        self.image_sub.unregister()

    def get_video(self):
        stream = ffmpeg.input(os.path.join(self.dir, 'frame%04d.jpg'))
        stream = ffmpeg.output(stream, os.path.join(os.path.split(self.dir)[0], 'output.mp4'))
        ffmpeg.run(stream)


if __name__ == "__main__":
    vr = VideoRecorder('./')
    # vr.start_recording('k4a/rgb/image_rect_color')
    # time.sleep(10)
    # vr.stop_recording()
    vr.get_video()
