#! /usr/bin/env python

import rospy
import tf
from tf.transformations import quaternion_from_euler

def main():
    rospy.init_node('hand_aruco_publisher')
    br = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        try:
            latesttime = listener.getLatestCommonTime('/base', '/right_connector_plate_mount')
            br.sendTransform((0.0, 0.0, 0.00),
                        #  quaternion_from_euler(1.5707963,-1.5707963,1.5707963),
                         quaternion_from_euler(1.5707963,0, 0),

                         latesttime,
                         "right_vacuum_gripper_base_aruco",
                         "right_connector_plate_mount",)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()

if __name__ == '__main__':
    main()
