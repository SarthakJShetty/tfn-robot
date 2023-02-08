#! /usr/bin/env python  
import rospy
import tf
from tf.transformations import quaternion_from_euler

if __name__ == '__main__':
    rospy.init_node('fixed_tf_broadcaster')
    br = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        try:
            latesttime = listener.getLatestCommonTime('/base', '/right_hand')
            br.sendTransform((0.0, 0.0, 0.12),
                            quaternion_from_euler(0,0,0),
                            latesttime,
                            "/right_vacuum_gripper_tip_hack",
                            "right_hand",)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()