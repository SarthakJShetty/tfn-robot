import rospy

import moveit_commander

if __name__ == "__main__":
    joint_state_topic = ['joint_states:=/robot/joint_states']
    moveit_commander.roscpp_initialize(joint_state_topic)
    rospy.init_node('init_scene')

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "panda_leftfinger"
    box_pose.pose.orientation.w = 1.0
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))