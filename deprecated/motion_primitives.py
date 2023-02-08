"""
Based on Ben/Harry's `motion_primitives.py` script and redone for MM.
"""
import time
import click
from numpy import diff
import geometry_msgs.msg
import moveit_commander
import numpy as np
# from mp_flow_pred import calc_rot
import rospy
import tf
from geometry_msgs.msg import Point, PoseStamped
from intera_core_msgs.msg import DigitalOutputCommand
from intera_interface import Limb
from scipy.spatial.transform import Rotation as R


class FlowbotController:
    """TODO
    """

    def __init__(self):
        # MoveIt! Stuff
        self.robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        # init_scene(scene)
        self.group = moveit_commander.MoveGroupCommander("right_arm")

        # Vacuum stuff.
        self.publishVacuumCommand = rospy.Publisher(
            "/robot/digital_io/command", DigitalOutputCommand, queue_size=30
        )

        # Intera SDK stuff.
        self.limb = Limb()
        self.joint_names = self.limb.joint_names()

        self.group.set_max_velocity_scaling_factor(0.6)
        self.group.set_max_acceleration_scaling_factor(0.6)
        self.group.set_planner_id("RRTstarkConfigDefault")
        self.group.set_planning_time(1.0)

        # For experiments. TODO(daniel) how to get EE positions? Might be easier.
        # This is what Ben and Harry used.
        self.home_jas = [-1.90256, -2.57107, 0.92954, 2.22587, -2.37902, -1.5741, 3.81140]

        # NOTE(daniel): just set this by manually adjusting the robot and querying j-angles.
        # But, for some reason MoveIt! keeps telling me abotu out of bounds joints.
        #self.home_jas = [-1.54608, -0.65492, 1.7265, -0.70596, 1.03651, -1.9498, 4.71302]

    def moveit_to_pose(self, t, r):
        pose_target = geometry_msgs.msg.Pose()
        pose_target.position.x = t[0]
        pose_target.position.y = t[1]
        pose_target.position.z = t[2]
        pose_target.orientation.x = r[0]
        pose_target.orientation.y = r[1]
        pose_target.orientation.z = r[2]
        pose_target.orientation.w = r[3]

        # Planning
        self.group.set_start_state(self.robot.get_current_state())
        self.group.set_pose_target(pose_target)
        plan = self.group.plan()
        pts = plan.joint_trajectory.points
        if not pts:
            for _ in range(10):
                plan = self.group.plan()
                pts = plan.joint_trajectory.points
                if pts:
                    break

        self.group.execute(plan, wait=True)
        self.group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.group.clear_pose_targets()

    def go_to_experiment_home(self):
        """NOTE(daniel): Ben/Harry use this at the start of each trial."""
        self.group.set_start_state(self.robot.get_current_state())
        self.group.set_joint_value_target(self.home_jas)

        # NOTE(daniel) be careful! Maybe always have some kind of query / check.
        import pdb; pdb.set_trace()  # pause then continue
        plan = self.group.plan()
        self.group.execute(plan, wait=True)
        self.group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.group.clear_pose_targets()

    def move_until_contact(self, vector, speed, timeout, contact_force=15.0):
        # Get the force exerted on the endpoint.
        f = self.limb.endpoint_effort()["force"]
        fnorm = np.linalg.norm(f)

        t0 = rospy.get_time()
        r = rospy.Rate(50)

        # The vector is just a direction, so we set a particular speed.
        vector = vector / np.linalg.norm(vector) * speed

        while fnorm < contact_force:
            # Compute the estimated joint velocities.
            x_des = np.array([vector[0], vector[1], vector[2], 0, 0, 0])
            curr_ja = self.group.get_current_joint_values()
            jacobian = np.array(self.group.get_jacobian_matrix(curr_ja))
            th_ex = np.matmul(np.linalg.pinv(jacobian), x_des)

            # Give the command.
            cmd = {name: th for name, th in zip(self.joint_names, th_ex)}
            self.limb.set_joint_velocities(cmd)

            # Sleep for a bit.
            r.sleep()

            # Compute the force exerted on the end-effector.
            f = self.limb.endpoint_effort()["force"]
            f = np.array([f.x, f.y, f.z])
            fnorm = np.linalg.norm(f)

            if rospy.get_time() - t0 > timeout:
                rospy.loginfo("timeout occurred before contact")
                return

        rospy.loginfo("detected contact!")
        return

    def move_with_compliance(self, vector, speed, dist, timeout):

        init_xyz = self.limb.endpoint_pose()["position"]
        init_xyz = np.array([init_xyz.x, init_xyz.y, init_xyz.z])

        curr_dist = 0.0

        t0 = rospy.get_time()
        r = rospy.Rate(50)

        # TODO: implement compliance lol.

        # The vector is just a direction, so we set a particular speed.
        vector = vector / np.linalg.norm(vector) * speed

        while curr_dist < dist:
            # Compute the estimated joint velocities.
            x_des = np.array([vector[0], vector[1], vector[2], 0, 0, 0])
            curr_ja = self.group.get_current_joint_values()
            jacobian = np.array(self.group.get_jacobian_matrix(curr_ja))
            th_ex = np.matmul(np.linalg.pinv(jacobian), x_des)

            # Give the command.
            cmd = {name: th for name, th in zip(self.joint_names, th_ex)}
            self.limb.set_joint_velocities(cmd)

            # Sleep for a bit.
            r.sleep()

            # Compute the force exerted on the end-effector.
            curr_xyz = self.limb.endpoint_pose()["position"]
            curr_xyz = np.array([curr_xyz.x, curr_xyz.y, curr_xyz.z])
            curr_dist = np.linalg.norm(curr_xyz - init_xyz)

            if rospy.get_time() - t0 > timeout:
                rospy.loginfo("timeout occurred before distance reached")
                return

        rospy.loginfo("distance moved")
        return

    def rot_align(self, v1, v2):
        # Aligns v1 to v2
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.dot(v1, v2))
        axis = np.cross(v1, v2)
        axis = axis/np.linalg.norm(axis)
        qx = axis[0] * np.sin(angle/2)
        qy = axis[1] * np.sin(angle/2)
        qz = axis[2] * np.sin(angle/2)
        qw = np.cos(angle/2)
        quat = np.array([qx, qy, qz, qw])
        quat = quat / np.linalg.norm(quat)
    
        return quat

    def move_with_flow(self, vector, hand_axis, tip_rot, speed, dist, timeout):

        init_xyz = self.limb.endpoint_pose()["position"]
        init_xyz = np.array([init_xyz.x, init_xyz.y, init_xyz.z])

        init_quat = self.limb.endpoint_pose()["orientation"]
        init_quat = np.array([init_quat.x, init_quat.y, init_quat.z, init_quat.w])
        # Calculate goal rotation
        hand_rot = R.from_quat(tip_rot).as_dcm()
        diff_quat = self.rot_align(hand_axis, -vector)
        diff_rot = R.from_quat(diff_quat).as_euler('xyz')
        

        curr_dist = 0.0

        t0 = rospy.get_time()
        r = rospy.Rate(50)

        # The vector is just a direction, so we set a particular speed.
        vector = vector / np.linalg.norm(vector) * speed
        rot_speed = 0.8 * diff_rot

        while curr_dist < dist:
            # Compute the estimated joint velocities.
            # x_des = np.array([vector[0], vector[1], vector[2], goal_rpy[0], goal_rpy[1], goal_rpy[2]])
            x_des = np.array([vector[0], vector[1], vector[2], rot_speed[0], rot_speed[1], rot_speed[2]
            ])
            curr_ja = self.group.get_current_joint_values()
            jacobian = np.array(self.group.get_jacobian_matrix(curr_ja))
            th_ex = np.matmul(np.linalg.pinv(jacobian), x_des)

            # Give the command.
            cmd = {name: th for name, th in zip(self.joint_names, th_ex)}
            self.limb.set_joint_velocities(cmd)

            # Sleep for a bit.
            r.sleep()

            # Compute the force exerted on the end-effector.
            curr_xyz = self.limb.endpoint_pose()["position"]
            curr_xyz = np.array([curr_xyz.x, curr_xyz.y, curr_xyz.z])
            curr_dist = np.linalg.norm(curr_xyz - init_xyz)

            if rospy.get_time() - t0 > timeout:
                rospy.loginfo("timeout occurred before distance reached")
                return

        rospy.loginfo("distance moved")
        return

    def move_with_dagger(self, vector, hand_axis, tip_rot, speed, dist, timeout):

        init_xyz = self.limb.endpoint_pose()["position"]
        init_xyz = np.array([init_xyz.x, init_xyz.y, init_xyz.z])
        

        curr_dist = 0.0

        t0 = rospy.get_time()
        r = rospy.Rate(50)

        # The vector is just a direction, so we set a particular speed.
        vector = vector / np.linalg.norm(vector) * speed

        while curr_dist < dist:
            # Compute the estimated joint velocities.
            # x_des = np.array([vector[0], vector[1], vector[2], goal_rpy[0], goal_rpy[1], goal_rpy[2]])
            x_des = np.array([vector[0, 0], vector[0, 1], vector[0, 2], 0, 0, 0
            ])
            curr_ja = self.group.get_current_joint_values()
            jacobian = np.array(self.group.get_jacobian_matrix(curr_ja))
            th_ex = np.matmul(np.linalg.pinv(jacobian), x_des)

            # Give the command.
            cmd = {name: th for name, th in zip(self.joint_names, th_ex)}
            self.limb.set_joint_velocities(cmd)

            # Sleep for a bit.
            r.sleep()

            # Compute the force exerted on the end-effector.
            curr_xyz = self.limb.endpoint_pose()["position"]
            curr_xyz = np.array([curr_xyz.x, curr_xyz.y, curr_xyz.z])
            curr_dist = np.linalg.norm(curr_xyz - init_xyz)

            if rospy.get_time() - t0 > timeout:
                rospy.loginfo("timeout occurred before distance reached")
                return

        rospy.loginfo("distance moved")
        return

    def begin_suction(self):
        cmd_1a = DigitalOutputCommand()
        cmd_1b = DigitalOutputCommand()
        cmd_1a.name = "right_valve_1a"
        cmd_1b.name = "right_valve_1b"
        cmd_1a.value = 0
        cmd_1b.value = 1
        self.publishVacuumCommand.publish(cmd_1a)
        self.publishVacuumCommand.publish(cmd_1b)
        cmd_1a.value = 1
        cmd_1b.value = 0
        self.publishVacuumCommand.publish(cmd_1a)
        self.publishVacuumCommand.publish(cmd_1b)

    def end_suction(self):
        cmd_1a = DigitalOutputCommand()
        cmd_1b = DigitalOutputCommand()
        cmd_1a.name = "right_valve_1a"
        cmd_1b.name = "right_valve_1b"
        cmd_1a.value = 1
        cmd_1b.value = 0
        self.publishVacuumCommand.publish(cmd_1a)
        self.publishVacuumCommand.publish(cmd_1b)
        cmd_1a.value = 0
        cmd_1b.value = 1
        self.publishVacuumCommand.publish(cmd_1a)
        self.publishVacuumCommand.publish(cmd_1b)


@click.group()
def cli():
    pass


@cli.command()
def compliance_demo():
    print("compliance")
    # Get current pose.

    # Compute a velocity.

    # Move in that velocity until a distance has been reached.
    pass


@cli.command()
def contact_demo():
    rospy.init_node("contact_demo")

    robot = FlowbotController()

    print('Created the flowbot controller!')
    robot.move_until_contact([0, 0, -1], 0.001, 10.0)
    print('Done moving...')
    time.sleep(10)
    #robot.begin_suction()

    #time.sleep(1)
    #robot.move_with_compliance([0, 0, 1], 0.03, 0.01, 5.0)
    #time.sleep(1)
    #robot.move_with_compliance([0, 0, 1], 0.03, 0.01, 5.0)
    #time.sleep(1)
    #robot.move_with_compliance([0, 0, 1], 0.03, 0.01, 5.0)

    #robot.end_suction()

    rospy.signal_shutdown("done")
    exit(0)


if __name__ == "__main__":
    cli()
