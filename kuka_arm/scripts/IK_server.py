#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
from __future__ import print_function
import rospy
import tf
import numpy as np
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


# KUKA 210
class Kuka210(object):

    def __init__(self):
        ### Create symbols for joint variables

        self.q = dict(('q%d' % k, symbols('q%d' % k)) for k in range(1, 8))
        # d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        self.d = dict(('d%d' % k, symbols('d%d' % k)) for k in range(1, 8))
        # a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        self.a = dict(('a%d' % k, symbols('a%d' % k)) for k in range(7))
        # alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
        self.alpha = dict(('alpha%d' % k, symbols('alpha%d' % k)) for k in range(7))
        # rpy
        self.r = symbols('r')
        self.p = symbols('p')
        self.y = symbols('y')

        self.s = {  # KUKA KR210 DH Parameters
            self.alpha['alpha0']: 0, self.a['a0']: 0, self.d['d1']: 0.75,
            self.alpha['alpha1']: -pi / 2, self.a['a1']: 0.35, self.d['d2']: 0, self.q['q2']: self.q['q2'] - pi / 2,
            self.alpha['alpha2']: 0, self.a['a2']: 1.25, self.d['d3']: 0,
            self.alpha['alpha3']: -pi / 2, self.a['a3']: -0.054, self.d['d4']: 1.50,
            self.alpha['alpha4']: pi / 2, self.a['a4']: 0, self.d['d5']: 0,
            self.alpha['alpha5']: -pi / 2, self.a['a5']: 0, self.d['d6']: 0,
            self.alpha['alpha6']: 0, self.a['a6']: 0, self.d['d7']: 0.303, self.q['q7']: 0
        }

        # kr210.urdf.xacro
        self.limit = {'theta1': {'lower': radians(-185), 'upper': radians(185)},
                      'theta2': {'lower': radians(-45), 'upper': radians(85)},
                      'theta3': {'lower': radians(-210), 'upper': radians(155 - 90)},
                      'theta4': {'lower': radians(-350), 'upper': radians(350)},
                      'theta5': {'lower': radians(-125), 'upper': radians(125)},
                      'theta6': {'lower': radians(-350), 'upper': radians(350)}}

        # Homogeneous transformation
        self.T0_1 = self.T(self.alpha['alpha0'], self.a['a0'], self.q['q1'], self.d['d1'])
        self.T0_1 = self.T0_1.subs(self.s)

        self.T1_2 = self.T(self.alpha['alpha1'], self.a['a1'], self.q['q2'], self.d['d2'])
        self.T1_2 = self.T1_2.subs(self.s)

        self.T2_3 = self.T(self.alpha['alpha2'], self.a['a2'], self.q['q3'], self.d['d3'])
        self.T2_3 = self.T2_3.subs(self.s)

        self.T3_4 = self.T(self.alpha['alpha3'], self.a['a3'], self.q['q4'], self.d['d4'])
        self.T3_4 = self.T3_4.subs(self.s)

        self.T4_5 = self.T(self.alpha['alpha4'], self.a['a4'], self.q['q5'], self.d['d5'])
        self.T4_5 = self.T4_5.subs(self.s)

        self.T5_6 = self.T(self.alpha['alpha5'], self.a['a5'], self.q['q6'], self.d['d6'])
        self.T5_6 = self.T5_6.subs(self.s)

        self.T6_G = self.T(self.alpha['alpha6'], self.a['a6'], self.q['q7'], self.d['d7'])
        self.T6_G = self.T6_G.subs(self.s)

        self.Rot_x = self.R_x(self.r)
        self.Rot_y = self.R_y(self.p)
        self.Rot_z = self.R_z(self.y)

        self.R_corr = self.Rot_z.subs(self.y, np.pi) * self.Rot_y.subs(self.p, -np.pi / 2)

        self.T0_EE = self.T0_1 * self.T1_2 * self.T2_3 * self.T3_4 * self.T4_5 * self.T5_6 * self.T6_G * self.R_corr

        self.Rot_EE = self.Rot_z * self.Rot_y * self.Rot_x * self.R_corr

    def T(self, alpha, a, q, d):
        return Matrix([[cos(q), -sin(q), 0, a],
                       [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                       [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), cos(alpha) * d],
                       [0, 0, 0, 1]])

    def R_z(self, alpha):
        return Matrix([[cos(alpha), -sin(alpha), 0, 0],
                       [sin(alpha), cos(alpha), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    def R_y(self, beta):
        return Matrix([[cos(beta), 0, sin(beta), 0],
                       [0, 1, 0, 0],
                       [-sin(beta), 0, cos(beta), 0],
                       [0, 0, 0, 1]])

    def R_x(self, gamma):
        return Matrix([[1, 0, 0, 0],
                       [0, cos(gamma), -sin(gamma), 0],
                       [0, sin(gamma), cos(gamma), 0],
                       [0, 0, 0, 1]])

    def ik(self, req, x):
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([req.poses[x].orientation.x,
                                                                       req.poses[x].orientation.y,
                                                                       req.poses[x].orientation.z,
                                                                       req.poses[x].orientation.w])

        Rot_rpy = self.Rot_EE.subs({self.r: roll, self.p: pitch, self.y: yaw})
        Rot_rpy = Rot_rpy[0:3, 0:3]

        px = req.poses[x].position.x
        py = req.poses[x].position.y
        pz = req.poses[x].position.z

        EE = Matrix([[px],
                     [py],
                     [pz]])

        WC = EE - 0.303 * Rot_rpy[:, 2]  # (3, 1) vector

        # Calculate theta1 for joint1
        theta1 = atan2(WC[1], WC[0])

        # For joint2, joint3, and joint4
        len_x = sqrt(WC[0] ** 2 + WC[1] ** 2) - 0.35
        len_y = WC[2] - 0.75

        side_a = 1.501
        side_b = sqrt(len_x ** 2 + len_y ** 2)
        side_c = 1.25

        angle_a = acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2. * side_b * side_c))
        angle_b = acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2. * side_a * side_c))
        # angle_c = acos((side_a**2 + side_b**2 - side_c**2) / (2. * side_a * side_b))

        theta2 = pi / 2 - angle_a - atan2(len_y, len_x)
        theta3 = pi / 2 - angle_b - 0.036  # ???

        R0_3 = self.T0_1[0:3, 0:3] * self.T1_2[0:3, 0:3] * self.T2_3[0:3, 0:3]
        R0_3 = R0_3.evalf(subs={self.q['q1']: theta1, self.q['q2']: theta2, self.q['q3']: theta3})
        # R3_6 = R0_3.inv("LU") * Rot_EE_sub[0:3, 0:3]
        R3_6 = R0_3.transpose() * Rot_rpy

        # For joint4, joint5, and joint6
        theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
        theta5 = atan2(sqrt(R3_6[0, 2] ** 2 + R3_6[2, 2] ** 2), R3_6[1, 2])
        theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])

        theta1 = theta1.evalf()
        theta2 = theta2.evalf()
        theta3 = theta3.evalf()
        theta4 = theta4.evalf()
        theta5 = theta5.evalf()
        theta6 = theta6.evalf()

        theta1 = np.clip(theta1, self.limit['theta1']['lower'], self.limit['theta1']['upper'])
        theta2 = np.clip(theta2, self.limit['theta2']['lower'], self.limit['theta2']['upper'])
        theta3 = np.clip(theta3, self.limit['theta3']['lower'], self.limit['theta3']['upper'])
        theta4 = np.clip(theta4, self.limit['theta4']['lower'], self.limit['theta4']['upper'])
        theta5 = np.clip(theta5, self.limit['theta5']['lower'], self.limit['theta5']['upper'])
        theta6 = np.clip(theta6, self.limit['theta6']['lower'], self.limit['theta6']['upper'])

        #print('EE: {} \nWC: {} \ntheta1: {}\ntheta2: {} \ntheta3: {} \ntheta4: {} \ntheta5: {} \ntheta6: {}'.
        #      format(EE, WC, theta1, theta2, theta3, theta4, theta5, theta6))

        FK = self.T0_EE.evalf(subs={self.q['q1']: theta1,
                                    self.q['q2']: theta2,
                                    self.q['q3']: theta3,
                                    self.q['q4']: theta4,
                                    self.q['q5']: theta5,
                                    self.q['q6']: theta6})
        # FK_rot = FK[0:3, 0:3]
        # FK_x = FK[:, 3]

        return theta1, theta2, theta3, theta4, theta5, theta6  # , FK, WC, EE


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print("No valid poses received")
        return -1
    else:

        ### Your FK code here
        # Create symbols
        #
        #
        # Create Modified DH parameters
        #
        #
        # Define Modified DH Transformation matrix
        #
        #
        # Create individual transformation matrices
        #
        #
        # Extract rotation matrices from the transformation matrices
        #
        #
        ###

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            # px = req.poses[x].position.x
            # py = req.poses[x].position.y
            # pz = req.poses[x].position.z

            # (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            #    [req.poses[x].orientation.x, req.poses[x].orientation.y,
            #     req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
            # Compensate for rotation discrepancy between DH parameters and Gazebo
            #
            #
            # Calculate joint angles using Geometric IK method
            #
            #
            ###
            theta1, theta2, theta3, theta4, theta5, theta6 = kuka210.ik(req, x)

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print("Ready to receive an IK request")
    rospy.spin()


if __name__ == "__main__":
    # KUKA 210 object
    kuka210 = Kuka210()
    IK_server()
