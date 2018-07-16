from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2, acos
from sympy.matrices import Matrix
import numpy as np
import collections
import json
import time
from tf import transformations


class Kuka(object):

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

    def debug(self, joint_state_path):
        while True:
            _ = raw_input("[continue]")

            with open(joint_state_path, 'r') as f:
                joint_state_dict = json.load(f)

            joint_state = collections.namedtuple('JointState', joint_state_dict.keys())(**joint_state_dict)

            print(joint_state)

            T0_P = self.T0_EE.evalf(subs={self.q['q1']: joint_state.joint_1,
                                          self.q['q2']: joint_state.joint_2,
                                          self.q['q3']: joint_state.joint_3,
                                          self.q['q4']: joint_state.joint_4,
                                          self.q['q5']: joint_state.joint_5,
                                          self.q['q6']: joint_state.joint_6})
            print("T0_E = ", T0_P)
            r_P = T0_P.col(-1)
            r_P = np.array(r_P).astype(np.float64)
            print("r_P  = ", r_P)

    def ik(self, req, x):

        (roll, pitch, yaw) = transformations.euler_from_quaternion([req.poses[x].orientation.x,
                                                                    req.poses[x].orientation.y,
                                                                    req.poses[x].orientation.z,
                                                                    req.poses[x].orientation.w])

        print('roll: {} pitch: {} yaw: {}'.format(roll, pitch, yaw))

        Rot_EE_sub = self.Rot_EE.subs({self.r: roll, self.p: pitch, self.y: yaw})

        px = req.poses[x].position.x
        py = req.poses[x].position.y
        pz = req.poses[x].position.z

        EE = Matrix([[px],
                     [py],
                     [pz]])

        WC = EE - 0.303 * Rot_EE_sub[:-1, 2]  # (3, 1) vector

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

        theta2 = np.pi / 2 - angle_a - atan2(len_y, len_x)
        theta3 = np.pi / 2 - angle_b - 0.036  # ???

        R0_3 = self.T0_1[0:3, 0:3] * self.T1_2[0:3, 0:3] * self.T2_3[0:3, 0:3]
        R0_3 = R0_3.evalf(subs={self.q['q1']: theta1, self.q['q2']: theta2, self.q['q3']: theta3})
        R3_6 = R0_3.inv("LU") * Rot_EE_sub[0:3, 0:3]

        # For joint4, joint5, and joint6
        theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
        theta5 = atan2(sqrt(R3_6[0, 2] ** 2 + R3_6[2, 2] ** 2), R3_6[1, 2])
        theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])

        print('EE: {} \nWC: {} \ntheta1: {}\ntheta2: {} \ntheta3: {} \ntheta4: {} \ntheta5: {} \ntheta6: {}'.
              format(EE, WC, theta1, theta2, theta3, theta4, theta5, theta6))

        FK = self.T0_EE.evalf(subs={self.q['q1']: theta1,
                                    self.q['q2']: theta2,
                                    self.q['q3']: theta3,
                                    self.q['q4']: theta4,
                                    self.q['q5']: theta5,
                                    self.q['q6']: theta6})
        #FK_rot = FK[0:3, 0:3]
        #FK_x = FK[:, 3]

        return theta1, theta2, theta3, theta4, theta5, theta6, FK, WC, EE

test_cases = {1: [[[2.16135, -1.42635, 1.55109],
                   [0.708611, 0.186356, -0.157931, 0.661967]],
                  [1.89451, -1.44302, 1.69366],
                  [-0.65, 0.45, -0.36, 0.95, 0.79, 0.49]],
              2: [[[-0.56754, 0.93663, 3.0038],
                   [0.62073, 0.48318, 0.38759, 0.480629]],
                  [-0.638, 0.64198, 2.9988],
                  [-0.79, -0.11, -2.33, 1.94, 1.14, -3.68]],
              3: [[[-1.3863, 0.02074, 0.90986],
                   [0.01735, -0.2179, 0.9025, 0.371016]],
                  [-1.1669, -0.17989, 0.85137],
                  [-2.99, -0.12, 0.94, 4.06, 1.29, -4.12]],
              4: [],
              5: []}


def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0

    class Position:
        def __init__(self, EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]

    class Orientation:
        def __init__(self, EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position, orientation)

    class Pose:
        def __init__(self, comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time.time()

    ########################################################################################
    ##

    ## Insert IK code here!
    kuka = Kuka()
    theta1, theta2, theta3, theta4, theta5, theta6, FK, WC, EE = kuka.ik(req, x)

    FK_rot = FK[0:3, 0:3]
    FK_x = FK[:,3]

    print('EE_ground_truth:')
    print('[WC location]: {}'.format(test_case[1]))
    print('[joint angles]: {}'.format(test_case[2]))
    print('FK_rot: {}'.format(FK_rot))
    print('FK_x: {}'.format(FK_x))

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = [WC[0], WC[1], WC[2]]  # <--- Load your calculated WC values in this array
    your_ee = [EE[0], EE[1], EE[2]]  # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time.time() - start_time))

    # Find WC error
    if not (sum(your_wc) == 3):
        wc_x_e = abs(your_wc[0] - test_case[1][0])
        wc_y_e = abs(your_wc[1] - test_case[1][1])
        wc_z_e = abs(your_wc[2] - test_case[1][2])
        wc_offset = sqrt(wc_x_e ** 2 + wc_y_e ** 2 + wc_z_e ** 2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1 - test_case[2][0])
    t_2_e = abs(theta2 - test_case[2][1])
    t_3_e = abs(theta3 - test_case[2][2])
    t_4_e = abs(theta4 - test_case[2][3])
    t_5_e = abs(theta5 - test_case[2][4])
    t_6_e = abs(theta6 - test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
               \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
               \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not (sum(your_ee) == 3):
        ee_x_e = abs(your_ee[0] - test_case[0][0][0])
        ee_y_e = abs(your_ee[1] - test_case[0][0][1])
        ee_z_e = abs(your_ee[2] - test_case[0][0][2])
        ee_offset = sqrt(ee_x_e ** 2 + ee_y_e ** 2 + ee_z_e ** 2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)


if __name__ == "__main__":
    # kuka = Kuka()
    # kuka.debug(
    #     '/Users/saminda/Udacity/RoboND/catkin_ws/src/RoboND-Kinematics-Project/proj2_debug/debug_joint_state.json')

    test_case_number = 2
    test_code(test_cases[test_case_number])
