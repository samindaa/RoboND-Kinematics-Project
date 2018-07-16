from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2
from sympy.matrices import Matrix
import numpy as np
import collections
import json
import collections


# Saminda Abeyruwan
# Testing TF with RViz for corrections.

def T(alpha, a, q, d):
    return Matrix([[cos(q), -sin(q), 0, a],
                   [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                   [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), cos(alpha) * d],
                   [0, 0, 0, 1]])


def R_z(theta):
    return Matrix([[cos(theta), -sin(theta), 0, 0],
                   [sin(theta), cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])


def R_x(theta):
    return Matrix([[1, 0, 0, 0],
                   [0, cos(theta), -sin(theta), 0],
                   [0, sin(theta), cos(theta), 0],
                   [0, 0, 0, 1]])


def R_y(theta):
    return Matrix([[cos(theta), 0, sin(theta), 0],
                   [0, 1, 0, 0],
                   [-sin(theta), 0, cos(theta), 0],
                   [0, 0, 0, 1]])


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


joint_state_path = '/Users/saminda/Udacity/RoboND/catkin_ws/src/RoboND-Kinematics-Project/proj2_debug/debug_joint_state.json'

def test2():
    JointState = namedtuple_with_defaults("JointState",
                                          " ".join(['joint_{}'.format(i + 1) for i in range(6)]),
                                          [0 for _ in range(6)])
    joint_state = JointState()
    with open(joint_state_path, 'w') as f:
        json.dump(joint_state.__dict__, f)

def test3():
    with open(joint_state_path, 'r') as f:
        joint_state_dict = json.load(f)

    joint_state = collections.namedtuple('JointState', joint_state_dict.keys())(**joint_state_dict)

    print(joint_state)


def test4():
    ### Create symbols for joint variables
    q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')  # theta i
    d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
    a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
    alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

    # KUKA KR210
    # DH Parameters
    s = {alpha0: 0, a0: 0, d1: 0.75,
         alpha1: -pi / 2, a1: 0.35, d2: 0, q2: q2 - pi / 2,
         alpha2: 0, a2: 1.25, d3: 0,
         alpha3: -pi / 2, a3: -0.054, d4: 1.50,
         alpha4: pi / 2, a4: 0, d5: 0,
         alpha5: -pi / 2, a5: 0, d6: 0,
         alpha6: 0, a6: 0, d7: 0.303, q7: 0
         }

    # Homogeneous transformation
    T0_1 = T(alpha0, a0, q1, d1)
    T0_1 = T0_1.subs(s)

    T1_2 = T(alpha1, a1, q2, d2)
    T1_2 = T1_2.subs(s)

    T2_3 = T(alpha2, a2, q3, d3)
    T2_3 = T2_3.subs(s)

    T3_4 = T(alpha3, a3, q4, d4)
    T3_4 = T3_4.subs(s)

    T4_5 = T(alpha4, a4, q5, d5)
    T4_5 = T4_5.subs(s)

    T5_6 = T(alpha5, a5, q6, d6)
    T5_6 = T5_6.subs(s)

    T6_G = T(alpha6, a6, q7, d7)
    T6_G = T6_G.subs(s)

    # Transform from base link to end effector
    T0_2 = simplify(T0_1 * T1_2)  # O_fixed_base to O_origin2
    T0_3 = simplify(T0_2 * T2_3)  # O_fixed_base to O_origin3
    T0_4 = simplify(T0_3 * T3_4)  # O_fixed_base to O_origin4
    T0_5 = simplify(T0_4 * T4_5)  # O_fixed_base to O_origin5
    T0_6 = simplify(T0_5 * T5_6)  # O_fixed_base to O_origin2
    T0_G = simplify(T0_6 * T6_G)  # O_fixed_base to O_originG

    R_corr = simplify(R_z(np.pi) * R_y(-np.pi / 2))

    # Total homogeneous transformation between O_fixed_base to O_gripper
    # with orientation correction applied

    T0_E = simplify(T0_G * R_corr)

    # print("T0_1 = ", T0_1.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_2 = ", T0_2.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_3 = ", T0_3.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_4 = ", T0_4.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_5 = ", T0_5.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_6 = ", T0_6.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
    # print("T0_G = ", T0_G.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))

    # T0_E = T0_EE.evalf(subs={q1: 0.28,q2: 0, q3: 0, q4: 0,q5: 0,q6: 0})
    # print("T0_E = ", T0_E)
    # r_E = T0_E.col(-1)
    # r_E = np.array(r_E).astype(np.float64)
    # print("r_E  = ", r_E)

    while True:
        _ = raw_input("[continue]")

        with open(joint_state_path, 'r') as f:
            joint_state_dict = json.load(f)

        joint_state = collections.namedtuple('JointState', joint_state_dict.keys())(**joint_state_dict)

        print(joint_state)

        T0_P = T0_E.evalf(subs={q1: joint_state.joint_1,
                                q2: joint_state.joint_2,
                                q3: joint_state.joint_3,
                                q4: joint_state.joint_4,
                                q5: joint_state.joint_5,
                                q6: joint_state.joint_6})
        print("T0_E = ", T0_P)
        r_P = T0_P.col(-1)
        r_P = np.array(r_P).astype(np.float64)
        print("r_P  = ", r_P)


def test5():
    #symbols_dict = dict(('a%d' % k, symbols('a%d' % k)) for k in range(3))
    #print(type(symbols_dict['a0']))
    # q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
    q_dict = dict(('q%d' % k, symbols('q%d' % k)) for k in range(1, 8))
    # d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
    d_dict = dict(('d%d' % k, symbols('d%d' % k)) for k in range(1, 8))
    # a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
    a_dict = dict(('a%d' % k, symbols('a%d' % k)) for k in range(7))
    # alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
    alpha_dict = dict(('alpha%d' % k, symbols('alpha%d' % k)) for k in range(7))
    print(q_dict)
    print(d_dict)
    print(a_dict)
    print(alpha_dict)


def test6():
    pass

if __name__ == "__main__":
    #test2()
    #test3()
    test4()
    # test5()
