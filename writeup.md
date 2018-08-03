## Project: Kinematics Pick & Place
### Saminda Abeyruwan

In this project, I have used  inverse kinematics to calculate joint angles for Kuka KR210 
to perform a pick and place task within Gazebo simulator. I have observed that the implementation has
successfully completed the cycles (more than 8 / 10) during the simulations, and I have provided herewith 
my implementation details, and conclusions. 

[//]: # (Image References)

[image1]: ./misc_images/dh_config.png
[image2]: ./misc_images/theta1_2_3.png
[image3]: ./misc_images/theta4_5_6.png
[image4]: ./misc_images/pic0.png
[image5]: ./misc_images/pic1.png
[image6]: ./misc_images/pic2.png
[image7]: ./misc_images/pic3.png
[image8]: ./misc_images/pic4.png
[image9]: ./misc_images/pic5.png
[image10]: ./misc_images/pic6.png
[image11]: ./misc_images/pic7.png
[image12]: ./misc_images/pic8.png
[image13]: ./misc_images/pic9.png


### Kinematic Analysis
#### 1. Run the forward_kinematics demo and evaluate the kr210.urdf.xacro file to perform kinematic analysis of Kuka KR210 robot and derive its DH parameters.

I have used the following figure to perform kinematic analysis and define the DH parameters.  

![alt text][image1]

The figure contains both DH parameters (red and green axes) and axes with respect to URDF file (black).
The DH parameter table is as follows:

Links | alpha(i-1) | a(i-1) | theta(i) | d(i)
--- | --- | --- | --- | ---
0->1 | 0 | 0 | theta1 | d1 = 0.75
1->2 | -pi/2 | a1 = 0.35 | theta2 - pi/2| 0
2->3 | 0 | a2 = 1.25 | theta3 | 0
3->4 | -pi/2 | a3 = -0.054 | theta4 | d4 = 1.50
4->5 | +pi/2 | 0 | theta5 | 0
5->6 | -pi/2 | 0 | theta6 | 0
6->G | 0 | 0 | 0 | dG = 0.303

where alpha(i-1): twist angle, a(i-1): the distance along X(i-1), theta(i): 
joint angles, and d(i): offset to joint origin O(i). 6->G represent the distance 
from the wrist center to gripper (O_gripper in the figure). 

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.

I have implemented a class _class Kuka210_, which contains the transformation matrices and homogeneous transformations.

```python
class Kuka210(object):
    
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
```

In the constructor, I have setup the necessary homogeneous transformations using _sympy_ and _mpmath_.

```python
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
```

I have used the _self.s_ to hold the DH parameters, and the keys are generated from sympy using the dicts, _self.q_,
_self.d_, _self.a_, and _self.alpha_. _self.limit_ contains the range of the joints w.r.t  kr210.urdf.xacro.

#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

The joint angles are calculated from inverse kinematics, and the implementation is available in:

```python
def ik(self, req, x):
```

where _req_ is _CalculateIKRequest_ and _x_ is index of the current _geometry_msgs/Pose_. I have used 
the following figure as a reference to calculate _theta1_, _theta2_, and _theta3_.

![alt text][image2]

The implementation of the concepts is as follows:

```python
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
        theta3 = pi / 2 - angle_b - 0.036
```

I have used the following figure as a reference to calculate _theta4_, _theta5_, and _theta6_.

![alt text][image3]

The implementation is as follows:

```python
        R0_3 = self.T0_1[0:3, 0:3] * self.T1_2[0:3, 0:3] * self.T2_3[0:3, 0:3]
        R0_3 = R0_3.evalf(subs={self.q['q1']: theta1, self.q['q2']: theta2, self.q['q3']: theta3})
        R3_6 = R0_3.transpose() * Rot_rpy

        # For joint4, joint5, and joint6
        theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
        theta5 = atan2(sqrt(R3_6[0, 2] ** 2 + R3_6[2, 2] ** 2), R3_6[1, 2])
        theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])
```

The joint angles have been clipped as follows:

```python
        theta1 = np.clip(theta1, self.limit['theta1']['lower'], self.limit['theta1']['upper'])
        theta2 = np.clip(theta2, self.limit['theta2']['lower'], self.limit['theta2']['upper'])
        theta3 = np.clip(theta3, self.limit['theta3']['lower'], self.limit['theta3']['upper'])
        theta4 = np.clip(theta4, self.limit['theta4']['lower'], self.limit['theta4']['upper'])
        theta5 = np.clip(theta5, self.limit['theta5']['lower'], self.limit['theta5']['upper'])
        theta6 = np.clip(theta6, self.limit['theta6']['lower'], self.limit['theta6']['upper'])
```

The object:

```python
kuka210 = Kuka210()
```

has been initialized in the _main_ method and called from the _handle_calculate_IK_ method.

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 

I have implemented the code as listed above. In order to time optimize sympy initializations, I have created a class _Kuka210_ 
and calculated the joint angles. I have used the debug code _RoboND-Kinematics-Project/proj2_debug/kuka_dh_opt.py_ code
to test forward kinematics and inverse kinematics calculations. 

Due to resource limitation in my Mac, I have ran full simulations only once, and the task completed successfully. I have 
captured following figures while the simulation was running. 

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]