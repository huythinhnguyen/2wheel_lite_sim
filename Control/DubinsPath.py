import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/./Simulation/")

import numpy as np
from math import sin, cos, atan2, sqrt, acos, pi, hypot

from Simulation.Motion import State

### Utility function wrap to pi ###
def wrap_to_pi(a):
    if a >= pi:
        a -= 2*pi
    elif a < -pi:
        a += 2*pi
    return a
"""
DUBINS PATH PLANNER USER GUIDE.
Author: Thinh Nguyen

sample code:
##############################
import numpy as np
from Control import DubinsPath
from matplotlib import pyplot as plt

start_pose = np.array([0.0,0.0, -np.pi/4])
goal_pose = np.array([-5., 10.5, 3*np.pi/4])
min_turn_radius = 2.0

course, inter_poses, distances, modes = DubinsPath.generate_course(start_pose, goal_pose, min_turn_radius, speed=0.1, dt=0.1)
checkpoints = np.vstack((start_pose, 
                         inter_poses['A'], 
                         inter_poses['B'], 
                         goal_pose
                        ))

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(course[:,0], course[:,1])

ax.scatter(checkpoints[:,0],checkpoints[:,1])
ax.quiver(checkpoints[:,0],checkpoints[:,1],
           0.175*np.cos(checkpoints[:,2]),0.175*np.sin(checkpoints[:,2]),
           pivot='tail',scale_units='xy',scale=0.15,width=0.01,color='b')
ax.set_aspect('equal')

#############################

To get the checkpoints alone, please use DubinsPath.plan()

"""


def plan(start_pose, goal_pose, min_turn_radius, selected_path_types=None):
    if selected_path_types is None:
        path_funcs = _DUBINS_PATH_TYPES.values()
    else:
        path_funcs = [_DUBINS_PATH_TYPES[path_type] for path_type in selected_path_types]
    best_cost = float('inf')
    best_inter_poses, best_modes = None, None
    for plan in path_funcs:
        inter_poses, distances, modes = plan(start_pose, goal_pose, min_turn_radius)
        cost = sum(distances)
        if best_cost > cost:
            best_inter_poses, best_distances, best_modes, best_cost = inter_poses, distances, modes, cost

    return best_inter_poses, best_distances, best_modes


def generate_course(start_pose, goal_pose=None, min_turn_radius=None, distances=None, modes=None, speed=0.1, dt=0.1, selected_path_types=None):
    if goal_pose is not None:
        inter_poses, distances, modes = plan(start_pose, goal_pose, min_turn_radius, selected_path_types)
    if min_turn_radius is None or distances is None or modes is None:
        raise ValueError('min_turn_radius, distances, modes cannot be None')
    state = State(pose=start_pose, kinematic = [speed, 0.0], dt=dt)
    course = start_pose.reshape(1,3)
    w = speed/min_turn_radius
    for i in range(3):
        dist = 0
        while dist < distances[i]:
            omega = w if modes[i]=='L' else -w if modes[i]=='R' else 0.0
            state.update_kinematic(new_w = omega)
            state.update_pose()
            dist += min_turn_radius*w*dt
            course = np.vstack((course, state.pose))
    if goal_pose is None:
        return course
    else:
        return course, inter_poses, distances, modes
    

    
def _LSL(start_pose, goal_pose, min_turn_radius):
    modes = ['L','S','L']    
    r = min_turn_radius
    C1 = start_pose[:2] + r*np.array([cos(start_pose[2]+pi/2),sin(start_pose[2]+pi/2)])
    C2 = goal_pose[:2] + r*np.array([cos(goal_pose[2]+pi/2), sin(goal_pose[2]+pi/2)])
    yaw_C2C1 = atan2(C2[1]-C1[1], C2[0]-C1[0])
    A = C1 + r*np.array([cos(yaw_C2C1 - pi/2), sin(yaw_C2C1 - pi/2)])
    B = C2 + r*np.array([cos(yaw_C2C1 - pi/2), sin(yaw_C2C1 - pi/2)])
    A_pose = np.append(A, yaw_C2C1)
    B_pose = np.append(B, yaw_C2C1)
    
    theta1 = abs(wrap_to_pi(A_pose[2]-start_pose[2]))
    theta1 = 2*pi - theta1 if wrap_to_pi(A_pose[2]-start_pose[2]) < 0 else theta1
    #theta2 = 0.0
    theta3 =  abs(wrap_to_pi(goal_pose[2]-B_pose[2]))
    theta3 = 2*pi - theta3 if wrap_to_pi(goal_pose[2]-B_pose[2]) < 0 else theta3
    
    d1 = theta1*r
    d2 = hypot(B[1]-A[1], B[0]-A[0])
    d3 = theta3*r
    
    inter_poses = {'A': A_pose, 'B': B_pose}
    distances = [d1,d2,d3]
    
    return inter_poses, distances, modes, 

def _LSR(start_pose, goal_pose, min_turn_radius):
    modes = ['L','S','R']    
    r = min_turn_radius
    C1 = start_pose[:2] + r*np.array([cos(start_pose[2]+pi/2),sin(start_pose[2]+pi/2)])
    C2 = goal_pose[:2] + r*np.array([cos(goal_pose[2]-pi/2), sin(goal_pose[2]-pi/2)])
    yaw_C2C1 = atan2(C2[1]-C1[1], C2[0]-C1[0])
    if 2*r < hypot(C2[1]-C1[1], C2[0]-C1[0]):
        alpha = acos(2*r/ hypot(C2[1]-C1[1], C2[0]-C1[0]))
        A = C1 + r*np.array([cos(yaw_C2C1 - alpha), sin(yaw_C2C1 - alpha)])
        B = C2 + r*np.array([cos(yaw_C2C1 - alpha + pi), sin(yaw_C2C1 +  - alpha + pi)])
        A_pose = np.append(A, wrap_to_pi(yaw_C2C1 - alpha + pi/2))
        B_pose = np.append(B, wrap_to_pi(yaw_C2C1 - alpha + pi/2))
    
        theta1 = abs(wrap_to_pi(A_pose[2]-start_pose[2]))
        theta1 = 2*pi - theta1 if wrap_to_pi(A_pose[2]-start_pose[2]) < 0 else theta1
        #theta2 = 0.0
        theta3 =  abs(wrap_to_pi(goal_pose[2]-B_pose[2]))
        theta3 = 2*pi - theta3 if wrap_to_pi(goal_pose[2]-B_pose[2]) > 0 else theta3
        
        d1 = theta1*r
        d2 = hypot(B[1]-A[1], B[0]-A[0])
        d3 = theta3*r
    
        inter_poses = {'A': A_pose, 'B': B_pose}
        distances = [d1,d2,d3]
    else:
        inter_poses = None
        distances = [float('inf'), 0, 0]
    
    
    return inter_poses, distances, modes, 


def _RSL(start_pose, goal_pose, min_turn_radius):
    modes = ['R','S','L']    
    r = min_turn_radius
    C1 = start_pose[:2] + r*np.array([cos(start_pose[2]-pi/2),sin(start_pose[2]-pi/2)])
    C2 = goal_pose[:2] + r*np.array([cos(goal_pose[2]+pi/2), sin(goal_pose[2]+pi/2)])
    yaw_C2C1 = atan2(C2[1]-C1[1], C2[0]-C1[0])
    if 2*r < hypot(C2[1]-C1[1], C2[0]-C1[0]):
        alpha = acos(2*r/ hypot(C2[1]-C1[1], C2[0]-C1[0]))
    
        A = C1 + r*np.array([cos(yaw_C2C1 + alpha), sin(yaw_C2C1 + alpha)])
        B = C2 + r*np.array([cos(yaw_C2C1 + alpha - pi), sin(yaw_C2C1 + alpha - pi)])
        A_pose = np.append(A, wrap_to_pi(yaw_C2C1 + alpha - pi/2))
        B_pose = np.append(B, wrap_to_pi(yaw_C2C1 + alpha - pi/2))
    
    
        theta1 = abs(wrap_to_pi(A_pose[2]-start_pose[2]))
        theta1 = 2*pi - theta1 if wrap_to_pi(A_pose[2]-start_pose[2]) > 0 else theta1
        #theta2 = 0.0
        theta3 =  abs(wrap_to_pi(goal_pose[2]-B_pose[2]))
        theta3 = 2*pi - theta3 if wrap_to_pi(goal_pose[2]-B_pose[2]) < 0 else theta3
    
        d1 = theta1*r
        d2 = hypot(B[1]-A[1], B[0]-A[0])
        d3 = theta3*r
    
        inter_poses = {'A': A_pose, 'B': B_pose}
        distances = [d1,d2,d3]
    else:
        inter_poses = None
        distances = [float('inf'), 0, 0]
    
    return inter_poses, distances, modes, 


def _RSR(start_pose, goal_pose, min_turn_radius):
    modes = ['R','S','R']
    r = min_turn_radius
    C1 = start_pose[:2] + r*np.array([cos(start_pose[2]-pi/2),sin(start_pose[2]-pi/2)])
    C2 = goal_pose[:2] + r*np.array([cos(goal_pose[2]-pi/2), sin(goal_pose[2]-pi/2)])
    yaw_C2C1 = atan2(C2[1]-C1[1], C2[0]-C1[0])
    A = C1 + r*np.array([cos(yaw_C2C1 + pi/2), sin(yaw_C2C1 + pi/2)])
    B = C2 + r*np.array([cos(yaw_C2C1 + pi/2), sin(yaw_C2C1 + pi/2)])
    A_pose = np.append(A, yaw_C2C1)
    B_pose = np.append(B, yaw_C2C1)
    
    theta1 = abs(wrap_to_pi(A_pose[2]-start_pose[2]))
    theta1 = 2*pi - theta1 if wrap_to_pi(A_pose[2]-start_pose[2]) > 0 else theta1
    #theta2 = 0.0
    theta3 =  abs(wrap_to_pi(goal_pose[2]-B_pose[2]))
    theta3 = 2*pi - theta3 if wrap_to_pi(goal_pose[2]-B_pose[2]) > 0 else theta3
    
    d1 = theta1*r
    d2 = hypot(B[1]-A[1], B[0]-A[0])
    d3 = theta3*r
    
    inter_poses = {'A': A_pose, 'B': B_pose}
    distances = [d1,d2,d3]
    
    return inter_poses, distances, modes


_DUBINS_PATH_TYPES = {'LSL': _LSL, 'LSR': _LSR, 'RSL': _RSL, 'RSR': _RSR, }

