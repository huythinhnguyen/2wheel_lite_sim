from curses import def_prog_mode
from math import radians
import matplotlib
import numpy as np
from .Motion import GPS

def wrap_to_pi(a):
    if a > np.pi:
        a -= 2*np.pi
    elif a < -np.pi:
        a += 2*np.pi
    return a

class NaiveSquare:
    def __init__(self, init_pose, width, dt=0.01, speed=0.05, max_omega = np.pi/10):
        self.init_pose = init_pose if type(init_pose)==np.ndarray else np.array(init_pose)
        self.width = width
        self.dt = 0
        self.corners = np.zeros((4,3))
        self.corners[0] = self.init_pose
        for i in range(3):
            self.corners[i+1] = self.corners[i] + np.array([ self.width*np.cos(self.corners[i,2]),
                                                             self.width*np.sin(self.corners[i,2]), 
                                                             np.pi/2 ])
            self.corners[i+1,2] = wrap_to_pi(self.corners[i+1,2])
        self.gps = GPS(pose = init_pose)
        self.goal_pose = self.corners[1]
        self.speed = speed
        self.max_omega = max_omega
        self._run_ended = False
        self.stage = 1

    
    def reset(self, pose=None):
        self.gps.reset(pose)
        self.goal_pose = self.corners[1]
        self._run_ended = False


    def update_goal(self, stage):
        self.goal_pose = self.corners[stage]


    def simple_w_calculate(self, current_pose):
        heading_error = wrap_to_pi(self.goal_pose[2] - current_pose[2])
        w = self.max_omega * np.sign(heading_error)
        cosine_score = np.cos(heading_error)
        if cosine_score < np.cos(radians(20)):
            w = w/1
        elif cosine_score < 0.99:
            w = w/2
        else:
            w = 0.0
        return w


    def simple_v_calculate(self,current_pose):
        distance_to_goal = np.linalg.norm(current_pose[:2] - self.goal_pose[:2])
        heading_error = wrap_to_pi(self.goal_pose[2] - current_pose[2])
        if distance_to_goal > 0.5:
            v = self.speed
        else:
            v = ((self.speed - 0.0)/0.5)*distance_to_goal + 0.0
        if np.cos(heading_error) < 0.1: v = 0.0
        return v


    def track_goal(self, pose):
        self.gps.update(pose)
        if self._run_ended:
            return self.reset()
        d_err = np.linalg.norm(self.gps.pose_hat[:2] - self.goal_pose[:2])
        h_err = wrap_to_pi(self.goal_pose[2] - self.gps.pose_hat[2])
        if d_err < 0.05:
            if self.stage == 0:
                self._run_ended = True
            self.stage += 1
            self.stage = 0 if self.stage == 4 else self.stage
            self.update_goal(self.stage)
        v = self.simple_v_calculate(self.gps.pose_hat)
        w = self.simple_w_calculate(self.gps.pose_hat)
        kinematic = np.array([v,w])
        return kinematic

