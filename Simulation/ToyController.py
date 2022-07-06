from math import radians
import numpy as np
from Motion import GPS


class NaiveSquare:
    def __init__(self, init_pose, width, dt=0.5, speed=0.05, max_omega = 0.03):
        self.init_pose = init_pose if type(init_pose)==np.ndarray else np.array(init_pose)
        self.width = width
        self.dt = 0
        self.corners = np.zeros((4,3))
        self.corners[0] = self.init_pose
        for i in range(3):
            self.corners[i+1] = self.corners[i] + np.array([ self.width*np.cos(self.corners[i,2]),
                                                             self.width*np.sin(self.corners[i,2]), 
                                                             np.pi ])
            self.corners[i+1,2] = np.clip(self.corners[i+1,2],-np.pi, np.pi)
        self.gps = GPS(pose = init_pose)
        self.goal_pose = self.corners[1]
        self.speed = speed
        self.max_omega = max_omega

    
    def reset(self, pose=None):
        self.gps.reset(pose)
        self.goal_pose = self.corners[1]


    def update_goal(self, stage):
        self.goal_pose = self.corners[stage]


    def simple_w_calulate(self, current_pose):
        heading_error = np.clip(self.goal_pose[2] - current_pose[2],-np.pi, np.pi)
        w = self.max_omega * np.sign(heading_error)
        cosine_score = np.cos(heading_error)
        if cosine_score < np.cos(radians(10)):
            w = w/3
        elif cosine_score < 0.999:
            w = w/3
        else:
            w = 0.0
        return w


    def simple_v_calculate(self,current_pose):
        distance_to_goal = np.linalg.norm(current_pose[:2] - self.goal_pose[:2])
        if distance_to_goal > 0.1:
            v = self.speed
        else:
            v = (distance_to_goal/0.1)*self.speed
        return v


    def track_goal(self):
        pass
        ### Need to work on this!!