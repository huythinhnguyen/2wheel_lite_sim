import numpy as np
from Motion import GPS

def wrap_to_pi(a):
    if a > np.pi:
        a -= 2*np.pi
    elif a < -np.pi:
        a += 2*np.pi
    return a

class NaiveSquare:
    def __init__(self, init_pose, width, dt=0.5):
        self.init_pose = init_pose if type(init_pose)==np.ndarray else np.array(init_pose)
        self.width = width
        self.dt = 0
        self.corners = np.zeros((4,3))
        self.corners[0] = self.init_pose
        for i in range(3):
            self.corners[i+1] = self.corners[i] + np.array([ self.width*np.cos(self.corners[i,2]),
                                                             self.width*np.sin(self.corners[i,2]), 
                                                             np.pi ])
            self.corners[i+1,2] = wrap_to_pi(self.corners[i+1,2])
        self.gps = GPS(pose = init_pose)
        self.goal_pose = self.corners[1]

    
    def reset(self):
        self.current_pose = np.copy(self.init_pose)
        self.goal_pose = self.corners[1]


    def update_goal(self, stage):
        self.goal_pose = self.corners[stage]


    def turn_toward_goal(self, current_pose):
        pass
