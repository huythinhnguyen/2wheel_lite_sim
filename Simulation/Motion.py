import numpy as np
import sys
import warnings
from . import Setting


class State:
    def __init__(self, pose:'[x, y, yaw] (meter)'=None, kinematic:'[v (m/s), w(rad/s)]'=None, dt=0.5):
        if type(pose)==np.ndarray:
            self.pose = pose.reshape(3,)
        else:
            self.pose = np.array([0.0, 0.0, 0.0]) if pose == None else np.array(pose)
        if type(kinematic)==np.ndarray:
            self.kinematic = kinematic.reshape(2,)
        else:
            self.kinematic = np.array([0.0, 0.0]) if kinematic==None else np.array(kinematic)
        self._init_state = np.concatenate((self.pose, self.kinematic))
        self.dt = dt


    def reset(self, pose:'[x, y, yaw] (meter)'=None, kinematic:'[v (m/s), w(rad/s)]'=None):
        if type(pose)==np.ndarray:
            self.pose = pose.reshape(3,)
        else:
            self.pose = self._init_state[:3] if pose == None else np.array(pose)
        if type(kinematic)==np.ndarray:
            self.kinematic = kinematic.reshape(2,)
        else:
            self.kinematic = self._init_state[3:] if kinematic ==  None else np.array(kinematic)
        self._init_state = np.concatenate((self.pose, self.kinematic))


    def update_kinematic(self, new_v:'m/s'=None, new_w:'rad/s'=None):
        if new_v!=None: self.kinematic[0] = new_v
        if new_w!=None: self.kinematic[1] = new_w
        # Maybe add a acceleration limitter here

    
    def turning_radius(self):
        if np.abs(self.w) < 0.001: return self.v/self.w
        else: return 'inf'


    def ICC(self):
        x, y, yaw = self.pose
        R = self.turning_radius()
        if R != 'inf': 
            Cx = x - R*np.sin(yaw)
            Cy = y + R*np.cos(yaw)
            return [Cx, Cy]
        else: 
            return None


    def update_pose(self):
        x, y, yaw = self.pose
        v,w = self.kinematic
        ICC = self.ICC()
        if ICC != None:
            turn = w*self.dt
            Rotation = np.array( [[ np.cos(turn), -np.sin(turn), 0.0 ],
                                  [ np.sin(turn),  np.cos(turn), 0.0 ],
                                  [ 0.0         ,   0.0        , 0.0 ]] )
            translation = -1*np.array([*ICC,0]).reshape(3,1)
            inverse_translation = np.array([*ICC, turn]).reshape(3,1)
            new_pose = np.matmul( Rotation,self.pose.reshape(3,1) + translation) + inverse_translation
        else:
            move = v*self.dt
            translation = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            new_pose = self.pose + translation
        self.pose = new_pose.reshape(3,)


class GPS:
    def __init__(self, var=Setting.steam_gps_var, pose=None):
        self.var = np.array(var)
        if type(pose)==np.ndarray:
            self.pose = pose.reshape(3,)
        else:
            self.pose = np.array([0.0, 0.0, 0.0]) if pose==None else np.array(pose)
        self.pose_hat = pose + np.sqrt(self.var)*np.random.randn(3)
        self.init_pose = np.copy(self.pose)
    

    def reset(self,pose = None):
        self.pose = np.copy(self.init_pose) if pose==None else np.array(pose)
        self.pose_hat = pose + np.sqrt(self.var)*np.random.randn(3)
        self.init_pose = np.copy(self.pose)


    def update(self,pose):
        self.pose = pose
        self.pose_hat = pose + np.sqrt(self.var)*np.random.randn(3)


class Drive:
    def __init__(self, mode='create2'):
        if mode=='create2':
            self.bot = Setting.Create2()
        self.kinematic = np.array([0.0, 0.0]) # v, w
        self.direct = np.array([0.0, 0.0]) # vL, vR
        self.mode = mode


    def reset(self):
        self.bot.reset()

        
    def update_kinematic(self, new_v, new_w):
        self.kinematic[0] = new_v
        self.kinematic[1] = new_w


    def update_direct(self, new_vL, new_vR):
        self.direct[0] = new_vL
        self.direct[1] = new_vR


    def kinematic_to_direct(self,int=False, wrapping=True):
        self.direct[0] = self.kinematic[0] - (self.bot.wheelbase/2) * self.kinematic[1]
        self.direct[1] = self.kinematic[0] - (self.bot.wheelbase/2) * self.kinematic[1]
        if int:
            self.direct = np.round(self.direct,3)
        self.direct[ np.argwhere(abs(self.direct)<0.011)] = 0
        if wrapping:
            self.direct = np.clip(self.direct, self.bot.wheel_velocity_range[0], self.bot.wheel_velocity_range[1])


    def direct_to_kinematic(self):
        self.kinematic[0] = (self.direct[1] + self.direct[0])/2
        self.kinematic[1] = (self.direct[1] - self.direct[0])/self.bot.wheelbase


    def noisy_direct(self, factors, var):
        # offset left and right to biasing to one side (depending on the setting)
        self.direct = factors * self.direct
        # add gaussian noise
        self.direct = self.direct + np.sqrt(var) * np.random.randn(2)


    def kinematic_update(self, new_kinematic):
        self.update_kinematic(new_kinematic[0], new_kinematic[1])
        self.kinematic_to_direct(int=True)
        # ADD NOISE TO WHEEL VELOCITIES
        self.noisy_direct(factors = np.array([self.bot.v_left_factor, self.bot.v_right_factor]),
                          var = self.bot.wheel_velocity_var)
        self.direct_to_kinematic()
