import numpy as np
import sys
import warnings
import Setting


class State:
    def __init__(self, pose:'[x, y, yaw] (meter)'=None, kinematic:'[v (m/s), w(rad/s)]'=None, dt=0.5):
        self.pose = np.array([0.0, 0.0, 0.0]) if pose == None else np.array(pose)
        self.kinematic = np.array([0.0, 0.0]) if kinematic==None else np.array(kinematic)
        self._init_state = np.concatenate((self.pose, self.kinematic))
        self.dt = dt


    def reset(self, pose:'[x, y, yaw] (meter)', kinematic:'[v (m/s), w(rad/s)]'):
        self.pose = self._init_state[:3] if pose == None else np.array(pose)
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
        R = self.turning_radius
        if R != 'inf': return [x - R*np.sin(yaw), y + R*np.cos(yaw)]
        else: return None


    def update_pose(self):
        x, y, yaw = self.pose
        v,w = self.kinematic
        ICC = self.ICC
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


class Drive:
    def __init__(self, mode='create2'):
        if mode=='create2':
            self.bot = Setting.Create2
        self.kinematic = np.array([0.0, 0.0]) # v, w
        self.direct = np.array([0.0, 0.0]) # vL, vR
        
    def update_kinematic(self, new_v, new_w):
        self.kinematic[0] = new_v
        self.kinematic[1] = new_w


    def update_direct(self, new_vL, new_vR):
        self.direct[0] = new_vL
        self.direct[1] = new_vR


    def kinematic_to_direct(self):
        self.direct[0] = self.kinematic[0] - (self.bot.wheelbase/2) * self.kinematic[1]
        self.direct[1] = self.kinematic[0] - (self.bot.wheelbase/2) * self.kinematic[1]
        

    def direct_to_kinematic(self):
        self.kinematic[0] = (self.direct[1] + self.direct[0])/2
        self.kinematic[1] = (self.direct[1] - self.direct[0])/self.bot.wheelbase


    #######################################################################
    # NEXT, LET'S TRY TO INJECT SOME NOISE TO TO THIS DRIVE!         
    # A good start will be using some sort of loop in inject noise   
    # during the conversion of direct and kinematic.                
    # Ideal 1:
    # -------
    # The Motion will receive a set v,w (kinematic) from a soon to-be
    # Control module. Next, it will will try to precess to from v, w
    # to vL, vR. 
    # Next, round vL, vR to the closest integer. Add some noise  from wheel
    # velocity var.
    # Next steps, use the noisy vL, vR and convert it back to v, w to
    # find the next pose.
    # Feed the x,y,yaw (pose) back to the Control module for next action   
    # 
    #######################################################################