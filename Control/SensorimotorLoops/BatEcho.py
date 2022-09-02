import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from Sensors.BatEcho import Setting, Spatializer
from . import Setting as config

render = Spatializer.Render()
BACKGROUND = render.get_compressed([])
garbage = [render]
del render
del garbage

"""
reference:
Lee, David (1992): Common principle of guidance of echolocation and vision
Padfield, Gareth (2011): The tau of flight control
--> Derived the linear velocity profile to be:
v = v_max * K(d), v[v>v_max] = v_max
K(d) = 1 - (1 - k*a*d)^(k^-1 - 1)
v: velocity, k = 0.1, 0.2 --> others are not computation --> power result is imaginary number
a = between 1-9 but found optimal around 2,3 as a larger a is the stepper the decellaration

Angular velocity are determined by different algorithm. --> Need tested.

"""


class Cue:
    def __init__(self, background = None,):
        self.DISTANCE_REFERENCE = Setting.COMPRESSED_DISTANCE_ENCODING
        self.window_size=10
        self.emission_encoding = Setting.EMISSION_ENCODING
        self.IID_CAP = 10
        self.IID_QUIET_RAND = 0.1
        self.onset_threshold = 1 * Setting.QUIET_THRESHOLD
        if background is None: self.left_bg, self.right_bg = BACKGROUND.values()
        else:
            if type(background)==tuple or type(background)==list: self.left_bg, self.right_bg = background
            elif type(background)==dict: self.left_bg, self.right_bg = background.values()

        self.cache = {}


    def get_cues(self, input_echoes, cache=True):
        onset_distance, onset_index = self._get_onset(input_echoes).values()
        left_loudness = self._calc_window_loudness(input_echoes['left'], onset_index)
        right_loudness= self._calc_window_loudness(input_echoes['right'], onset_index)
        IID = self._calc_iid(left_loudness, right_loudness)
        cues = {'onset_distance': onset_distance,
                'onset_index': onset_index,
                'left_loudness': left_loudness,
                'right_loudness': right_loudness,
                'IID': IID }
        if cache: self.cache = cues.copy()
        return cues
            

    def _get_onset(self,inp):
        if type(inp)==tuple or type(inp)==list: left, right = inp
        elif type(inp)==dict: left, right = inp.values()
        onset_index_left = np.argmax( (left - self.left_bg) > self.onset_threshold)
        onset_index_right= np.argmax( (right-self.right_bg) > self.onset_threshold)
        if onset_index_left == 0: onset_index_left = len(left) - 1
        if onset_index_right== 0: onset_index_right= len(right)- 1
        onset_index = np.min([onset_index_left, onset_index_right])
        onset_distance = self.DISTANCE_REFERENCE[onset_index]
        onset = {'distance': onset_distance, 'index': onset_index}
        return onset


    def _calc_window_loudness(self, data, onset_index):
        return np.sum(data[onset_index:onset_index+self.window_size])


    def _calc_iid(self, left_loudness, right_loudness):
        if left_loudness*right_loudness != 0:
            iid = 10*np.log10(left_loudness/right_loudness)
        elif (left_loudness + right_loudness) == 0:
            iid = self.IID_QUIET_RAND * np.random.rand() - 0.5*self.IID_QUIET_RAND
        else:
            iid = self.IID_CAP * np.sign(left_loudness - right_loudness)
        return iid
        

class Avoid(Cue):
    def __init__(self, background=None):
        super().__init__(background)
        self.bot_convert = config.ROBOT_CONVERSION
        self.max_linear_velocity = config.MAX_LINEAR_VELOCITY if not self.bot_convert else config.ROBOT_MAX_LINEAR_VELOCITY
        self.max_angular_velocity= config.MAX_ANGULAR_VELOCITY if not self.bot_convert else config.ROBOT_MAX_ANGULAR_VELOCITY
        self.max_angular_acceleration = config.MAX_ANGULAR_ACCELERATION if not self.bot_convert else config.ROBOT_MAX_ANGULAR_ACCELERATION
        self.linear_velo_offset = config.LINEAR_VELOCITY_OFFSET if not self.bot_convert else config.ROBOT_LINEAR_VELOCITY_OFFSET
        self.dt = 1/config.CHIRP_RATE if not self.bot_convert else 1/config.ROBOT_CHIRP_RATE
        self.A = config.DECELERATION_FACTOR
        self.K = config.TAU_K
        self.g = config.GRAVI_ACCEL
        self.centri_accel = config.CENTRIFUGAL_ACCEL
        self.kine_cache = {'v': 0., 'omega': 0.}
        self.body_radius = config.BODY_RADIUS
        self.B = config.BAIL_DISTANCE_MULTIPLIER
        self.linear_accel_limit = config.LINEAR_ACCEL_LIMIT
        self.linear_decel_limit = config.LINEAR_DECEL_LIMIT

    
    def get_kinematic(self, input_echoes):
        cues = self.get_cues(input_echoes)
        v = self._get_linear_velocity(cues)
        omega = self._get_angular_velocity(cues, v=v)
        self.kine_cache.update({'v': v, 'omega': omega})
        return v, omega


    def _get_linear_velocity(self,cues):
        distance = cues['onset_distance']
        v = (self.max_linear_velocity - self.linear_velo_offset)*\
            (1 - np.power((1-self.K*self.A*(distance-self.body_radius)),1/self.K-1)) \
            + self.linear_velo_offset
        if v>self.max_linear_velocity: v=self.max_linear_velocity
        return self._linear_accel_cap(v)


    def _linear_accel_cap(self,v):
        v_dot = (v - self.kine_cache['v'])/self.dt
        if v_dot< self.linear_decel_limit:
            return self.kine_cache['v'] + (self.linear_decel_limit/self.max_linear_velocity)*self.kine_cache['v']*self.dt
        #    #return self.kine_cache['v'] - self.linear_accel_limit*self.dt
        elif v_dot > self.linear_accel_limit:
            return self.kine_cache['v'] + self.linear_accel_limit*self.dt
        else:
            return v
    

    def _get_angular_velocity(self,cues, v=None):
        iid = cues['IID']
        if v is None:
            v = self._get_linear_velocity(cues)
        omega = self._plan_B(v, iid, cues['onset_distance'])
        if np.abs(omega) > self.max_angular_velocity:
            omega = np.sign(omega)*self.max_angular_velocity
        return self._angular_accel_cap(omega)


    def _angular_accel_cap(self, omega):
        omega_dot = (omega - self.kine_cache['omega'])/self.dt
        if np.abs(omega_dot) > self.max_angular_acceleration:
            return self.kine_cache['omega'] + self.max_angular_acceleration*self.dt*np.sign(omega_dot)
        else: return omega
        

    def _plan_A(self, v, iid):
        R_min = np.power(v,2) / self.centri_accel
        omega = -np.sign(iid) * (v/R_min)
        return omega

    ### USE THIS PLAN! ###
    def _plan_B(self, v, iid, onset_distance):
        R_min = np.power(v,2) / self.centri_accel
        w = (v/R_min) * np.exp(-1*(onset_distance-self.body_radius))
        omega = -np.sign(iid) * w if onset_distance > self.B*self.body_radius else np.sign(self.kine_cache['omega'])*w
        return omega

    
    
class Approach(Cue):
    def __init__(self, background=None):
        super().__init__(background)
        self.bot_convert = config.ROBOT_CONVERSION
        self.max_linear_velocity = config.MAX_LINEAR_VELOCITY if not self.bot_convert else config.ROBOT_MAX_LINEAR_VELOCITY
        self.max_angular_velocity= config.MAX_ANGULAR_VELOCITY if not self.bot_convert else config.ROBOT_MAX_ANGULAR_VELOCITY
        self.max_angular_acceleration = config.MAX_ANGULAR_ACCELERATION if not self.bot_convert else config.ROBOT_MAX_ANGULAR_ACCELERATION
        self.linear_velo_offset = config.LINEAR_VELOCITY_OFFSET if not self.bot_convert else config.ROBOT_LINEAR_VELOCITY_OFFSET
        self.dt = 1/config.CHIRP_RATE if not self.bot_convert else 1/config.ROBOT_CHIRP_RATE
        self.A = config.DECELERATION_FACTOR
        self.K = config.TAU_K
        self.g = config.GRAVI_ACCEL
        self.centri_accel = config.CENTRIFUGAL_ACCEL
        self.kine_cache = {'v': 0., 'omega': 0.}
        self.body_radius = config.BODY_RADIUS
        self.linear_accel_limit = config.LINEAR_ACCEL_LIMIT
        self.linear_decel_limit = config.LINEAR_DECEL_LIMIT
        self.steer_damper = config.APPROACH_STEER_DAMPING
    

    def get_kinematic(self, input_echoes):
        cues = self.get_cues(input_echoes)
        v = self._get_linear_velocity(cues)
        omega = self._get_angular_velocity(cues)
        self.kine_cache.update({'v': v, 'omega': omega})
        return v, omega


    def _get_linear_velocity(self,cues):
        distance = cues['onset_distance']
        v = (self.max_linear_velocity - self.linear_velo_offset)*\
            (1 - np.power((1-self.K*self.A*(distance-self.body_radius)),1/self.K-1)) \
            + self.linear_velo_offset
        if v>self.max_linear_velocity: v=self.max_linear_velocity
        return self._linear_accel_cap(v)


    def _linear_accel_cap(self,v):
        v_dot = (v - self.kine_cache['v'])/self.dt
        if v_dot< self.linear_decel_limit:
            return self.kine_cache['v'] + (self.linear_decel_limit/self.max_linear_velocity)*self.kine_cache['v']*self.dt
        #    #return self.kine_cache['v'] - self.linear_accel_limit*self.dt
        elif v_dot > self.linear_accel_limit:
            return self.kine_cache['v'] + self.linear_accel_limit*self.dt
        else:
            return v


    def _get_angular_velocity(self, cues,v=None):
        iid = cues['IID']
        if v is None:
            v = self._get_linear_velocity(cues)
        omega = self._plan_B(v, iid)
        if np.abs(omega) > self.max_angular_velocity:
            omega = np.sign(omega)*self.max_angular_velocity
        return self._angular_accel_cap(omega)
    

    def _angular_accel_cap(self, omega):
        omega_dot = (omega - self.kine_cache['omega'])/self.dt
        if np.abs(omega_dot) > self.max_angular_acceleration:
            return self.kine_cache['omega'] + self.max_angular_acceleration*self.dt*np.sign(omega_dot)
        else: return omega
    

    def _plan_A(self, v, iid):
        return self.max_angular_acceleration * iid / self.steer_damper

    ### Use this Plan
    def _plan_B(self, v, iid):
        R_min = np.power(v,2) / self.centri_accel
        temp = iid/self.steer_damper * np.pi
        temp = np.pi/2 if temp>np.pi/2 else -np.pi/2 if temp<-np.pi/2 else temp
        if temp==0:
            return 0.
        else:
            R_select = 1/np.tan(temp)+np.sign(iid)*R_min
            return v/R_select


class AvoidApproach(Avoid):
    def __init__(self, background=None, **kwargs):
        super().__init__(background)
        self.steer_damper = config.APPROACH_STEER_DAMPING
        self.approach_factor = 0 if 'approach_factor' not in kwargs.keys() else kwargs['approach_factor']


    def get_kinematic(self, input_echoes, approach_factor=None):
        if approach_factor is not None: self.approach_factor = approach_factor
        cues = self.get_cues(input_echoes)
        v = self._get_linear_velocity(cues)
        omega = self._get_angular_velocity(cues)
        self.kine_cache.update({'v': v, 'omega': omega})
        return v, omega


    def _get_angular_velocity(self, cues,v=None):
        iid = cues['IID']
        if v is None:
            v = self._get_linear_velocity(cues)
        turning_radius = self._calc_turning_radius(iid, v, cues['onset_distance'])
        omega = self._calc_angular_velocity(v, turning_radius)
        if np.abs(omega) > self.max_angular_velocity:
            omega = np.sign(omega)*self.max_angular_velocity
        return self._angular_accel_cap(omega)
    

    def _calc_angular_velocity(v, R):
        return v/R

    
    def _calc_turning_radius(self, iid, v, onset_distance):
        R_min = np.power(v,2) / self.centri_accel
        A = self.approach_factor
        scaled_IID = iid / self.steer_damper * np.pi
        approach_term = 1/np.tan(scaled_IID) + np.sign(iid)*R_min if scaled_IID!= 0 else float('inf')
        avoid_term = -np.sign(iid) * R_min * np.exp(onset_distance - self.body_radius)
        return A*approach_term + (1-A)*avoid_term

"""
    def _approach_A(self, v, iid):
        return self.max_angular_acceleration * iid / self.steer_damper

    def _approach_B(self, v, iid):
        R_min = np.power(v,2) / self.centri_accel
        temp = iid/self.steer_damper * np.pi
        temp = np.pi/2 if temp>np.pi/2 else -np.pi/2 if temp<-np.pi/2 else temp
        if temp==0:
            return 0.
        else:
            R_select = 1/np.tan(temp)+np.sign(iid)*R_min
            return v/R_select

    def _avoid_A(self, v, iid):
        R_min = np.power(v,2) / self.centri_accel
        omega = -np.sign(iid) * (v/R_min)
        return omega

    def _avoid_B(self, v, iid, onset_distance):
        R_min = np.power(v,2) / self.centri_accel
        w = (v/R_min) * np.exp(-1*(onset_distance-self.body_radius))
        omega = -np.sign(iid) * w if onset_distance > self.B*self.body_radius else np.sign(self.kine_cache['omega'])*w
        return omega
"""