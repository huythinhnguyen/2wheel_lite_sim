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
        onset_index = np.argmax( (left - self.left_bg) > self.onset_threshold )
        onset_index = np.min([onset_index, np.argmax((right - self.right_bg) > self.onset_threshold)])
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
        self.max_linear_velocity = config.MAX_LINEAR_VELOCITY if self.bot_convert else config.ROBOT_MAX_LINEAR_VELOCITY
        self.max_angular_velocity= config.MAX_ANGULAR_VELOCITY if self.bot_convert else config.ROBOT_MAX_ANGULAR_VELOCITY
        self.linear_velo_offset = config.LINEAR_VELOCITY_OFFSET
        self.A = config.DECELERATION_FACTOR
        self.K = 0.1

    
    def get_kinematic(self, input_echoes):
        cues = self.get_cues(input_echoes)
        v = self._get_linear_velocity(cues)
        omega = self._get_angular_velocity(cues)
        return v, omega


    def _get_linear_velocity(self,cues):
        distance = cues['onset_distance']
        v = (self.max_linear_velocity - self.linear_velo_offset)*\
            (1 - np.power((1-self.K*self.A*distance),1/self.K-1)) \
            + self.linear_velo_offset
        if v>self.max_linear_velocity: v=self.max_linear_velocity
        return v
    

    def _get_angular_velocity(self,cues):
        iid = cues['IID']
        distance = cues['distance']
        omega = None # input function here
        return omega

    
class Approach(Cue):
    def __init__(self, background=None):
        super().__init__(background)
        self.bot_convert = config.ROBOT_CONVERSION
        self.max_linear_velocity = config.MAX_LINEAR_VELOCITY if self.bot_convert else config.ROBOT_MAX_LINEAR_VELOCITY
        self.max_angular_velocity= config.MAX_ANGULAR_VELOCITY if self.bot_convert else config.ROBOT_MAX_ANGULAR_VELOCITY
        self.linear_velo_offset = config.LINEAR_VELOCITY_OFFSET
        self.A = config.DECELERATION_FACTOR
        self.K = 0.1
    
    def get_kinematic(self, input_echoes):
        cues = self.get_cues(input_echoes)
        v = self._get_linear_velocity(cues)
        omega = self._get_angular_velocity(cues)
        return v, omega


    def _get_linear_velocity(self,cues):
        distance = cues['onset_distance']
        v = (self.max_linear_velocity - self.linear_velo_offset)*\
            (1 - np.power((1-self.K*self.A*distance),1/self.K-1)) \
            + self.linear_velo_offset
        if v>self.max_linear_velocity: v=self.max_linear_velocity
        return v
    

    def _get_angular_velocity(self,cues):

        return omega
    
