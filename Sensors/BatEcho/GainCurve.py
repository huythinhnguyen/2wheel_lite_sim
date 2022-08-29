from re import L
import numpy as np
from . import Setting


class EarGain:
    def __init__(self, mode=None, **kwargs):
        self.mode = mode.capitalize() if mode is not None else 'Null'
        self.ear_angle = Setting.LEFT_EAR_FIT_ANGLE if self.mode[0]=='L' else Setting.RIGHT_EAR_FIT_ANGLE if self.mode[0]=='R' else kwargs['ear_angle']
        self.rose_k = Setting.NUMBER_OF_PEDALS if 'rose_k' not in kwargs.keys() else kwargs['rose_k']
        self.rose_b = Setting.ROSE_CURVE_B if 'rose_b' not in kwargs.keys() else kwargs['rose_b']
        self.rose_a = 1 - self.rose_b
        

    def get_gain_ratio(self, target_angle, ref_angle, radians=True):
        target_gain = self._normalized_gain(target_angle, radians=radians)
        ref_gain = self._normalized_gain(ref_angle, radians=radians)
        return np.divide(target_angle, ref_angle)
    
    
    def _normalized_gain(self, angle, radians=True):
        theta = np.copy(angle) if radians else np.radians(angle)
        return self.rose_a  + self.rose_b*np.cos(self.rose_k*theta - 2*self.ear_angle)
