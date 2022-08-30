import numpy as np
from . import Setting


class EarGain:
    def __init__(self, mode=None, **kwargs):
        self.mode = mode.capitalize() if mode is not None else 'Null'
        self.ear_angle = Setting.LEFT_EAR_FIT_ANGLE if self.mode[0]=='L' else Setting.RIGHT_EAR_FIT_ANGLE if self.mode[0]=='R' else kwargs['ear_angle']
        self.rose_k = Setting.NUMBER_OF_PEDALS if 'rose_k' not in kwargs.keys() else kwargs['rose_k']
        self.rose_b = Setting.ROSE_CURVE_B if 'rose_b' not in kwargs.keys() else kwargs['rose_b']
        self.rose_a = 1 - self.rose_b
        self.min_gain = Setting.MIN_GAIN_DB
    

    def get_gain_ratio(self, target_angle, ref_angle, radians=True):
        gain_diff = self._normalized_gain(target_angle,radians=radians)-self._normalized_gain(ref_angle,radians=radians)
        if gain_diff != 0:
            return np.power(10, gain_diff/20)
        else:
            if (self.mode[0]=='L' and target_angle<0) or (self.mode[0]=='R' and target_angle>0):
                gain_diff = self._tail_gain(target_angle,radians=radians)-self._tail_gain(ref_angle,radians=radians)
                return np.power(10,gain_diff/20)
            else:
                return np.power(10, gain_diff/20)


    def _normalized_gain(self, angle, radians=True):
        theta = np.copy(angle) if radians else np.radians(angle)
        gain=self.rose_a  + self.rose_b*np.cos(self.rose_k*theta - self.ear_angle)
        if gain<np.power(10,self.min_gain/20): gain = np.power(10,self.min_gain/20)
        return 20*np.log10(gain)


    def _tail_gain(self, angle, radians=True):
        theta = np.copy(angle) if radians else np.radians(angle)
        gain = 0.1*self.rose_b* np.cos(self.rose_k*theta + self.ear_angle)
        if gain<np.power(10, self.min_gain/20): gain = np.power(10, self.min_gain/20)
        return 20*np.log10(gain)