"""
Cochlear Transformation using Gammatone Filtering
"""

import numpy as np
from scipy.signal import butter, lfilter, gammatone


class CochlearFilter:
    def __init__(self):
        self.emission_freq = 4.2e4
        self.sampling_freq = 3e5
        self.broadband_spec = {'order':4, 'low':2e4, 'high':8e4}
        self.gammatone_banksize = 1
        self.exp_compression_power = 0.4
        self.lowpass_freq = 1e3


    def transform(self, data):
        # Broad bandpass filter
        b, a = butter(self.broadband_spec['order'],
                      [self.broadband_spec['low'],self.broadband_spec['high']],
                      fs = self.sampling_freq )
        y = lfiter(b,a,data)
        # Gammatone Filter
        b, a = gammatone(self.emission_freq, 'fir', fs=self.sampling_freq)
        y = lfilter(b,a,y)
        # halfwave rectifier
        y[y<0] = 0
        # exponential compression
        y = np.power(y,0.4)
        # lowpass filter
        b,a = butter(2, self.lowpass_freq, 'low', fs=self.sampling_freq)
        y = lfilter(b, a, y)
        return y
