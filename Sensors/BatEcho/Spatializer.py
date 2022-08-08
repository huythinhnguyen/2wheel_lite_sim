"""
Spatializer reference sounds into scence when given locations of sound sources
"""

import numpy as np
from . import Cochlear
import sys


SAMPLE_FREQ = 3e5
SPEED_OF_SOUND = 340
DISTANCE_ENCODING = np.arange(1,7001) * 0.5 (1/sample_freq) * SPEEED_OF_SOUND

class Retriever:
    def __init__(self, ):
        self.pole_starts = _POLE_STARTS
        self.pole_endS   = _POLE_ENDS
        self.polerev_starts = _POLEREV_STARTS
        self.polerev_ends   = _POLEREV_ENDS
        self.plant_starts   = _PLANT_STARTS
        self.plant_ends     = _PLANT_ENDS
        self.bg_sigma = 0.6
        self.bg_mu = 5e-4
        self.emission_encoding = 0.33




    def _get_reference(dist, angle, klass, random=True, index=None):
        path = None
        left_set = np.load(path + 'left.npy')
        right_set = np.load(path + 'right.npy')
        if not random:
            np.random.seed(1)
            if index is not None: selection = index
            else: raise ValueError('must input index when random=False')
        else:
            selection = np.random.randint(len(left_set))
        return left_set[selection,:], right_set[selection,:]
    

            

    def _snip_pole(self, echo, reference_distance):
        start = self.pole_starts[reference_distance]
        end = self.pole_ends[reference_distance]
        start_idx = np.argmin(np.abs(DISTANCE_ENCODING - start))
        end_idx = np.argmin(np.abs(DISTANCE_ENCODING - end))
        left_snip = echo['left'][start_idx:end_idx]
        right_snip = echo['right'[start_idx:end_idx]]
        return left_snip, right_snip


    def _snip_polerev(self, echo, reference_distance):
        start = self.polerev_starts[reference_distance]
        end = self.polerev_ends[reference_distance]
        start_idx = np.argmin(np.abs(DISTANCE_ENCODING - start))
        end_idx = np.argmin(np.abs(DISTANCE_ENCODING - end))
        left_snip = echo['left'][start_idx:end_idx]
        right_snip = echo['right'[start_idx:end_idx]]
        return left_snip, right_snip

    
    def _snip_plant(self, echo, reference_distance):
        start = self.plant_starts[reference_distance]
        end = self.plant_ends[reference_distance]
        start_idx = np.argmin(np.abs(DISTANCE_ENCODING - start))
        end_idx = np.argmin(np.abs(DISTANCE_ENCODING - end))
        left_snip = echo['left'][start_idx:end_idx]
        right_snip = echo['right'[start_idx:end_idx]]
        return left_snip, right_snip
    

class Transformer:
    def __init__(self):
        pass


class Render:
    def __init__(self):
        pass
    


_POLE_STARTS  = {0.25: 0.13, 0.5: 0.38, 0.75: 0.62, 1.0: 0.88, 1.25: 1.12, 1.5: 1.36, 1.75: 1.62, 2.0: 1.87, 2.25: 2.11, 2.5: 2.35}
_POLE_ENDS    = {0.25: 0.47, 0.5: 0.66, 0.75: 0.88, 1.0: 1.1, 1.25: 1.31, 1.5: 1.53, 1.75: 1.76, 2.0: 1.98, 2.25: 2.22, 2.5: 2.48}
_POLREV_STARTS= {0.25: 0.89, 0.5: 0.79, 0.75: 0.94,1.0: 1.13, 1.25: 1.33, 1.5: 1.55, 1.75: 1.78, 2.0: 2.01, 2.25: 2.24}
_POLREV_ENDS  = {0.25: 1.13, 0.5: 0.99, 0.75: 1.15, 1.0: 1.33, 1.25: 1.51, 1.5: 1.71, 1.75: 1.92, 2.0: 2.14, 2.25: 2.35}
_PLANT_STARTS = {0.5: 0.26, 0.75: 0.49, 1.0: 0.76, 1.25: 0.99, 1.5: 1.24, 1.75: 1.49, 2.0: 1.74, 2.25: 1.91, 2.5: 2.24}
_PLANT_ENDS   = {0.5: 1.29, 0.75: 1.35, 1.0: 1.57, 1.25: 1.71, 1.5: 1.76, 1.75: 1.88, 2.0: 2.38, 2.25: 2.47, 2.5: 2.72}
