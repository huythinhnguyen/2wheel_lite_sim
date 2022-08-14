import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__path__)) + '/./Sensors/BatEcho/')

class Avoid:
    def __init__(self, DISTANCE_REFERENCE=None):
        if DISTANCE_REFERENCE is None:
            raise ValueError('pass Compressors.Subsample.DISTANCE_REFERENCE to DISTANCE_REFERENCE')
        self.DISTANCE_REFERENCE = DISTANCE_REFERENCE
        self.window_size=10


    def _get_onset_distance(self,inp):
        if type(inp)==tuple or type(inp)==list: left, right = inp
        elif type(inp)==dict: left, right = inp.values()
        ### COME BACK HERE!!!!


class Approach:
    def __init__(self):
        pass


