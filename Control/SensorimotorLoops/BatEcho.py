import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__path__)) + '/./Sensors/BatEcho/')

from Sensors.BatEcho import Setting, Spatializer

render = Spatializer.Render()
BACKGROUND = render.get_compressed([])
garbage = [render]
del render
del garbage


class Cue:
    def __init__(self, background = None):
        self.DISTANCE_REFERENCE = Setting.COMPRESSED_DISTANCE_ENCODING
        self.window_size=10
        self.emission_encoding = Setting.EMISSION_ENCODING
        
        self.onset_threshold = 1 * Setting.QUIET_THRESHOLD
        if background is None: self.left_bg, self.right_bg = BACKGROUND.values()
        else:
            if type(background)==tuple or type(background)==list: self.left_bg, self.right_bg = background
            elif type(background)==dict: self.left_bg, self.right_bg = background.values()
            

    def _get_onset_distance(self,inp):
        if type(inp)==tuple or type(inp)==list: left, right = inp
        elif type(inp)==dict: left, right = inp.values()
        onset_index = np.argmax( (left - self.left_bg) > self.onset_threshold )
        onset_index = np.min([onset_index, np.argmax((right - self.right_bg) > self.onset_threshold)])
        onset_distance = self.DISTANCE_REFERENCE[onset_distance]
        onset = {'distance': onset_distance, 'index': onset_index}
        return onset
        

class Avoid(Cue):
    def __init__(self):
        pass


class Approach(Cue):
    def __init__(self):
        pass


