import os
import sys

import numpy as np
from matplotlib import pyplot as plot

class Agent:
    def __init__(self):
        self.arrow_color = 'tab:brow'
        self.dot_color = 'tab:orange'
        self.arrow_length = 1.0
        self.arrow_aspect = 0.01
        self.arrow_width = self.arrow_length * self.aspect


    def _quiver(self,ax,poses):
        if poses.shape[1] != 3:
            poses.reshape(-1,3)

        bbox = ax.get_window_extent()
            
        ax.quiver(poses[:,0], poses[:,1], self.arrow_length*np.cos(poses[:,2]), self.arrow_length*np.sin(poses[:,2]),
                  pivot='tail', scale_units='xy', scale=1.0, width = self.arrow_width, color=self.arrow_collor)
        ax.scatter(poses[:,0], poses[:,1], s=0.1*min(bbox.widht, bbox.height))
