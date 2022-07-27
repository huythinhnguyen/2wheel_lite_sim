import os
import sys

import numpy as np
from matplotlib import pyplot as plot

class Agent:
    def __init__(self):
        self.arrow_color = 'tab:red'
        self.dot_color = 'tab:orange'
        self.arrow_length = 1.0
        self.arrow_aspect = 0.005
        self.arrow_width = self.arrow_length * self.arrow_aspect
        self.arrow_alpha = 0.8
        self.course_color = 'tab:blue'
        self.course_size = 0.2
        self.course_alpha = 0.5


    def _quiver(self,ax,poses):
        if poses.shape[1] != 3:
            poses.reshape(-1,3)
    
        ax.quiver(poses[:,0], poses[:,1], self.arrow_length*np.cos(poses[:,2]), self.arrow_length*np.sin(poses[:,2]),
                  pivot='tail', scale_units='xy', scale=1.0, width = self.arrow_width, color=self.arrow_color, alpha=self.arrow_alpha)
        ax.set_aspect('equal')
        xr = ax.get_xlim()[1] - ax.get_xlim()[0] + 1
        yr = ax.get_ylim()[1] - ax.get_ylim()[0] + 1
        scatter_scale = ax.get_window_extent().width/(max(xr,yr)**2)
        ax.scatter(poses[:,0], poses[:,1], s=20*scatter_scale, alpha=self.arrow_alpha)
        return ax


    def _course(self, ax, poses, mode='plot'):
        if poses.shape[1]!=3:
            poses.reshape(-1,3)
        if mode=='plot' or mode=='p':
            ax.plot(poses[:,0], poses[:,1], linewidth=self.course_size*10, alpha=self.course_alpha)
        elif mode=='scatter' or mode=='s':
            ax.scatter(poses[:,0], poses[:,1], s=self.course_size, alpha=self.course_alpha)
        else:
            raise ValueError('mode need to be either scatter or plot')
        ax.set_aspect('equal')
        return ax


class Food:
    def __init__(self):
        pass


class Plant:
    def __init__(self):
        self.leave_shape = 'v^<>v^<>'
        self.leave_density = 3
        self.leave_alpha = 0.7
        self.leave_color = 'tab:green'
        self.leave_size = 50
        self.plant_width = 0.2


    def plot(self, ax, plants):
        if plants.shape[1] == 3:
            plants = plants[:,:2]
        ax.scatter(plants[:,0], plants[:,1], s=1, marker=',', linewidths=0.0, alpha=self.leave_alpha, c=self.leave_color)
        ax.set_aspect('equal')
        xr = ax.get_xlim()[1] - ax.get_xlim()[0] + 1
        yr = ax.get_ylim()[1] - ax.get_ylim()[0] + 1
        scatter_scale = ax.get_window_extent().width/(max(xr,yr)**2)
        for _ in range(self.leave_density):
            for i in range(len(self.leave_shape)):
                ww = plants + (2*self.plant_width*np.random.rand(plants.shape[0],2)-self.plant_width)
                ax.scatter(ww[:,0], ww[:,1], s=scatter_scale*self.leave_size, marker=self.leave_shape[i], linewidths=0.0, alpha=self.leave_alpha, c=self.leave_color)
                
                
