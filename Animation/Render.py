import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import os
import sys

from . import ObjUtils


class StillImage:
    def __init__(self, figsize=10, objects_types=None, filepath=None, fontsize=16, sparseness=100, flipcourse=True):
        if objects_types is None:
            self.plot_obj_funcs = _OBJECTS_TYPES.values()
            self.objkeys = list(_OBJECTS_TYPES.keys())
        else:
            self.plot_obj_funcs = [_OBJECTS_TYPES[obj_type] for obj_type in objects_types]
            self.objkeys = objects_types
        self.figsize = figsize
        self.fontsize = fontsize
        self.filepath = filepath

        self.sparseness = sparseness
        self.flipcourse = flipcourse

        self.fig, self.ax = plt.subplots(figsize=(self.figsize,self.figsize))
        plt.rcParams.update({'font.size': self.fontsize})

        
    def render(self, objects, labels=['x (m)','y (m)'], leavedensity=None, show=True):
        for i, plotter in enumerate(self.plot_obj_funcs):
            self.ax = plotter(self.ax, objects[self.objkeys[i]],
                              sparseness=self.sparseness,
                              flipcourse=self.flipcourse,
                              leavedensity = leavedensity)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        if show: self.fig.show()


    def save(self, filename, filetype='svg', filepath=None):
        if filepath is not None:
            self.filepath = filepath
        saveas = os.path.join(filepath, filename+'.'+filetype)
        self.fig.savefig(saveas)
        plt.close()
        self._reset()
        

    def _reset(self):
        # garbage collection
        garbage = [self.fig, self.ax]
        del self.fig, self.ax
        del garbage
        self.fig, self.ax = plt.subplots(figsize=(self.figsize,self.figsize))
        plt.rcParams.update({'font.size': self.fontsize})



class Sequences:
    def __init__(self, figsize=10, objects_types=None, filepath=None, fontsize=16, sparseness=100, flipcourse=True):
        if objects_types is None:
            self.plot_obj_funcs = _OBJECTS_TYPES.values()
            self.objkeys = list(_OBJECTS_TYPES.keys())
        else:
            self.plot_obj_funcs = [_OBJECTS_TYPES[obj_type] for obj_type in objects_types]
            self.objkeys = objects_types
        self.figsize = figsize
        self.fontsize = fontsize
        self.filepath = filepath

        self.fig, self.ax = plt.subplots(figsize=(self.figsize,self.figsize))
        plt.rcParams.update({'font.size': self.fontsize})


    def _fig_init(self):
        # create all the set_data thingi
        pass


    def _animate(self,i):
        # pass the animation for step i
        pass
    

def _render_agent(ax, poses, **kwarg):
    agent = ObjUtils.Agent()
    if kwarg['flipcourse']:
        poses = np.flipud(poses)
    ax = agent._course(ax, poses)
    ax = agent._quiver(ax, poses[::kwarg['sparseness']])
    return ax


def _render_food(ax, foods, **kwarg):
    food = ObjUtils.Food()
    ax = food.plot(ax, foods)
    return ax


def _render_plant(ax, plants, **kwarg):
    plant = ObjUtils.Plant()
    if kwarg['leavedensity'] is None:
        pass
    else:
        plant.leave_density = kwarg['leavedensity']
    ax = plant.plot(ax, plants)
    return ax


_OBJECTS_TYPES = {'plant': _render_plant, 'food': _render_food, 'agent': _render_agent}
