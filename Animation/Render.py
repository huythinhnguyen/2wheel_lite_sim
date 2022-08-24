import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

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


    def save(self, filename, filetype='svg', filepath=None, reset=True):
        if filepath is not None:
            self.filepath = filepath
        saveas = os.path.join(filepath, filename+'.'+filetype)
        self.fig.savefig(saveas)
        plt.close()
        if reset: self._reset()
        

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
        self.objects = {}
        

    def render(self, objects, repeat=False, show=False, fps=60):
        self.objects = objects
        self.anim = FuncAnimation(self.fig, self._animate, init_func=None, interval=int(1000/fps),
                                  frames=range(len(self.objects['agent'])), repeat=repeat, repeat_delay=100)
        if show: self.fig.show()
        

    def save(self, filename, filetype='gif', fps=60, filepath=None, dpi=300, reset=True):
        if filepath is not None:
            self.filepath = filepath
        saveas = os.path.join(filepath, filename+'.'+filetype)
        writer = PillowWriter(fps=fps)
        self.anim.save(saveas, writer=writer, dpi=dpi)
        if reset: self._reset()
        

    def _fig_init(self):tsize=16, sparseness=100, flipcourse=True):
        self.ax.set_ylabel(labels[1])


    def _animate(self,i):
        self.ax.clear()
        self.ax = _render_plant(self.ax, self.objects['plant'])
        self.ax = _render_food(self.ax, self.objects['food'][i].reshape(-1,2))
        self.ax = _render_agent(self.ax, self.objects['agent'][:i].reshape(-1,3),plotcourse=True, plotarrow=False)
        self.ax = _render_agent(self.ax, self.objects['agent'][i].reshape(-1,3), plotcourse=False,plotarrow=True)


    def _reset(self):
        garbage = [self.fig, self.ax, self.anim]
        del self.fig, self.ax, self.anim
        del garbage
        self.fig, self.ax = plt.subplots(figsize=(self.figsize,self.figsize))
        plt.rcParams.update({'font.size': self.fontsize})
        self.objects = {}


def _render_agent(ax, poses, plotcourse=True, plotarrow=True, **kwarg):
    agent = ObjUtils.Agent()
    if 'flipcourse' in kwarg.keys():
        if kwarg['flipcourse']: poses = np.flipud(poses)
    if plotcourse: ax = agent._course(ax, poses)
    quivers = poses[::kwarg['sparseness']] if 'sparseness' in kwarg.keys() else poses
    if plotarrow: ax = agent._quiver(ax, quivers)
    return ax


def _render_food(ax, foods, **kwarg):
    food = ObjUtils.Food()
    ax = food.plot(ax, foods)
    return ax


def _render_plant(ax, plants, **kwarg):
    plant = ObjUtils.Plant()
    if 'leavedensity' in kwarg.keys():
        plant.leave_density = kwarg['leavedensity']
    ax = plant.plot(ax, plants)
    return ax


_OBJECTS_TYPES = {'plant': _render_plant, 'food': _render_food, 'agent': _render_agent}
