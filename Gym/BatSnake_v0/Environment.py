from importlib.resources import path
import numpy as np
import pathlib
import os
import sys

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from . import Helper as help
if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(pathlib.Path(os.path.abspath(__file__)).parents[2])

from Sensors.BatEcho import Spatializer
from Control.SensorimotorLoops.BatEcho import AvoidApproach
from Sensors.BatEcho import Setting as sensorconfig
from Control.SensorimotorLoops import Setting as controlconfig
from Simulation.Motion import State

class DiscreteAction(py_environment.PyEnvironment):
    def __init__(self, init_pose, time_limit):
        self.locomotion = State(pose = init_pose, dt=1/controlconfig.CHIRP_RATE)
        self.sensor = Spatializer.Render()
        self.controller = AvoidApproach()

        self.objects = help.maze_builder(mode='box')

        self._action_spec=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec=array_spec.BoundedArraySpec(shape=(100,), dtype=np.float64, minimum=0, name='observation')
        self.echoes = self.sensor.run(pose=self.locomotion.pose, objects=self.objects)
        self._state = np.asarray(list(self.echoes.values())).reshape(-1,)
        self._episode_ended = False
        self.time_limit = time_limit
        self.cache = {'init_pose': init_pose}
        

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    
    def _step(self, action, **kwargs):
        reward = 0
        if self._episode_ended:
            return self._reset()
        # Open your eyes, see where you are
        if help.collision_check(self.sensor.cache['inview'], sensorconfig.OBJECTS_DICT['plant']):
            reward += help.reward_function(hit='plant')
            self._episode_ended = True
        elif help.collision_check(self.sensor.cache['inview'], sensorconfig.OBJECTS_DICT['pole']):
            reward += help.reward_function(hit='pole')
        # Move according to action
        v, omega = self.controller.get_kinematic(self.echoes, approach_factor=action)
        self.locomotion.update_kinematic(kinematic=[v, omega])
        self.locomotion.update_pose()
        self.echoes = self.sensor.run(pose=self.locomotion.pose, objects=self.objects)
        self._state = np.asarray(list(self.echoes.values())).reshape(-1,)
        # 
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    
    def _reset(self, **kwargs):
        self._episode_ended=False
        init_pose = kwargs['init_pose'] if 'init_pose' in kwargs.keys() else self.cache['init_pose']
        self.locomotion = State(pose=init_pose, dt=1/controlconfig.CHIRP_RATE)
        self.sensor = Spatializer()
        self.controller = AvoidApproach()

        return ts.restart(self._state)

    
