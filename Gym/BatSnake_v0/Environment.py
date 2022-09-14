import numpy as np
import pathlib
import os
import sys

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from . import Helper as help
if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parents[2]))

from Sensors.BatEcho import Spatializer
from Control.SensorimotorLoops.BatEcho import AvoidApproach
from Sensors.BatEcho import Setting as sensorconfig
from Control.SensorimotorLoops import Setting as controlconfig
from Simulation.Motion import State


OBSERVATION_SIZE = int(2*sensorconfig.RAW_DATA_LENGTH/sensorconfig.N_SAMPLE)


class DiscreteAction(py_environment.PyEnvironment):
    def __init__(self, time_limit, level=0, phase=0, difficulty=0, mode='box', max_level=1, log=False):
        init_pose = help.spawn_bat(mode, phase)
        self.locomotion = State(pose = init_pose, dt=1/controlconfig.CHIRP_RATE)
        self.sensor = Spatializer.Render()
        self.controller = AvoidApproach()
        self.objects = help.maze_builder(mode=mode)
        self.objects = np.vstack((help.spawn_food(mode, level, difficulty) , self.objects))
        #self.objects = np.vstack((np.asarray([ (help.MAZE_SIZE/2-help.TUNNEL_WIDTH/4),1.,1]).reshape(1,3) , self.objects))
        self._action_spec=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec=array_spec.BoundedArraySpec(shape=(OBSERVATION_SIZE,), dtype=np.float64, minimum=0, name='observation')
        self.echoes = self.sensor.run(pose=self.locomotion.pose, objects=self.objects)
        self._state = np.asarray(list(self.echoes.values())).reshape(-1,)
        self._episode_ended = False
        self.cache = {'time_limit': time_limit, 'init_level': level, 'phase': phase, 'difficulty': difficulty, 'mode': mode, 'max_level': max_level}
        self.step_count = 0
        self.level = level
        self.log = log
        if log:
            self.episode = 0
            self.status = ''
            self.records = {}
        

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    
    def _step(self, action, **kwargs):
        reward = 0
        if self._episode_ended:
            return self._reset()
        # Open your eyes, see where you are
        if help.collision_check(self.sensor.cache['inview'], 'plant'):
            reward += help.reward_function(hit='plant')
            self._episode_ended = True
            if self.log: self.status='hit'
        elif help.collision_check(self.sensor.cache['inview'], 'pole'):
            reward += help.reward_function(hit='pole')
            self.level += 1
        # Move according to action
        v, omega = self.controller.get_kinematic(self.echoes, approach_factor=action)
        self.locomotion.update_kinematic(kinematic=[v, omega])
        self.locomotion.update_pose()
        self.echoes = self.sensor.run(pose=self.locomotion.pose, objects=self.objects)
        self._state = np.asarray(list(self.echoes.values())).reshape(-1,)
        #
        if help.out_of_bound(pose=self.locomotion.pose, mode=self.cache['mode']):
            self._episode_ended = True
            if self.log: self.status = 'outbound'
        #
        self.step_count += 1
        if self.step_count >= self.cache['time_limit'] or self.level >= self.cache['max_level']:
            self._episode_ended = True
            if self.log: self.status = 'success' if reward>0 else 'timeout'
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    
    def _reset(self, **kwargs):
        if self.log:
            print('episode {0} : {1} steps, ({2},{3}) ending pose | '.format( self.episode, self.step_count, *np.round(self.locomotion.pose[:2],2)) + self.status)
            self.status = ''
            if self.episode > 0:
                if 'episode' not in self.records.keys(): self.records['episode'] = [self.episode]
                else: self.records['episode'].append(self.episode)
                if 'steps' not in self.records.keys(): self.records['steps'] = [self.step_count]
                else: self.records['episode'].append(self.episode)
                if 'hit' not in self.records.keys(): self.records['hit'] = 1 if self.status=='hit' else 0
                else: self.records['hit'] += 1 if self.status=='hit' else 0
                if 'success' not in self.records.keys(): self.records['success'] = 1 if self.status=='success' else 0
                else: self.records['success'] += 1 if self.status=='success' else 0
                if 'timeout' not in self.records.keys(): self.records['timeout'] = 1 if self.status=='timeout' else 0
                else: self.records['timeout'] += 1 if self.status=='timeout' else 0
                if 'outbound' not in self.records.keys(): self.records['outbound'] = 1 if self.status=='outbound' else 0
                else: self.records['outbound'] += 1 if self.status=='outbound' else 0
            else: self.records.clear()
            self.episode+=1
        self._episode_ended=False
        for key, val in zip(kwargs.keys(), kwargs.values()):
            self.cache[key] = val
        init_pose = help.spawn_bat(self.cache['mode'], self.cache['phase'])
        self.locomotion = State(pose=init_pose, dt=1/controlconfig.CHIRP_RATE)
        self.sensor = Spatializer.Render()
        self.controller = AvoidApproach()
        self.objects[self.objects[:,2]==sensorconfig.OBJECTS_DICT['pole']]=help.spawn_food(self.cache['mode'],
                                                                                           self.cache['init_level'], 
                                                                                           self.cache['difficulty'])
        #self.objects[0] = np.asarray([(help.MAZE_SIZE/2-help.TUNNEL_WIDTH/4),1.,1])
        self.echoes = self.sensor.run(pose=self.locomotion.pose, objects=self.objects)
        self._state = np.asarray(list(self.echoes.values())).reshape(-1,)
        self.level = self.cache['init_level']
        self.step_count = 0
        return ts.restart(self._state)

