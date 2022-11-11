import numpy as np
import os
import pathlib
import sys
import time
from joblib import load
import pandas as pd

if pathlib.Path(os.getcwd()).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.getcwd()).parents[2]))

from Sensors.BatEcho.Spatializer import Render
from Sensors.BatEcho import Setting as sensorconfig
from Gym.BeaconDocking_v0 import Helper
from Arena import Builder

from Control.SensorimotorLoops.BatEcho import AvoidApproach, Avoid
from Simulation.Motion import State
from Control.SensorimotorLoops import Setting as controlconfig

RUN_ID = int(input('Enter Run ID = '))
APPROACH_LIKELIHOOD = 0.3
MARGIN = 1.9
JITTER_LEVEL = 2
TIME_LIMIT = 1_000
NUMBER_OF_EPISODES = 2_000
COMPRESSED_SIZE = len(sensorconfig.COMPRESSED_DISTANCE_ENCODING)
RAW_SIZE = len(sensorconfig.DISTANCE_ENCODING)


def echo_dict_to_numpy(dict):
    ls = []
    for data in dict.values(): ls.append(data.reshape(-1,))
    return np.concatenate(ls)


if __name__=='__main__':
    obstacles = Helper.box_builder('');
    cls = load('dockingZone_classifier.joblib')
    compresses_ls, envelopes_ls, poses_ls, actions_ls, zones_ls= [],[],[],[],[]

    for e in range(NUMBER_OF_EPISODES):
        init_pose, beacons = Helper.initializer(jit=JITTER_LEVEL)
        objects = Helper.concatenate_beacons(beacon_objs=Helper.beacon2objects(beacons), objects=obstacles)
        pose = np.copy(init_pose)
        pose = np.copy(init_pose)
        render = Render()
        state = State(pose=pose, dt=1/controlconfig.CHIRP_RATE)
        controller = AvoidApproach()
        actions = []
        dockingZones = []
        poses = np.copy(pose).reshape(1,3)
        compresses = np.asarray([]).reshape(0,2*COMPRESSED_SIZE)
        envelopes = np.asarray([]).reshape(0, 2*RAW_SIZE)
        episode_ended = False
        result = 'out'
        for _ in range(TIME_LIMIT):
            compressed = render.run(pose, objects)
            compresses = np.vstack((compresses, echo_dict_to_numpy(compressed).reshape(1,-1)))
            envelopes = np.vstack((envelopes, np.concatenate([render.cache['left_envelope'].reshape(-1,),
                                        render.cache['right_envelope'].reshape(-1,)]) ))
            docked = Helper.dockingCheck(pose, beacons=beacons)
            if docked:
                result = 'docked'
                episode_ended = True
                break
            inview = render.cache['inview']
            if Helper.collision_check(inview, 'plant') or Helper.collision_check(inview, 'pole'):
                result = 'hit'
                episode_ended = True
                break
            action, zone = Helper.behavior(pose, beacons

            