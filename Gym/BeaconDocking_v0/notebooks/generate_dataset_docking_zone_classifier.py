import numpy as np
import os
import pathlib
import sys

if pathlib.Path(os.getcwd()).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.getcwd()).parents[2]))

from Sensors.BatEcho.Spatializer import Render
from Sensors.BatEcho import Setting as sensorconfig
from Gym.BeaconDocking_v0 import Helper
from Arena import Builder

from Control.SensorimotorLoops.BatEcho import AvoidApproach, Avoid
from Simulation.Motion import State
from Control.SensorimotorLoops import Setting as controlconfig

import time

OUTBOUND_LIMIT = 10.
RANDOM_LIMIT = 8.
DEFAULT_BEACON = [0., 0., 0.]
N = 100_000
RUN_ID = input('ENTER RUN_ID')

def outbound(pose, limit=OUTBOUND_LIMIT):
    if (np.sum(np.power(pose[:2],2)) > limit**2): return True
    else: return False


def tooCloseToCenter(pose, limit=0.5):
    if (np.sum(np.power(pose[:2],2)) < limit**2): return True
    else: return False


def init_random_pose(limit=RANDOM_LIMIT):
    x, y = np.random.uniform(low=-limit, high=limit, size=2)
    yaw = np.random.uniform(low = -np.pi, high=np.pi)
    return np.asarray([x, y, yaw], dtype=np.float32).reshape(3,)
    

def run_episode(init_pose, beacon_pose, outbound_func=outbound, time_limit=1000):
    if type(init_pose) is list: pose = np.asarray(init_pose, dtype=np.float32).reshape(3,)
    elif type(init_pose) is np.ndarray: pose = np.copy(init_pose).reshape(3,)
    else: raise ValueError('init_pose must be a list or numpy array')
    
    if type(beacon_pose) is list: beacons = np.asarray(beacon_pose, dtype=np.float32).reshape(1,3)
    elif type(beacon_pose) is np.ndarray: pose = np.copy(beacon_pose).reshape(1,3)
    else: raise ValueError('beacon_pose must be a list or numpy array')
    
    objects = Helper.beacons2objects(beacons=beacons)

    controller = AvoidApproach(approach_factor=1)
    render = Render()
    state = State(pose=pose, dt=1/controlconfig.CHIRP_RATE)

    for _ in range(time_limit):
        compressed = render.run(pose, objects)
        docked = Helper.dockingCheck(pose, beacons)
        if docked: return 'docked'
        if Helper.collision_check(render.cache['inview'], 'plant'): return 'hit'
        if Helper.collision_check(render.cache['inview'], 'pole'):  return 'hit'
        if outbound_func(pose): return 'out'
        v, omega = controller.get_kinematic(compressed, approach_factor=1)
        state.update_kinematic(kinematic=[v, omega])
        state.update_pose()
        pose = state.pose
    return 'out'


if __name__ == "__main__":
    print('RUN ID = '+RUN_ID)
    tic = time.time()
    X = np.zeros((N,3))
    Y = np.zeros((N,1))
    for i in range(N):
        init_pose = init_random_pose()
        while outbound(init_pose) or tooCloseToCenter(init_pose):
            init_pose = init_random_pose()
        X[i] = init_pose
        outcome = run_episode(init_pose, DEFAULT_BEACON)
        Y[i] = 1 if outcome=='docked' else 0 if outcome=='out' else -1

        if (i+1)%1_000==0:
            toc = time.time()
            print('Progress >> {0}/{1} \t time={2}h'.format(i+1,N, np.round((toc-tic)/3600,2) ))
            np.savez('dockingZone_dataset_'+RUN_ID+'.npz', X=X, Y=Y)
    np.savez('dockingZone_dataset_'+RUN_ID+'.npz', X=X, Y=Y)

