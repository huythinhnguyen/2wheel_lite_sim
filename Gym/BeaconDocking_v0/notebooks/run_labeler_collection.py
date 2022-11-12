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
MAX_RUN = 4
# get today date dot separated format MM.DD.YY
DATE = time.strftime("%m.%d.%y")


def echo_dict_to_numpy(dict):
    ls = []
    for data in dict.values(): ls.append(data.reshape(-1,))
    return np.concatenate(ls)


if __name__=='__main__':
    obstacles = Helper.box_builder('');
    cls = load('dockingZone_classifier.joblib')
    compresses_ls, envelopes_ls, poses_ls, actions_ls, zones_ls= [],[],[],[],[]
    episode = 0
    while episode < NUMBER_OF_EPISODES:
        init_pose, beacons = Helper.initializer(jit=JITTER_LEVEL)
        objects = Helper.concatenate_beacons(beacon_objs=Helper.beacons2objects(beacons), objects=obstacles)
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
        #episode_ended = False
        result = 'out'
        for _ in range(TIME_LIMIT):
            compressed = render.run(pose, objects)
            compresses = np.vstack((compresses, echo_dict_to_numpy(compressed).reshape(1,-1)))
            envelopes = np.vstack((envelopes, np.concatenate([render.cache['left_envelope'].reshape(-1,),
                                        render.cache['right_envelope'].reshape(-1,)]) ))
            docked = Helper.dockingCheck(pose, beacons=beacons)
            if docked:
                result = 'docked'
                #episode_ended = True
                break
            inview = render.cache['inview']
            if Helper.collision_check(inview, 'plant') or Helper.collision_check(inview, 'pole'):
                result = 'hit'
                #episode_ended = True
                break
            action, zone = Helper.behavior(pose, beacons=beacons, classifier=cls)
            actions.append(action)
            dockingZones.append(zone)
            v, omega = controller.get_kinematic(compressed, approach_factor=action)
            state.update_kinematic(kinematic=[v, omega])
            state.update_pose()
            pose = np.copy(state.pose)
            poses = np.vstack((poses, pose.reshape(1,3)))

        if result == 'docked':
            episode += 1
            compresses_ls.append(compresses)
            envelopes_ls.append(envelopes)
            poses_ls.append(poses)
            actions_ls.append(actions)
            zones_ls.append(dockingZones)
            print(f'Episode {episode} completed')
        if result == 'hit' or result == 'out':
            print(f'Episode {episode+1} failed')
        
        if episode%100 == 0 and episode != 0:
            df = pd.DataFrame({'compresses': compresses_ls, 'envelopes': envelopes_ls, 'poses': poses_ls, 'actions': actions_ls, 'zones': zones_ls})
            # if labeled_echo_data folder does not exist, create it
            if not os.path.exists('labeled_echo_data'): os.makedirs('labeled_echo_data')
            df.to_pickle(f'./labeled_echo_data/run_{RUN_ID}.pkl')

    if RUN_ID==MAX_RUN:
        time.sleep(1800)
        for i in range(1,MAX_RUN+1):
            if i==1: 
                df = pd.read_pickle(f'./labeled_echo_data/run_{i}.pkl')
                continue
            df_temp = pd.read_pickle(f'./labeled_echo_data/run_{i}.pkl')
            df = pd.concat([df, df_temp], ignore_index=True)
        df.to_pickle(f'./labeled_echo_data/run_ApproachProb{APPROACH_LIKELIHOOD}_{DATE}.pkl')

    # Print out completion message
    print('Collection completed')
        
