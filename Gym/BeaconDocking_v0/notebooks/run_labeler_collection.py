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

from Gym.BeaconDocking_v0 import ConfigOverwite

ConfigOverwite.overwrite_config(controlconfig, sensorconfig)

RUN_ID = int(input('Enter Run ID = '))
APPROACH_LIKELIHOOD = float(input('Enter Approach Likelihood (0.0-1.0) = '))
MARGIN = 1.7
JITTER_LEVEL = 2
TIME_LIMIT = 10_000
NUMBER_OF_EPISODES = 3_000
COMPRESSED_SIZE = len(sensorconfig.COMPRESSED_DISTANCE_ENCODING)
#RAW_SIZE = len(sensorconfig.DISTANCE_ENCODING)
MAX_RUN = 4
# get today date dot separated format MM.DD.YY
DATE = time.strftime("%m.%d.%y")

N_LAST_STEPS = 200
SAFE_STEPS = 5

def echo_dict_to_numpy(dict):
    ls = []
    for data in dict.values(): ls.append(data.reshape(-1,))
    return np.concatenate(ls)


def echo_numpy_to_dict(array):
    dict = {
        'left': array[:COMPRESSED_SIZE].reshape(-1,),
        'right': array[COMPRESSED_SIZE:2*COMPRESSED_SIZE].reshape(-1,),
    }
    return dict


def remove_last_step(compresses, poses):
    compresses = compresses[:-1]
    poses = poses[:-1]
    return compresses, poses

def flip_action(action):
    if action==0: return 1
    if action==1: return 0
    return action

def check_status(pose, inview, beacons, objects):
    if Helper.dockingCheck(pose, beacons=beacons):
        return 'docked'
    if Helper.collisionCheck(inview, 'plant') or Helper.collisionCheck(inview, 'pole'):
        return 'hit'
    return 'safe'

def replay(compresses, poses, actions, zones, vs, omegas, 
            cls, objects, beacons, safe_steps=SAFE_STEPS):
    i = 0
    controller = AvoidApproach()
    render = Render()
    state = State(pose=poses[-1], dt=controlconfig.CHIRP_RATE)
    
    while True:
        next_poses = np.asarray([], dtype=np.float32).reshape(0,3)
        next_compresses = np.asarray([], dtype=np.float32).reshape(0,COMPRESSED_SIZE*2)
        next_actions, next_zones, next_vs, next_omegas = [], [], [], []
        poses, compresses, actions, zones, vs, omegas = poses[:-i], compresses[:-i], actions[:-i], zones[:-i], vs[:-i], omegas[:-i]
        pose, compressed, action, zone = (poses[-1], echo_numpy_to_dict(compresses[-1]), flip_action(actions[-1]), -1)
        actions[-1]=action
        zones[-1]=zone

        controller.kine_cache['v'] = vs[-2]
        controller.kine_cache['omega'] = omegas[-2]
        v, omega = controller.get_kinematic(compressed, approach_factor=action)
        vs[-1] = v
        omegas[-1] = omega
        state.pose = np.copy(pose)
        state.update_kinematic(kinematic=[v, omega])
        state.update_pose()
        next_pose = np.copy(state.pose)
        next_compressed = render.run(pose=next_pose, objects=objects)
        inview = render.cache['inview']
        status = check_status(pose=next_pose, inview=inview, beacons=beacons, objects=objects)
        if status=='docked':
            return compresses, poses, actions, zones, vs, omegas
        if status=='hit':
            i+=1
            continue
        if status=='safe':
            for _ in range(safe_steps+i):
                next_poses = np.vstack((next_poses, next_pose.reshape(1,3)))
                next_compresses = np.vstack((next_compresses, echo_dict_to_numpy(next_compressed).reshape(1,-1)))
                action, zone = Helper.behavior(pose=next_pose, beacons=beacons, classifier=cls)
                next_actions.append(action)
                next_zones.append(zone)
                v, omega = controller.get_kinematic(next_compressed, approach_factor=action)
                vs.append(v)
                omegas.append(omega)
                state.update_kinematic(kinematic=[v, omega])
                state.update_pose()
                next_pose = np.copy(state.pose)
                next_compressed = render.run(pose=next_pose, objects=objects)
                inview = render.cache['inview']
                status = check_status(pose=next_pose, inview=inview, beacons=beacons, objects=objects)
                if status=='docked':
                    break
                if status=='hit':
                    i+=1
                    continue
            break

    compresses = np.vstack((compresses, next_compresses))
    poses = np.vstack((poses, next_poses))
    actions = actions + next_actions
    zones = zones + next_zones
    vs = vs + vs
    omegas = omegas + omegas
    return compresses, poses, actions, zones, vs, omegas


def get_N_last_steps(compresses, poses, actions, zones, result, N=N_LAST_STEPS):
    if result=='docked':
        compresses = compresses[-N:]
        poses = poses[-N:]
        actions = actions[-N:]
        zones = zones[-N:]
    if result=='hit':
        compresses, poses, actions, zones = replay(compresses, poses, actions, zones)
    return compresses, poses, actions, zones

if __name__=='__main__':
    obstacles = Helper.box_builder('')
    cls = load('dockingZone_classifier.joblib')
    compresses_ls, envelopes_ls, poses_ls, actions_ls, zones_ls= [],[],[],[],[]
    episode = 0
    while episode < NUMBER_OF_EPISODES:
        init_pose, beacons = Helper.initializer(jit=JITTER_LEVEL)
        objects = Helper.concatenate_beacons(beacon_objs=Helper.beacons2objects(beacons), objects=obstacles)
        pose = np.copy(init_pose)
        render = Render()
        state = State(pose=pose, dt=1/controlconfig.CHIRP_RATE)
        controller = AvoidApproach()
        actions = []
        dockingZones = []
        poses = np.copy(pose).reshape(1,3)
        compresses = np.asarray([]).reshape(0,2*COMPRESSED_SIZE)
        #envelopes = np.asarray([]).reshape(0, 2*RAW_SIZE)
        #episode_ended = False
        result = 'out'
        for _ in range(TIME_LIMIT):
            compressed = render.run(pose, objects)
            compresses = np.vstack((compresses, echo_dict_to_numpy(compressed).reshape(1,-1)))
            #envelopes = np.vstack((envelopes, np.concatenate([render.cache['left_envelope'].reshape(-1,),
            #                            render.cache['right_envelope'].reshape(-1,)]) ))
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
            #envelopes_ls.append(envelopes)
            poses_ls.append(poses)
            actions_ls.append(actions)
            zones_ls.append(dockingZones)
            print(f'Episode {episode} completed')
        if result == 'hit' or result == 'out':
            print(f'Episode {episode+1} failed')
        
        if episode%100 == 0 and episode != 0:
            df = pd.DataFrame({'compresses': compresses_ls, 'poses': poses_ls, 'actions': actions_ls, 'zones': zones_ls})
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
        # remove the run_# files
        for i in range(1,MAX_RUN+1): os.remove(f'./labeled_echo_data/run_{i}.pkl')
        df.to_pickle(f'./labeled_echo_data/run_ApproachProb_{APPROACH_LIKELIHOOD}_{DATE}.pkl')

    # Print out completion message
    print('Collection completed')
        
