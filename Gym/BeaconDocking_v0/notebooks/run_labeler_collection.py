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
MARGIN = 1.5
JITTER_LEVEL = 2
DISTANCE_TO_CRASH = 0.6
TIME_LIMIT = 5_000
NUMBER_OF_EPISODES = 1_000
COMPRESSED_SIZE = len(sensorconfig.COMPRESSED_DISTANCE_ENCODING)
#RAW_SIZE = len(sensorconfig.DISTANCE_ENCODING)
MAX_RUN = 4
# get today date dot separated format MM.DD.YY
DATE = time.strftime("%m.%d.%y")
DOCKING_ZONE_CLASSIFIER_PATH = 'dockingZone_classifier_macOS.joblib'
N_LAST_STEPS = 300
SAFE_STEPS = 30


def run_1_primer_episode(time_limit=TIME_LIMIT):
    # initialize the episode
    obstacles = Helper.box_builder('')
    cls = load(DOCKING_ZONE_CLASSIFIER_PATH)
    init_pose, beacons = Helper.initializer(jit=JITTER_LEVEL)
    objects = Helper.concatenate_beacons(beacon_objs=Helper.beacons2objects(beacons), objects=obstacles)
    # initialize data containers
    poses, compresses, zones, actions, kine_caches, vs, omegas = [], [], [], [], [], [], []
    # initialize the simulation-render and controller:
    pose = np.copy(init_pose)
    render = Render()
    state = State(pose=np.copy(pose), dt=1/controlconfig.CHIRP_RATE)
    controller = AvoidApproach()
    result='out'

    for _ in range(time_limit):
        poses.append(pose)
        compressed = render.run(pose=pose, objects=objects)
        compresses.append(compressed)
        docked = Helper.dockingCheck(pose, beacons=beacons)
        if docked:
            result = 'docked'
            _,_ = poses.pop(), compresses.pop()
            break
        inview = render.cache['inview']
        if Helper.collision_check(inview, 'plant') or Helper.collision_check(inview, 'pole'):
            result = 'hit'
            _,_ = poses.pop(), compresses.pop()
            break
        action, zone = Helper.behavior(pose, beacons=beacons, classifier=cls)
        zones.append(zone)
        actions.append(action)
        kine_caches.append(controller.kine_cache)
        v, omega = controller.get_kinematic(compressed, approach_factor=action)
        vs.append(v)
        omegas.append(omega)
        state.update_kinematic(kinematic=[v, omega])
        state.update_pose()
        pose = np.copy(state.pose)
    return result, beacons, objects, poses, compresses, zones, actions, kine_caches, vs, omegas


def replay(beacons, objects, poses, compresses, actions, kine_caches, dist2crash=DISTANCE_TO_CRASH):
    result = 'hit'
    crashsite = poses[-1]
    kth = 0
    while (result=='hit'):
        kth -= 1
        if actions[kth]==0: continue
        for N in [abs(kth)-1]+list(range(abs(kth))):
            replay_poses, replay_compresses = [poses[kth]], [compresses[kth]]
            replay_zones, replay_actions = [-1], [0]
            replay_kine_caches, replay_vs, replay_omegas = [kine_caches[kth]], [], []
            cls = load(DOCKING_ZONE_CLASSIFIER_PATH)

            # take the very first step (the course correction step)
            pose = np.copy(replay_poses[0])
            render = Render()
            state = State(pose=np.copy(pose), dt=1/controlconfig.CHIRP_RATE)
            controller = AvoidApproach()
            controller.kine_cache.update(replay_kine_caches[0])
            result='out'
            action = replay_actions[0]
            v, omega = controller.get_kinematic(input_echoes=replay_compresses[0], approach_factor=action)
            replay_vs.append(v)
            replay_omegas.append(omega)
            state.update_kinematic(kinematic=[v, omega])
            state.update_pose()
            pose = np.copy(state.pose)
            i = 0
            N_replay = SAFE_STEPS
            while (i<N_replay):
                replay_poses.append(pose)
                compressed = render.run(pose=pose, objects=objects)
                replay_compresses.append(compressed)
                docked = Helper.dockingCheck(pose, beacons=beacons)
                if docked:
                    result = 'docked'
                    _,_ = replay_poses.pop(), replay_compresses.pop()
                    break
                inview = render.cache['inview']
                if Helper.collision_check(inview, 'plant') or Helper.collision_check(inview, 'pole'):
                    result = 'hit'
                    _,_ = replay_poses.pop(), replay_compresses.pop()
                    break
                if i<N: action, zone=0,-1
                else: action, zone = Helper.behavior(pose, beacons=beacons, classifier=cls)

                replay_zones.append(zone)
                replay_actions.append(action)
                replay_kine_caches.append(controller.kine_cache)
                v, omega = controller.get_kinematic(compressed, approach_factor=action)
                replay_vs.append(v)
                replay_omegas.append(omega)
                state.update_kinematic(kinematic=[v, omega])
                state.update_pose()
                pose = np.copy(state.pose)
                i += 1
                if i==N_replay:
                    current_dist2crash = np.linalg.norm(pose[:2]-crashsite[:2])
                    if current_dist2crash < dist2crash: N_replay += SAFE_STEPS
            if N==(abs(kth)-1):
                if result=='hit': break
                else: continue
            if result!='hit': break
    return kth, N, replay_poses, replay_compresses, replay_zones, replay_actions, replay_kine_caches, replay_vs, replay_omegas


def replace_with_replay(kth, poses, compresses, zones, actions, kine_caches, vs, omegas,
                        replay_poses, replay_compresses, replay_zones, replay_actions, replay_kine_caches, replay_vs, replay_omegas):
    poses[kth:] = replay_poses
    compresses[kth:] = replay_compresses
    zones[kth:] = replay_zones
    actions[kth:] = replay_actions
    kine_caches[kth:] = replay_kine_caches
    vs[kth:] = replay_vs
    omegas[kth:] = replay_omegas
    return poses, compresses, zones, actions, kine_caches, vs, omegas


def compile_data_into_array(beacons, objects, poses, compresses, zones, actions, vs, omegas, nsteps=N_LAST_STEPS):
    # place each data into a numpy array
    # if data len is > nsteps, truncate the beginning
    # else do nothing

    if type(beacons) is not np.ndarray: beacons = np.array(beacons).reshape(-1,3)
    if type(objects) is not np.ndarray: objects = np.array(objects).reshape(-1,3)
    poses = np.asarray(poses).reshape(-1,3)
    if len(poses)>nsteps: poses = poses[-nsteps:]
    # compresses is a list of dictionaries with keys 'left' and 'right'
    # convert compresses into a list of numpy array which is concatenated from the 1-D array from 'left' and 'right'
    # do this without using an explicit for loop
    if len(compresses)>nsteps: compresses = compresses[-nsteps:]
    for i in range(len(compresses)):
        compresses[i] = np.concatenate([compresses[i]['left'], compresses[i]['right']])
    compresses = np.asarray(compresses).reshape(-1, 2*COMPRESSED_SIZE)
    if len(zones)>nsteps: zones = zones[-nsteps:]
    zones = np.asarray(zones)
    if len(actions)>nsteps: actions = actions[-nsteps:]
    actions = np.asarray(actions)
    if len(vs)>nsteps: vs = vs[-nsteps:]
    vs = np.asarray(vs)
    if len(omegas)>nsteps: omegas = omegas[-nsteps:]
    omegas = np.asarray(omegas)
    return beacons, objects, poses, compresses, zones, actions, vs, omegas


def main():
    beacons_ls, objects_ls, poses_ls, compresses_ls, zones_ls, actions_ls, vs_ls, omegas_ls = [], [], [], [], [], [], [], []
    episode = 0
    while episode < NUMBER_OF_EPISODES:
        result, beacons, objects, poses, compresses, zones, actions, kine_caches, vs, omegas = run_1_primer_episode()
        if result=='out': continue
        if result=='hit':
            kth, N, replay_poses, replay_compresses, replay_zones, replay_actions, replay_kine_caches, replay_vs, replay_omegas = replay(
                beacons=beacons, objects=objects, poses=poses, compresses=compresses, actions=actions, kine_caches=kine_caches
            )
            poses, compresses, zones, actions, kine_caches, vs, omegas = replace_with_replay(
                kth, poses, compresses, zones, actions, kine_caches, vs, omegas,
                replay_poses, replay_compresses, replay_zones, replay_actions, replay_kine_caches, replay_vs, replay_omegas)
        beacons, objects, poses, compresses, zones, actions, vs, omegas = compile_data_into_array(beacons, objects, poses, compresses, zones, actions, vs, omegas)
        episode += 1
        beacons_ls.append(beacons)
        objects_ls.append(objects)
        poses_ls.append(poses)
        compresses_ls.append(compresses)
        zones_ls.append(zones)
        actions_ls.append(actions)
        vs_ls.append(vs)
        omegas_ls.append(omegas)
        
        print('Episode {}/{} >> result: {}'.format(episode, NUMBER_OF_EPISODES, result))

        if episode%200==0 and episode!=0:
            df = pd.DataFrame({'beacons': beacons_ls,
                                'objects': objects_ls,
                                'poses': poses_ls,
                                'compresses': compresses_ls,
                                'zones': zones_ls,
                                'actions': actions_ls,
                                'vs': vs_ls,
                                'omegas': omegas_ls})
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

    return None

if __name__ == '__main__':
    main()