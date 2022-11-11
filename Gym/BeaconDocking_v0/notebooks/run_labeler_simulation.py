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
NUM_OF_ITERATION = 100
LIST_OF_MARGINS = [1.5, 1.7, 1.9]
JITTER_LEVEL = 2
LIST_OF_THRESHOLDS = np.round(np.arange(0,1.1,0.1),1)
TIME_LIMIT = 1_000
cache = []

def avoid_overwrite(pose, maze_size=Helper.MAZE_SIZE, margin = 2):
    x, y = pose[:2]
    t = 0.5*maze_size - margin
    if (np.abs(x) > t) or (np.abs(y) > t): return True
    return False

def sort_beacons_by_distance(pose, beacons):
    x_sq = np.power(beacons[:,0] - pose[0],2)
    y_sq = np.power(beacons[:,1] - pose[1],2)
    distance_squared = x_sq + y_sq
    sorted_beacons = beacons[np.argsort(distance_squared)]
    return sorted_beacons, np.sqrt(distance_squared)

def beacon_centric_pose_converter(pose, beacon):
    p = np.copy(pose).reshape(1,3)
    # translate -> beacons center and zero zero.
    p[:,:2] = p[:,:2] - beacon[:2].reshape(-1,2)
    # rotate -> beacon direction yaw = 0
    theta = -beacon[2]
    rotmat = np.asarray([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], dtype=np.float32).reshape(2,2)
    p[:,:2] = np.matmul(rotmat, p[:,:2].T).T
    p[:,2] = Builder.wrap2pi(p[:,2] + theta)
    return p

def behavior(pose, beacons, avoid_overwrite_func, sort_beacons_by_distance_func, beacon_centric_pose_convert_func, cls, random_approach_threshold=0.8, margin=2):
    if avoid_overwrite_func(pose, margin=margin):
        return 0., 3
    sorted_beacons, sorted_distances = sort_beacons_by_distance_func(pose, beacons)
    for beacon, dist in zip(sorted_beacons, sorted_distances):
        if dist > 8: continue
        beacon_centric_pose = beacon_centric_pose_convert_func(pose, beacon)
        dockingZone_indicator = cls.predict(beacon_centric_pose.reshape(1,3))[0]
        if dockingZone_indicator == 0: 
            cache.append(beacon_centric_pose.reshape(3,))
            return 1., dockingZone_indicator
        if dockingZone_indicator == 1: return 0., dockingZone_indicator
    if np.random.rand() < random_approach_threshold: return 1., dockingZone_indicator
    else: return 0., dockingZone_indicator


if __name__=='__main__':
    obstacles = Helper.box_builder('')
    cls = load('dockingZone_classifier.joblib')
    poses_ls, objects_ls, beacons_ls, actions_ls, zones_ls, times_ls, results_ls = [],[],[],[],[],[],[]
    margin_ls, threshold_ls = [],[]
    margin = LIST_OF_MARGINS[RUN_ID-1]
    for k, threshold in enumerate(LIST_OF_THRESHOLDS):
        print('Threshold = '+str(threshold), end=' | ')
        docked_score, pole_score, plant_score, wall_score = 0,0,0,0
        for _ in range(NUM_OF_ITERATION):
            margin_ls.append(margin)
            threshold_ls.append(threshold)
            init_pose, beacons = Helper.initializer(jit = JITTER_LEVEL)
            objects = Helper.concatenate_beacons(beacon_objs=Helper.beacons2objects(beacons),
                                                objects=obstacles)
            pose = np.copy(init_pose)
            render = Render()
            state = State(pose=pose, dt=1/controlconfig.CHIRP_RATE)
            controller = AvoidApproach()
            actions = []
            dockingZones = []
            poses = np.copy(pose).reshape(1,3)
            episode_ended = False

            for i in range(TIME_LIMIT):
                compressed = render.run(pose, objects)
                docked = Helper.dockingCheck(pose, beacons)
                if docked:
                    results_ls.append('docked')
                    times_ls.append(i)
                    episode_ended = True
                    break
                if Helper.collision_check(render.cache['inview'], 'plant'):
                    results_ls.append('hit_plant')
                    times_ls.append(i)
                    episode_ended = True
                    break
                if Helper.collision_check(render.cache['inview'], 'plant'):
                    results_ls.append('hit_wall' if avoid_overwrite(pose) else 'hit_pole')
                    times_ls.append(i)
                    episode_ended =  True
                    break
                
                action, zone = behavior(pose, beacons,
                                avoid_overwrite_func=avoid_overwrite,
                                sort_beacons_by_distance_func=sort_beacons_by_distance,
                                beacon_centric_pose_convert_func=beacon_centric_pose_converter,
                                cls = cls, random_approach_threshold=threshold, margin=margin)
                actions.append(action)
                dockingZones.append(zone)
                v, omega = controller.get_kinematic(compressed, approach_factor=action)
                state.update_kinematic(kinematic=[v, omega])
                state.update_pose()
                pose = np.copy(state.pose)
                poses = np.vstack((poses, pose.reshape(1,3)))

            if not episode_ended: 
                results_ls.append('out')
                times_ls.append(TIME_LIMIT)
            poses_ls.append(poses)
            objects_ls.append(objects)
            beacons_ls.append(beacons)
            actions_ls.append(np.asarray(actions))
            zones_ls.append(np.asarray(dockingZones))

            if results_ls[-1]=='docked': docked_score+=1
            if results_ls[-1]=='hit_pole': pole_score+=1
            if results_ls[-1]=='hit_plant': plant_score+=1
            if results_ls[-1]=='hit_wall': wall_score+=1
        
        print('docked%={:}, pole%={:}, plant%={:}, wall%={:}, AvgTime={:.2f}'.format(docked_score,
                        pole_score, plant_score, wall_score,
                        np.mean(times_ls[k*NUM_OF_ITERATION:((k+1)*NUM_OF_ITERATION-1)])))

        
    ## SAVE RESULT!
    df = pd.DataFrame({'poses': poses_ls, 'objects': objects_ls, 'beacons': beacons_ls,
                    'actions': actions_ls, 'zones': zones_ls, 
                    'times': times_ls, 'results': results_ls})
    df.to_pickle('data_'+str(RUN_ID)+'.pkl')
    np.savez('zone0_poses_'+str(RUN_ID)+'.npz', poses=np.asarray(cache).reshape(-1,3))

    if RUN_ID==3:
        time.sleep(900)
        array = np.asarray([]).reshape(0,3)
        for i in range(1, RUN_ID+1):
            data = np.load('zone0_poses_'+str(i)+'.npz')
            temp = data['poses']
            array = np.vstack((array, temp))
        np.savez('zone0_poses.npz', poses=array)
        df = pd.read_pickle('data_1.pkl')
        for i in range(2, RUN_ID+1):
            df_temp = pd.read_pickle('data_'+str(i)+'.pkl')
            df = pd.concat([df, df_temp])

        df.to_pickle('data_labeler_evaluation.pkl')

        for i in range(1, RUN_ID+1):
            garbage = os.path.join(os.getcwd(), 'data_'+str(i)+'.pkl')
            os.remove(garbage)
            garbage = os.path.join(os.getcwd(), 'zone0_poses_'+str(i)+'.npz')
            os.remove(garbage)