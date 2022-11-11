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
NUM_OF_ITERATION = 500
LIST_OF_MARGINS = [1.5]
JITTER_LEVEL = 2
LIST_OF_THRESHOLDS = [0.25, 0.5, 0.75]
TIME_LIMIT = 1000
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
        return 0.
    if np.random.rand() < random_approach_threshold: return 1.
    return 0.
    

if __name__=='__main__':
    obstacles = Helper.box_builder('')
    cls = load('dockingZone_classifier.joblib')
    poses_ls, objects_ls, beacons_ls, actions_ls, times_ls, results_ls = [],[],[],[],[],[]
    margin_ls, threshold_ls = [],[]
    margin = LIST_OF_MARGINS[RUN_ID-1]
    for k, threshold in enumerate(LIST_OF_THRESHOLDS):
        print('Threshold = '+str(threshold), end=' | ')
        docked_score, pole_score, plant_score, wall_score, out_score = 0,0,0,0,0
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
                
                action = behavior(pose, beacons,
                                avoid_overwrite_func=avoid_overwrite,
                                sort_beacons_by_distance_func=sort_beacons_by_distance,
                                beacon_centric_pose_convert_func=beacon_centric_pose_converter,
                                cls = cls, random_approach_threshold=threshold, margin=margin)
                actions.append(action)
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

            if results_ls[-1]=='docked': docked_score+=1
            if results_ls[-1]=='hit_pole': pole_score+=1
            if results_ls[-1]=='hit_plant': plant_score+=1
            if results_ls[-1]=='hit_wall': wall_score+=1
            if results_ls[-1]=='out': out_score+=1
        
        print('docked%={:.1f}, pole%={:.1f}, plant%={:.1f}, wall%={:.1f}, out%={:.1f}, AvgTime={:.2f}'.format(
                        docked_score*100/NUM_OF_ITERATION, pole_score*100/NUM_OF_ITERATION, 
                        plant_score*100/NUM_OF_ITERATION, wall_score*100/NUM_OF_ITERATION, out_score*100/NUM_OF_ITERATION,
                        np.mean(times_ls[k*NUM_OF_ITERATION:((k+1)*NUM_OF_ITERATION-1)])))


    ## SAVE RESULT!
    df = pd.DataFrame({'poses': poses_ls, 'objects': objects_ls, 'beacons': beacons_ls,
                    'actions': actions_ls, 
                    'times': times_ls, 'results': results_ls})
    df.to_pickle('data_random_labeler_'+str(RUN_ID)+'.pkl')