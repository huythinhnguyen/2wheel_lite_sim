from pydoc import Helper
import numpy as np
import sys
import os
import pathlib
from Gym.BatSnake_v0.Helper import MAZE_SIZE

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parents[2]))

from Sensors.BatEcho import Setting as sensorconfig
from Arena import Builder


HIT_DISTANCE = {'pole': 0.3, 'plant': 0.4}
OBJECT_SPACING = {'pole': 0.1, 'plant': 0.3}

BEACON_SPECS = {'mode': 'bull', 'bull_angle': np.pi/3, 'hit_distance': 0.75,
                'hit_angle': np.pi/2} 

JITTER={'xy': 0.1, 'theta': np.pi/18}

def beacon2objects(x, y, theta, d, mode=BEACON_SPECS['mode'], bull_angle=BEACON_SPECS['bull_angle']):
    base = np.asarray([x, y, 2], dtype=np.float32).reshape(1,3)
    # make horn:
    if mode=='unicorn':
        horn = np.asarray([x + d*np.cos(theta), y +d*np.sin(theta), 1], dtype=np.float32).reshape(1,3)
    elif mode=='bull':
        horn = np.asarray([[x + d*np.cos(theta+bull_angle/2), y + d*np.sin(theta+bull_angle/2), 1],
                            [x + d*np.cos(theta-bull_angle/2), y + d*np.sin(theta-bull_angle/2), 1]], 
                            dtype=np.float32).reshape(2,3)
    return np.vstack((base, horn))


def beacons2objects(beacons: np.ndarray, d=0.3, mode=BEACON_SPECS['mode'], bull_angle=BEACON_SPECS['bull_angle']):
    objects = np.asarray([], dtype=np.float32).reshape(0,3)
    for beacon in beacons:
        x, y, theta = beacon
        new = beacon2objects(x, y, theta, d=d, mode=mode, bull_angle=bull_angle)
        objects = np.vstack((objects, new))
    return objects


def concatenate_beacons(beacon_objs: np.ndarray, objects: np.ndarray):
    return np.vstack((beacon_objs, objects))


def beacons_count(objects: np.ndarray):
    ans = np.sum(objects[:,2])
    if objects[1,2]!=objects[2,2]: return ans
    else: return int(ans/2)


def beacons_mode(objects: np.ndarray):
    if (objects[1,2]!=objects[2,2]): return 'unicorn'
    else: return 'bull'


## Arena only consist of Plants if the number of beacons is not given
def isolate_beacon_objs_from_objects(objects: np.ndarray, number_of_beacons=None):
    if number_of_beacons is None:
        number_of_beacons = beacons_count(objects)
    if number_of_beacons==0: return np.asarray([]).reshape(0,3)
    mode = beacons_mode(objects)
    beacon_objs = np.asarray([], dtype=np.float32).reshape(0,3)
    t = 3 if mode=='bull' else 2
    for i in range(number_of_beacons):
        beacon_objs = np.vstack((beacon_objs, objects[t*i:t(i+1)]))
    return beacon_objs


def object2beacon(beacon_obj: np.ndarray):
    N = len(beacon_obj)
    mode = 'bull' if N==3 else 'unicorn' if N==2 else None
    x, y = beacon_obj[0][:2]
    if mode=='unicorn':
        xh, yh = beacon_obj[1][:2]
    elif mode=='bull':
        xh = (beacon_obj[1][0] + beacon_obj[2][0])/2
        yh = (beacon_obj[1][1] + beacon_obj[2][1])/2
    theta = np.arctan2(yh-y, xh-x)
    return x, y, theta


def objects2beacons(beacon_objs: np.ndarray):
    number_of_beacons = beacons_count(beacon_objs)
    mode = beacons_mode(beacon_objs)
    t = 3 if mode=='bull' else 2
    beacons = np.zeros((number_of_beacons, 3))
    for i in range(number_of_beacons):
        beacons[i] = object2beacon(beacon_objs[t*i:t*(i+1)])
    return beacons


def getBeaconInView(pose:np.ndarray, beacons:np.ndarray, range=sensorconfig.FOV_LINEAR, fov=sensorconfig.FOV_ANGULAR):
    x, y, yaw = pose
    A = np.empty((len(beacons),2))
    A[:,0] = beacons[:,0] - x
    A[:,1] = beacons[:,1] - y
    Rho = np.power(A[:,0],2) + np.power(A[:,1],2)
    Alpha=np.arctan2(A[:,1], A[:,0])
    range_condition = Rho < range
    fov_condition = np.abs(Builder.wrap2pi(yaw-Alpha)) < fov
    return beacons[range_condition*fov_condition]


def isBatInViewBeacon(pose:np.ndarray, beacon:np.ndarray, range=BEACON_SPECS['hit_distance'], fov=BEACON_SPECS['hit_angle']):
    x, y = pose[:2]
    xb, yb, phi = beacon.reshape(3,)
    r2 = np.power(x-xb,2) + np.power(y-yb,2)
    alpha = np.arctan2(y-yb, x-xb)
    if r2 > np.power(range,2): return False
    if np.abs(Builder.wrap2pi(phi-alpha)) > fov/2: return False
    return True


def isBatFacingBeacon(pose:np.ndarray, beacon:np.ndarray, fov=np.pi/3):
    xb, yb, phi = beacon.reshape(3,)
    x, y, yaw = pose
    alpha = np.arctan2(yb-y, xb-x)
    beta = np.arctan2(y-yb, x-xb)
    left = Builder.wrap2pi(phi - beta) < 0
    if np.abs(Builder.wrap2pi(phi-yaw)) < np.pi-(fov/2): return False
    if left:
        if Builder.wrap2pi(yaw-alpha) > fov/2: return False
        if Builder.wrap2pi(yaw-alpha) < 0: return False
    else:
        if Builder.wrap2pi(yaw-alpha) < -fov/2: return False
        if Builder.wrap2pi(yaw-alpha) > 0: return False
    return True


def dockingCheck(pose:np.ndarray, beacons:np.ndarray, **kwargs):
    beaconsInView = getBeaconInView(pose, beacons)
    if len(beaconsInView)<1: return False
    for beacon in beaconsInView:
        if ('beacon_range' in kwargs.keys()) and ('beacon_fov' in kwargs.keys()):
            bat_is_inView = isBatInViewBeacon(pose, beacon, range=kwargs['beacon_range'], fov=kwargs['beacon_fov'])
        else: bat_is_inView = isBatInViewBeacon(pose, beacon)
        bat_is_facing = isBatFacingBeacon(pose, beacon, fov=kwargs['facing_fov']) if 'facing_fov' in kwargs.keys() else isBatFacingBeacon(pose, beacon)

        if bat_is_facing and bat_is_inView: return True
    return False 


def collision_check(inview, mode):
    inview_of_klass = inview[inview[:,2]==sensorconfig.OBJECTS_DICT[mode]][:,:2]
    if np.sum(inview_of_klass[:,0]<HIT_DISTANCE[mode]) > 0:
        return True
    else:
        return False


def box_builder(mode, maze_size=MAZE_SIZE):
    starts, ends = [],[] 
    starts.append([-maze_size/2, -maze_size/2])
    ends.append([maze_size/2, -maze_size/2])
    starts.append([maze_size/2, -maze_size/2])
    ends.append([maze_size/2, maze_size/2])
    starts.append([maze_size/2, maze_size/2])
    ends.append([-maze_size/2, maze_size/2])
    starts.append([-maze_size/2, maze_size/2])
    ends.append([-maze_size/2, -maze_size/2])
    #
    spacings = OBJECT_SPACING['plant']*np.ones(8)
    maze = Builder.build_walls(starts, ends, spacings)

    return np.hstack((maze, sensorconfig.OBJECTS_DICT['plant']*np.ones((len(maze),1))))


def initializer(number_of_positions=9, number_of_poses=4, jit=0): # INIT 9 or 4 EVENLY SPACED POSITION WITH JITTER
    beacons = np.empty((number_of_positions, 3))
    poses = np.empty((number_of_poses, 3))
    d = MAZE_SIZE/(3.5)
    locs = [-d, 0, d]
    for i, pos in enumerate(beacons):
        pos[0] = locs[int(i%3)] + np.random.uniform(low=-1, high=1)*JITTER['xy']*jit
        pos[1]= locs[int(i/3)] + np.random.uniform(low=-1, high=1)*JITTER['xy']*jit
        pos[2] = np.random.uniform(low=-np.pi, high=np.pi)
    d = MAZE_SIZE/7
    locs = [-d, d]
    for i, pos in enumerate(poses):
        pos[0] = locs[int(i%2)] + np.random.uniform(low=-1, high=1)*JITTER['xy']*jit
        pos[1]= locs[int(i/2)] + np.random.uniform(low=-1, high=1)*JITTER['xy']*jit
        pos[2] = np.random.uniform(low=-np.pi, high=np.pi)
    bat_selector = np.random.randint(low=0, high=number_of_poses)
    bat_pose = poses[bat_selector].reshape(3,)
    return bat_pose, beacons





