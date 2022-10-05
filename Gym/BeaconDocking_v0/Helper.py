import numpy as np
import sys
import os
import pathlib

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parents[2]))

from Sensors.BatEcho import Setting as sensorconfig
from Arena import Builder

BEACON_KLASS = 3
BEACON_MODE = 'bull'
BEACON_HORN_ANGLE = np.pi/3

def beacon2objects(x, y, theta, d, mode='bull', bull_angle=np.pi/3):
    base = np.asarray([x, y, 2], dtype=np.float32).reshape(1,3)
    # make horn:
    if mode=='unicorn':
        horn = np.asarray([x + d*np.cos(theta), y +d*np.sin(theta), 1], dtype=np.float32).reshape(1,3)
    elif mode=='bull':
        horn = np.asarray([[x + d*np.cos(theta+bull_angle/2), y + d*np.sin(theta+bull_angle/2), 1],
                            [x + d*np.cos(theta-bull_angle/2), y + d*np.sin(theta-bull_angle/2), 1]], 
                            dtype=np.float32).reshape(2,3)
    return np.vstack((base, horn))


def beacons2objects(beacons: np.ndarray, d=0.3, mode='bull', bull_angle=np.pi/3):
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
        beacon_objs = np.vstack((beacon_objs, objects[t*i:t*i+t]))
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


def objects2beacons(beacon_objs: np.ndarray, mode: str):
    
    return None
