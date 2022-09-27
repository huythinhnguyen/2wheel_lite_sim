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
    objects = np.asarray([x, y, 2], dtype=np.float32).reshape(1,3)
    # make horn:
    if mode=='unicorn':
        horn = np.asarray([x + d*np.cos(theta), y +d*np.sin(theta), 1], dtype=np.float32).reshape(1,3)
    elif mode=='bull':
        horn = np.asarray([[x + d*np.cos(theta+bull_angle/2), y + d*np.sin(theta+bull_angle/2), 1],
                            [x + d*np.cos(theta-bull_angle/2), y + d*np.sin(theta-bull_angle/2), 1]], 
                            dtype=np.float32).reshape(2,3)
    return np.vstack((objects, horn))


def beacons2objects(beacon_objects, beacon_klass=BEACON_KLASS):
    theta = 0. # ??? NOTE: Need to fixed this!
    x, y = beacon2objects[beacon2objects[:,2]==beacon_klass][:2]
    return beacon2objects(x,y,theta, d='bull', bull_angle=BEACON_HORN_ANGLE)