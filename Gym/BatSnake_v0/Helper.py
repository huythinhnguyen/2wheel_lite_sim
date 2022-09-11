import numpy as np
import os
import sys
import pathlib

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(pathlib.Path(os.path.abspath(__file__)).parents[2])

from Sensors.BatEcho import Setting as sensorconfig
from Arena import Builder

HIT_DISTANCE = {'pole': 0.3, 'plant': 0.4}
OBJECT_SPACING = {'pole': 0.1, 'plant': 0.3}
MAZE_SIZE = 16.
TUNNEL_WIDTH = 3.


def collision_check(inview, mode):
    inview_of_klass = inview[inview[:,2]==sensorconfig.OBJECTS_DICT[mode]][:,:2]
    if np.sum(inview_of_klass[:,0]<HIT_DISTANCE[mode]) > 0:
        return True
    else:
        return False


def reward_function(**kwargs):
    if 'hit' in kwargs.keys():
        hit = kwargs['hit']
        if hit == 'plant': return -1
        if hit == 'pole': return +1
    else: return 0


def maze_builder(mode, maze_size=MAZE_SIZE, tunnel_width=TUNNEL_WIDTH):
    if mode=='box':
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
        starts.append([-(maze_size/2 - tunnel_width), -(maze_size/2 - tunnel_width)])
        ends.append([(maze_size/2 - tunnel_width), -(maze_size/2 - tunnel_width)])
        starts.append([(maze_size/2 - tunnel_width), -(maze_size/2 - tunnel_width)])
        ends.append([(maze_size/2 - tunnel_width), (maze_size/2 - tunnel_width)])
        starts.append([(maze_size/2 - tunnel_width), (maze_size/2 - tunnel_width)])
        ends.append([-(maze_size/2 - tunnel_width), (maze_size/2 - tunnel_width)])
        starts.append([-(maze_size/2 - tunnel_width), (maze_size/2 - tunnel_width)])
        ends.append([-(maze_size/2 - tunnel_width), -(maze_size/2 - tunnel_width)])
        #
        spacings = OBJECT_SPACING['plant']*np.ones(8)
        maze = Builder.build_walls(starts, ends, spacings)
    elif mode=='donut':
        maze = Builder.build_circles(np.zeros((2,2)), [maze_size/2, maze_size/2 - tunnel_width], OBJECT_SPACING['plant']*np.ones(2))

    return np.hstack((maze, sensorconfig.OBJECTS_DICT['plant']*np.ones(len(maze),1)))


def spawn_food(mode, level, difficulty=0, maze_size=MAZE_SIZE, tunnel_width=TUNNEL_WIDTH):
    if mode=='box':
        if level%4==0:
            xlim = [(maze_size/2-tunnel_width) + OBJECT_SPACING['plant'], maze_size/2 - OBJECT_SPACING['plant']]
            if difficulty==0: ylim = [0, (maze_size/2 - tunnel_width)]
            elif difficulty>0: ylim = [[0, maze_size/2 - OBJECT_SPACING['plant']]]
        elif level%4==1:
            ylim = [(maze_size/2-tunnel_width) + OBJECT_SPACING['plant'], maze_size/2 - OBJECT_SPACING['plant']]
            if difficulty==0: xlim = [-(maze_size/2 - tunnel_width), 0]
            elif difficulty>0: xlim = [-(maze_size/2 - OBJECT_SPACING['plant']), 0]
        elif level%4==2:
            pass
        elif level%4==3:
            pass

    if mode=='donut':
        pass
    return None