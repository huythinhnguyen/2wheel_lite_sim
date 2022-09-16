import numpy as np
import os
import sys
import pathlib

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parents[2]))

from Sensors.BatEcho import Setting as sensorconfig
from Arena import Builder

HIT_DISTANCE = {'pole': 0.3, 'plant': 0.4}
OBJECT_SPACING = {'pole': 0.1, 'plant': 0.3}
MAZE_SIZE = 16.
TUNNEL_WIDTH = 3.
JITTER={'xy': 0.1, 'theta': np.pi/18}


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
        if hit == 'pole': 
            if 'food_polar' in kwargs.keys():
                food_polar = kwargs['food_polar'].reshape(-1,)[:2]
                if np.abs(food_polar[1]) < np.pi/6: return 1
                elif np.abs(food_polar[1]) < np.pi/2: return 0.2
                else: return 0
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

    return np.hstack((maze, sensorconfig.OBJECTS_DICT['plant']*np.ones((len(maze),1))))


def spawn_food(mode, level, difficulty=0, maze_size=MAZE_SIZE, tunnel_width=TUNNEL_WIDTH):
    if mode=='box':
        xlim_ls_normal=[((maze_size/2-tunnel_width) + OBJECT_SPACING['plant'], maze_size/2 - OBJECT_SPACING['plant']),(-(maze_size/2 - tunnel_width), 0),
                        (-(maze_size/2 - OBJECT_SPACING['plant']),-(maze_size/2 - tunnel_width + OBJECT_SPACING['plant'])),(0, (maze_size/2 - tunnel_width))]
        xlim_ls_hard = xlim_ls_normal.copy()
        xlim_ls_hard[1], xlim_ls_hard[3] = (-(maze_size/2 - OBJECT_SPACING['plant']), 0), (0, maze_size/2 - OBJECT_SPACING['plant'])
        #
        ylim_ls_normal=[(0, (maze_size/2 - tunnel_width)),((maze_size/2-tunnel_width) + OBJECT_SPACING['plant'], maze_size/2 - OBJECT_SPACING['plant']),
                        (0, -(maze_size/2 - tunnel_width)),(-(maze_size/2 - OBJECT_SPACING['plant']), -(maze_size/2-tunnel_width+OBJECT_SPACING['plant']))]
        ylim_ls_hard = ylim_ls_normal.copy()
        ylim_ls_hard[0], ylim_ls_hard[2] = (0, maze_size/2 - OBJECT_SPACING['plant']), (0, -(maze_size/2 - OBJECT_SPACING['plant']))
        #
        xlim = xlim_ls_normal[int(level%4)] if difficulty==0 else xlim_ls_hard[int(level%4)]
        ylim = ylim_ls_normal[int(level%4)] if difficulty==0 else ylim_ls_hard[int(level%4)]
        #
        food = np.asarray([np.random.uniform(*xlim), np.random.uniform(*ylim), sensorconfig.OBJECTS_DICT['pole']]).reshape(1,3)
    if mode=='donut':
        rlim = (maze_size/2 - tunnel_width + OBJECT_SPACING['plant'], maze_size/2 - OBJECT_SPACING['plant'])
        alim = np.asarray([0, (1+difficulty)*np.pi/9])
        rho = np.random.uniform(*rlim)
        phi = np.random.uniform(*Builder.wrap2pi(alim + (level%4)*(np.pi/2)))
        food = np.asarray([ *Builder.pol2cart(rho,phi), sensorconfig.OBJECTS_DICT['pole']]).reshape(1,3)
    return food


def spawn_bat(mode, phase, maze_size=MAZE_SIZE, tunnel_width=TUNNEL_WIDTH, jitter=True): 
    if mode=='box':
        pose_ls = [(maze_size/2-tunnel_width/2,-0.5,np.pi/2),(maze_size/2-tunnel_width/2,-1.0,np.pi/2),
                   (maze_size/2-tunnel_width/2,-1.5,np.pi/2),(0,-(maze_size/2-tunnel_width/2),0.)]
        pose = np.asarray([*pose_ls[phase]])
    if mode=='donut':
        rho = maze_size/2-tunnel_width/2
        phi_ls = np.asarray([-1., -2., -3., -9])*(np.pi/18)
        phi = phi_ls[phase]
        pose = np.asarray([ *Builder.pol2cart(rho, phi), Builder.wrap2pi(phi + np.pi/2) ])
    if jitter:
        pose += np.asarray([JITTER['xy'], JITTER['xy'] , JITTER['theta']])*np.random.uniform(-1,-1,size=3)
    return pose


def out_of_bound(pose, mode, maze_size=MAZE_SIZE, tunnel_width=TUNNEL_WIDTH):
    x, y = pose[:2]
    if mode=='box':
        if x<0 and y>(maze_size/2-tunnel_width): return 1
        elif x<-(maze_size/2-tunnel_width) and y>-maze_size/2: return -1
        else: return 0
    if mode=='donut':
        rho, phi = Builder.cart2pol(x,y)
        if phi > np.pi/2: return 1
        elif phi < -3*np.pi/4: return -1
        else: return 0
