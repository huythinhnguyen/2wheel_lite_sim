import numpy as np
import os
import sys
import time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.getcwd())

from Arena import Builder

"""
different types of maze:
1. straight tunnels (narrow/ wide):
- 1.a: infinite straight
- 1.b: infinite zigzag of turn angles: 45, 90, 135.
- 1.c: infty meandering
2. Empty box:
- 2.a: donut:
- 2.b: square:
3. Infinite dense world: Different density.
4. Empty with cluster (implement when have time)
"""
class Maze:
    def __init__(self, maze_id, obstacles_type=['plant'], **kwargs):
        self.maze_id = maze_id
        if obstacles_type[0]=='plant': self.spacing = 0.3
        if obstacles_type[0]=='pole' : self.spacing = 0.1
        if 'maze_size' in kwargs.keys(): self.maze_size = kwargs['maze_size']
        if 'tunnel_width' in kwargs.keys(): self.tunnel_width=kwargs['tunnel_width']
        if 'cycle_number' in kwargs.keys(): self.cycle_number=kwargs['cycle_number']
        if 'zigzag_angle' in kwargs.keys(): self.zigzag_angle=kwargs['zigzag_angle']


    def _straight_tunnel(self):
        starts, ends, spacings = [], [], []
        # upper wall:
        starts.append([0, self.tunnel_width/2])
        ends.append([self.maze_size, self.tunnel_width/2])
        spacings.append(self.spacing)
        # lower wall:
        starts.append([0, -1*self.tunnel_width/2])
        ends.append([self.maze_size, -1*self.tunnel_width/2])
        spacings.append(self.spacing)
        return Builder.build_walls(starts, ends, spacings)


    def _meandering_tunnel(self):
        arcs = np.array([]).reshape(-1,2)
        for i in range(self.cycle_number):
            s1u, s1d = [0, self.maze_size/4 + self.tunnel_width/2],[0, self.maze_size/4 - self.tunnel_width/2]
            e1u, e1d = [self.maze_size/4 + self.tunnel_width/2, 0],[self.maze_size/4 - self.tunnel_width/2, 0]
            s2u, s2d = e1u.copy(), e1d.copy()
            e2u, e2d = [3*self.maze_size/4 - self.tunnel_width/2, 0],[3*self.maze_size/4 + self.tunnel_width/2, 0]
            s3u, s3d = e2u.copy(), e2d.copy()
            e3u, e3d = [self.maze_size, self.maze_size/4 + self.tunnel_width/2],[self.maze_size, self.maze_size/4 - self.tunnel_width/2]
            
            arc1u = Builder.build_arc(s1u,e1u, radius=self.maze_size/4 + self.tunnel_width/2, spacing=self.spacing, spin=-1)
            arc1d = Builder.build_arc(s1d,e1d, radius=self.maze_size/4 - self.tunnel_width/2, spacing=self.spacing, spin=-1)
            arc2u = Builder.build_arc(s2u,e2u, self.maze_size/4 - self.tunnel_width/2, self.spacing, spin= 1)
            arc2d = Builder.build_arc(s2d,e2d, self.maze_size/4 + self.tunnel_width/2, self.spacing, spin= 1)
            arc3u = Builder.build_arc(s3u,e3u, self.maze_size/4 + self.tunnel_width/2, self.spacing, spin=-1)
            arc3d = Builder.build_arc(s3d,e3d, self.maze_size/4 - self.tunnel_width/2, self.spacing, spin=-1)
            temp = np.vstack((arc1u, arc1d, arc2u, arc2d, arc3u, arc3d))
            temp[:,0] = temp[:,0] + i * self.maze_size
            arcs = np.vstack((arcs,temp))
        return arcs


    def _zigzag_tunnel(self):
        starts, ends = [],[]
        primer = [0., 0., 0.]
        starts.append([ primer[0] + (self.tunnel_width/2)*np.cos(primer[2] + np.pi/2),
                        primer[1] + (self.tunnel_width/2)*np.sin(primer[2] + np.pi/2)])
        starts.append([ primer[0] + (self.tunnel_width/2)*np.cos(primer[2] - np.pi/2),
                        primer[1] + (self.tunnel_width/2)*np.sin(primer[2] - np.pi/2)])
        
        primer[0] = primer[0] + np.cos(primer[2])*self.maze_size/4
        primer[1] = primer[1] + np.sin(primer[2])*self.maze_size/4

        ends.append([ primer[0] + (self.tunnel_width/2)*np.cos(primer[2] + np.pi/2),
                        primer[1] + (self.tunnel_width/2)*np.sin(primer[2] + np.pi/2)])
        ends.append([ primer[0] + (self.tunnel_width/2)*np.cos(primer[2] - np.pi/2),
                        primer[1] + (self.tunnel_width/2)*np.sin(primer[2] - np.pi/2)])
        
        for i in range(self.cycle_number):
            starts.append(ends[-2])
            starts.append(ends[-1])            
            primer[0] = primer[0] + np.cos(primer[2])*self.maze_size/4
            primer[1] = primer[1] + np.sin(primer[2])*self.maze_size/4
            primer[2] = Builder.wrap2pi(primer[2] + (np.pi - self.zigzag_angle))
            d = self.tunnel_width/(2*np.cos(0.5*(np.pi - self.zigzag_angle)))
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2] + self.zigzag_angle/2)),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2] + self.zigzag_angle/2))])
            starts.append(ends[-1])
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2]-(np.pi-self.zigzag_angle/2))),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2]-(np.pi-self.zigzag_angle/2)))])
            starts.append(ends[-1])
            primer[0] = primer[0] + np.cos(primer[2])*self.maze_size/2
            primer[1] = primer[1] + np.sin(primer[2])*self.maze_size/2
            primer[2] = Builder.wrap2pi(primer[2] - (np.pi - self.zigzag_angle))
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2] + (np.pi-self.zigzag_angle/2) )),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2] + (np.pi-self.zigzag_angle/2) ))])
            starts.append(ends[-1])
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2]-(self.zigzag_angle/2))),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2]-(self.zigzag_angle/2)))])
            starts.append(ends[-1])
            if i == self.cycle_number - 1:
                primer[0] = primer[0] + np.cos(primer[2])*self.maze_size/2
                primer[1] = primer[1] + np.sin(primer[2])*self.maze_size/2
            else:
                primer[0] = primer[0] + np.cos(primer[2])*self.maze_size/4
                primer[1] = primer[1] + np.sin(primer[2])*self.maze_size/4
                
            ends.append([primer[0] + (self.tunnel_width/2)*np.cos(primer[2] + np.pi/2),
                         primer[1] + (self.tunnel_width/2)*np.sin(primer[2] + np.pi/2)])
            ends.append([primer[0] + (self.tunnel_width/2)*np.cos(primer[2] - np.pi/2),
                         primer[1] + (self.tunnel_width/2)*np.sin(primer[2] - np.pi/2)])            
        spacings = np.ones(len(starts))*self.spacing
        
        return Builder.build_walls(starts, ends, spacings)


class Recorder:
    def __init__(self, maze, episode):
        self.poses = []
        self.maze = maze
        self.episode = episode
        self.echoes_L = [] 
        self.echoes_R = []
        self.IIDs = []
        self.cache = {}


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from Animation import ObjUtils
    from Animation.Render import StillImage
    from Simulation.Motion import State, Drive
    from Control.SensorimotorLoops.BatEcho import Avoid
    from Sensors.BatEcho.Spatializer import Render
    
    mazeBuilder = Maze(maze_id='1.a', maze_size=20., tunnel_width=3.0, cycle_number=1, zigzag_angle=np.pi/4)
    maze = mazeBuilder._zigzag_tunnel()
    maze = np.hstack((maze, np.ones(len(maze)).reshape(-1,1)))
    records={}
    plans = ['B']
    for i in range(1):
        ended = False
        rec = Recorder(maze, i)
        bat = State(pose=np.array([1.,0.,0.]), dt=1/50)
        controller = Avoid()
        controller.plan = plans[i]
        render = Render()
        rec.cache['v'] = []
        rec.cache['omega'] = []
        for _ in range(10000):
            rec.poses.append(bat.pose)
            echoes = render.run(bat.pose, maze)
            rec.echoes_L.append(echoes['left'])
            rec.echoes_R.append(echoes['right'])
            v, omega = controller.get_kinematic(echoes)
            rec.cache['v'].append(v)
            rec.cache['omega'].append(omega)
            rec.IIDs.append(controller.cache['IID'])
            bat.update_kinematic(kinematic=[v,omega])
            bat.update_pose()
            #print(bat.kinematic)
            #print('onset',controller.cache['onset_distance'])
            if bat.pose[0] < 0 or bat.pose[0] > 10: break
            if np.sum(render.cache['inview'][:,0] < 0.3)>0: break
        if 'poses' in records: records['poses'].append(rec.poses)
        else: records['poses'] = [rec.poses]

    for i in range(1):
        objects = {'plant': maze, 'agent': np.asarray(records['poses'][i])}
        imager = StillImage(objects_types=['plant', 'agent'], sparseness=200)
        imager.render(objects)
        plt.show()
        time.sleep(1)
        plt.close()

    np.savez('test_data_08.22.22.npz',
             poses = np.asarray(rec.poses),
             echoes_L = np.asarray(rec.echoes_L),
             echoes_R = np.asarray(rec.echoes_R),
             v = np.asarray(rec.cache['v']),
             omega = np.asarray(rec.cache['omega']),
             iid = np.asarray(rec.IIDs) )
    
             
            
