import numpy as np
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
print('Current working path:\n', os.getcwd())


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
        from Arena import Builder
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
            arc1u = Builder.build_arc(s1u,e1u, self.maze_size/4 + self.tunnel_width/2, self.spacing, spin=-1)
            arc1d = Builder.build_arc(s1d,e1d, self.maze_size/4 - self.tunnel_width/2, self.spacing, spin=-1)
            arc2u = Builder.build_arc(s2u,e2u, self.maze_size/4 - self.tunnel_width/2, self.spacing, spin= 1)
            arc2d = Builder.build_arc(s2d,e2d, self.maze_size/4 + self.tunnel_width/2, self.spacing, spin= 1)
            arc3u = Builder.build_arc(s3u,e3u, self.maze_size/4 + self.tunnel_width/2, self.spacing, spin=-1)
            arc3d = Builder.build_arc(s3d,e3d, self.maze_size/4 - self.tunnel_width/2, self.spacing, spin=-1)
            temp = np.vstack((arc1u, arc1d, arc2u, ar2d, arc3u, arc3d))
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
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2] + self.zigzag_angle/2)),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2] + self.zigzag_angle/2))])
            starts.append(ends[-1])
            ends.append([primer[0]+d*np.cos(Builder.wrap2pi(primer[2]-(np.pi-self.zigzag_angle/2))),
                         primer[1]+d*np.sin(Builder.wrap2pi(primer[2]-(np.pi-self.zigzag_angle/2)))])
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
        spacings = np.ones(len(starts))*self.spacings
        
        return Builder.build_walls(starts, ends, spacings)


class Recorder:
    def __init__(self, maze, episode):
        self.poses = []
        self.maze = maze
        self.episode = episode
        self.cache = {}



if __name__=='__main__':
    mazeBuilder = Maze(maze_id='1.a', maze_size=8., tunnel_width=3.)
    maze = mazeBuilder._straigth_tunnel()
    
