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
        starts, ends, spacings = [],[],[]
