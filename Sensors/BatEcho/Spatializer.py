"""
Spatializer reference sounds into scence when given locations of sound sources
"""

import numpy as np
from . import Cochlear
import sys
import os

SAMPLE_FREQ = 3e5
SPEED_OF_SOUND = 340
DISTANCE_ENCODING = np.arange(1,7001) * 0.5 * (1/SAMPLE_FREQ) * SPEED_OF_SOUND

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/dataset")

class Retriever:
    def __init__(self, ):
        self.pole_starts = _POLE_STARTS
        self.pole_ends   = _POLE_ENDS
        self.polerev_starts = _POLEREV_STARTS
        self.polerev_ends   = _POLEREV_ENDS
        self.plant_starts   = _PLANT_STARTS
        self.plant_ends     = _PLANT_ENDS
        self.bg_sigma = 0.6
        self.bg_mu = 5e-4
        self.emission_encoding = 0.33
        self.random = True
        self.non_random_index = None
        self.angle_step = 1
        self.raw_length = 7000
        self.objects_dict = {'background': 0, 'pole':1, 'plant':2}
        for i in range(-1,-len(sys.path), -1):
            if 'dataset' in sys.path[i]:
                self.DATAROOT = sys.path[i]
                break
        # TRANSFORMATION PARAMETER:
        self.outward_spread_factor = 1
        self.inward_spread_factor = 0.5
        self.air_absorption = 1.31

            
    def _get_reference(self,objects):
        if not self.random:
            np.random.seed(1)
            if self.non_random_index is not none: selection = self.non_random_index
            else: raise valueerror('must input index when random=false')
        distances = [objects[0]] if type(objects)==list else list(objects[:,0])
        angles = [objects[1]] if type(objects)==list else list(objects[:,1])
        klasses = [objects[2]] if type(objects)==list else list(objects[:,2])
        left_echoes, right_echoes = np.empty((len(distances),self.raw_length)), np.empty((len(distances),self.raw_length))
        for i, (distance, angle, klass) in enumerate(zip(distances, angles, klasses)):
            path = self._get_data_path(distance, angle, klass)
            left_set = np.load(os.path.join(path, 'left.npy'))
            right_set = np.load(os.path.join(path, 'right.npy'))
            if self.random: selection = np.random.randint(len(left_set))
            left_echoes[i], right_echoes[i] = left_set[selection,:], right_set[selection,:]
        return left_echoes, right_echoes


    def _get_data_path(self,distance,angle,klass):
        d = str(np.round(float(distance),2))
        a = str(np.round(float(angle) ))
        k = list(self.objects_dict.keys())[list(self.objects_dict.values()).index(int(klass))]
        path = os.path.join(self.DATAROOT,k,d+'_'+a)
        return path
        
        

    def _get_angle_interpolated_reference(self,objects): # LINEAR
        if type(objects)==list: objects = np.array(objects).reshape(-1,3)
        objects_A, objects_B = np.copy(objects), np.copy(objects)
        objects_A[:,1], objects_B[:,1] = np.floor(objects[:,1]), np.ceil(objects[:,1])
        weights = np.array([1-np.modf(objects[:,1])[0],np.modf(objects[:,1])[0]]).T
        leftA, rightA = self._get_reference(objects_A)
        leftB, rightB = self._get_reference(objects_B)
        left_echoes = leftA * weights[:,:1] + leftB * weights[:,1:]
        right_echoes=rightA * weights[:,:1] +rightB * weights[:,1:]
        return left_echoes, right_echoes


    def _get_snip(self, objects):
        #objects = objects[ np.argsort(objects[:,2]) ]
        left_echoes, right_echoes = self._get_angle_interpolated_reference(objects)
        masks = np.empty(left_echoes.shape)
        temp_indexes = objects[:,2]==self.objects_dict['pole']
        masks[ temp_indexes ] = self._pole_mask(objects[ temp_indexes][:,0])
        temp_indexes = objects[:,2]==self.objects_dict['plant']
        masks[ temp_indexes ] = self._plant_mask(objects[ temp_indexes][:,0])
        return masks*left_echoes, masks*right_echoes


    def _pole_mask(self, distances, main=True, rev=True):
        if main: starts, ends = [],[]
        if rev : starts2,ends2= [],[]
        ref_distances = self._calc_ref_distances(distances, mode='pole')
        for ref in distances:
            if main:
                starts.append(self.pole_starts[ref])
                ends.append(self.pole_ends[ref])
            if rev:
                starts2.append(self.polerev_starts[ref])
                ends2.append(self.polerev_ends[ref])
        if main:
            start_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(starts).reshape(-1,1)),axis=1)
            end_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(ends).reshape(-1,1)), axis=1)
        if rev:
            start2_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(starts2).reshape(-1,1)),axis=1)
            end2_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(ends2).reshape(-1,1)), axis=1)
        mask = np.zeros((len(distances),self.raw_length))
        if main:
            for i, (sid, eid) in enumerate(zip(start_indexes, end_indexes)): mask[i][sid:eid] = 1.
        if rev:
            for i, (sid, eid) in enumerate(zip(start2_indexes, end2_indexes)): mask[i][sid:eid] = 1.
        return mask


    def _plant_mask(self, distances):
        starts, ends = [],[]
        for ref in distances:
            starts.append(self.plant_starts[ref])
            ends.append(self.plant_ends[ref])
        start_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(starts).reshape(-1,1)), axis=1)
        end_indexes = np.argmin(np.abs(DISTANCE_ENCODING - np.asarray(ends).reshape(-1,1)), axis=1)
        mask = np.zeros((len(distances),self.raw_length))
        for i, (sid, eid) in enumerate(zip(start_indexes, end_indexes)): mask[i][sid:eid] = 1.
        return mask


    def _calc_ref_distances(self, distances, mode):
        if mode not in self.objects_dict.keys(): raise ValueError('_calc_ref_distance mode not valid')
        if mode=='pole':
            pass
        elif mode=='plant':
            pass
    

    def _get_background(self):
        left_echo, right_echo = self._get_reference([0.0, 0.0, 0])
        bg_index = np.argmin(np.abs(DISTANCE_ENCODING - self.emission_encoding)) + 1
        left_echo[:, bg_index:] = np.random.normal( self.bg_mu, self.bg_sigma, self.raw_length - bg_index )
        right_echo[:,bg_index:] = np.random.normal( self.bg_mu, self.bg_sigma, self.raw_length - bg_index )
    

class Render:
    def __init__(self):
        pass
    


_POLE_STARTS  = {0.25: 0.13, 0.5: 0.38, 0.75: 0.62, 1.0: 0.88, 1.25: 1.12, 1.5: 1.36, 1.75: 1.62, 2.0: 1.87, 2.25: 2.11, 2.5: 2.35}
_POLE_ENDS    = {0.25: 0.47, 0.5: 0.66, 0.75: 0.88, 1.0: 1.1, 1.25: 1.31, 1.5: 1.53, 1.75: 1.76, 2.0: 1.98, 2.25: 2.22, 2.5: 2.48}
_POLEREV_STARTS= {0.25: 0.89, 0.5: 0.79, 0.75: 0.94,1.0: 1.13, 1.25: 1.33, 1.5: 1.55, 1.75: 1.78, 2.0: 2.01, 2.25: 2.24, 2.5: 3.99}
_POLEREV_ENDS  = {0.25: 1.13, 0.5: 0.99, 0.75: 1.15, 1.0: 1.33, 1.25: 1.51, 1.5: 1.71, 1.75: 1.92, 2.0: 2.14, 2.25: 2.35, 2.5: 3.99}
_PLANT_STARTS = {0.5: 0.26, 0.75: 0.49, 1.0: 0.76, 1.25: 0.99, 1.5: 1.24, 1.75: 1.49, 2.0: 1.74, 2.25: 1.91, 2.5: 2.24}
_PLANT_ENDS   = {0.5: 1.29, 0.75: 1.35, 1.0: 1.57, 1.25: 1.71, 1.5: 1.76, 1.75: 1.88, 2.0: 2.38, 2.25: 2.47, 2.5: 2.72}
