"""
Spatializer reference sounds into scence when given locations of sound sources
"""

import numpy as np
from . import Cochlear, Compressors, Setting
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/dataset")

class Retriever:
    def __init__(self, ):
        self.pole_starts = Setting._POLE_STARTS
        self.pole_ends   = Setting._POLE_ENDS
        self.plant_starts   = Setting._PLANT_STARTS
        self.plant_ends     = Setting._PLANT_ENDS
        self.bg_sigma = Setting.BACKGROUND_SIGMA
        self.bg_mu = Setting.BACKGROUND_MU
        self.emission_encoding = Setting.EMISSION_ENCODING
        self.emission_index = Setting.EMISSION_INDEX
        self.random = True
        self.non_random_index = None
        self.angle_step = Setting.ANGLE_STEP
        self.raw_length = Setting.RAW_DATA_LENGTH
        self.objects_dict = Setting.OBJECTS_DICT
        for i in range(-1,-len(sys.path), -1):
            if 'dataset' in sys.path[i]:
                self.DATAROOT = sys.path[i]
                break
        # TRANSFORMATION PARAMETER:
        self.outward_spread_factor = Setting.OUTWARD_SPREAD
        self.inward_spread_factor = Setting.INWARD_SPREAD
        self.air_absorption = Setting.AIR_ABSORPTION
        self.DISTANCE_ENCODING = Setting.DISTANCE_ENCODING

        self.cache={}

            
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
        a = str(np.round(float(angle) )) if angle!=0 else str(float(0.0))
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


    def _propagate_snip(self, left_echoes, right_echoes, objects):
        temp_indexes = objects[:,2]==self.objects_dict['pole']
        rollings = self.cache['pole_rolls']
        for i, (le,re,r) in enumerate(zip(left_echoes[temp_indexes],right_echoes[temp_indexes],rollings)):
            left_echoes[temp_indexes][i] = np.roll(le, r)
            right_echoes[temp_indexes][i] =np.roll(re, r)
        temp_indexes = objects[:,2]==self.objects_dict['plant']
        rollings = self.cache['plant_rolls']
        for i, (le,re,r) in enumerate(zip(left_echoes[temp_indexes],right_echoes[temp_indexes],rollings)):
            left_echoes[temp_indexes][i] = np.roll(le, r)
            right_echoes[temp_indexes][i] =np.roll(re, r)
        self.cache.clear()
        return left_echoes, right_echoes
            

    def _get_snip(self, objects):
        ref_objects = np.empty(objects.shape)
        masks = np.empty((len(objects), self.raw_length))
        temp_indexes = objects[:,2]==self.objects_dict['pole']
        masks[ temp_indexes ] = self._pole_mask(objects[ temp_indexes][:,0])
        ref_objects[temp_indexes] = np.hstack((self.cache['pole_ref_distances'].reshape(-1,1),
                                               objects[temp_indexes][:,1:3]))
        temp_indexes = objects[:,2]==self.objects_dict['plant']
        masks[ temp_indexes ] = self._plant_mask(objects[ temp_indexes][:,0])
        ref_objects[temp_indexes] = np.hstack((self.cache['plant_ref_distances'].reshape(-1,1),
                                               objects[temp_indexes][:,1:3]))
        left_echoes, right_echoes = self._get_angle_interpolated_reference(ref_objects)
        return masks*left_echoes, masks*right_echoes


    def _pole_mask(self, distances):
        ref_start_indexes, ref_end_indexes, start_indexes, end_indexes = self._get_ref_indexes(distances, mode='pole')
        mask = np.zeros((len(distances),self.raw_length))
        attenuations = self._attenuation(ref_start_indexes, ref_end_indexes, start_indexes, end_indexes)
        for i, (s, e, attn) in enumerate(zip(ref_start_indexes, ref_end_indexes, attenuations)):
            mask[i][s:e] = attn
        return mask


    def _plant_mask(self, distances):
        ref_start_indexes, ref_end_indexes, start_indexes, end_indexes = self._get_ref_indexes(distances, mode='plant')
        mask = np.zeros((len(distances),self.raw_length))
        attenuations = self._attenuation(ref_start_indexes, ref_end_indexes, start_indexes, end_indexes)
        for i, (s, e, attn) in enumerate(zip(ref_start_indexes, ref_end_indexes, attenuations)):
            mask[i][s:e] = attn
        return mask

    def _attenuation(self, ref_sid, ref_eid, sid, eid):
        ref_distances = [self.DISTANCE_ENCODING[s:e] for s, e in zip(ref_sid, ref_eid)]
        to_distances = [self.DISTANCE_ENCODING[s:e] for s,e in zip(sid, eid)]
        attenuations = []
        for from_dist, to_dist in zip(ref_distances, to_distances):
            atmospheric = np.power(10, -self.air_absorption*2*(to_dist[0] - from_dist[0])/20)
            spreading = np.divide(from_dist , to_dist) ** (self.outward_spread_factor + self.inward_spread_factor)
            attenuations.append(atmospheric * spreading)
        return attenuations


    def _get_ref_indexes(self, distances, mode):
        start_dict, end_dict = (self.pole_starts,self.pole_ends) if mode=='pole' else (self.plant_starts,self.plant_ends)
        ref_starts, starts, ref_ends = [],[],[]
        ref_distances = self._calc_ref_distances(distances, mode=mode)
        for ref, dist in zip(ref_distances, distances):
            ref_starts.append(self.pole_starts[ref])
            ref_ends.append(self.pole_ends[ref])
            starts.append(self.pole_starts[ref] + dist - ref)
        ref_start_indexes = np.argmin(np.abs(self.DISTANCE_ENCODING - np.asarray(ref_starts).reshape(-1,1)),axis=1)
        ref_end_indexes = np.argmin(np.abs(self.DISTANCE_ENCODING - np.asarray(ref_ends).reshape(-1,1)), axis=1) + 1
        start_indexes = np.argmin(np.abs(self.DISTANCE_ENCODING - np.asarray(starts).reshape(-1,1)),axis=1)
        end_indexes = start_indexes + (ref_end_indexes - ref_start_indexes)
        self.cache[mode+'_rolls'] = start_indexes - ref_start_indexes
        return ref_start_indexes, ref_end_indexes, start_indexes, end_indexes
    

    def _calc_ref_distances(self, distances, mode):
        if type(distances) is not np.ndarray: distances = np.asarray(distances)
        distances = distances.reshape(-1,)
        if mode not in self.objects_dict.keys(): raise ValueError('_calc_ref_distance mode not valid')
        if mode=='pole':   standards = list(self.pole_starts.keys())
        elif mode=='plant':standards = list(self.plant_starts.keys())
        bar = np.concatenate(([0.], standards, [max(standards)+10]))
        comparison_matrix = (distances.reshape(-1,1) >= bar) ^ np.roll(( distances.reshape(-1,1) >= bar), -1)
        comparison_matrix[:,-1] = False
        result = np.clip(comparison_matrix @ bar, min(standards), max(standards))
        self.cache[mode+'_ref_distances'] = result
        return result
    

    def _get_background(self):
        left_echo, right_echo = self._get_reference([0.0, 0.0, 0])
        left_echo[:, self.emission_index:] = np.random.normal( self.bg_mu, self.bg_sigma, self.raw_length - self.emission_index )
        right_echo[:,self.emission_index:] = np.random.normal( self.bg_mu, self.bg_sigma, self.raw_length - self.emission_index )
        return left_echo, right_echo
        

class Render:
    def __init__(self, n_sample=Setting.N_SAMPLE):
        self.fetch = Retriever()
        self.cochlear = Cochlear.CochlearFilter()
        self.compressor = Compressors.Subsample(n_sample)
        self.cache = {}


    def get_envelope(self, objects, radians=True):
        if radians: objects = self._rad2deg(objects)
        scene = self._objects_to_scene(objects)
        envelope = self._envelope(scene)
        return envelope


    def get_compressed(self, objects, radians=True):
        envelope = self.get_envelope(objects, radians=radians)
        compressed_input = self._compress(envelope)
        return compressed_input
        

    def _objects_to_scene(self, objects):
        left_bg, right_bg = self.fetch._get_background()
        left_echoes, right_echoes = self.fetch._get_snip(objects)
        left_echoes, right_echoes = self.fetch._propagate_snip(left_echoes, right_echoes, objects)
        left_scene = left_bg + np.sum(left_echoes, axis=0)
        right_scene= right_bg +np.sum(right_echoes,axis=0)
        scene = {'left':left_scene.reshape(-1,), 'right':right_scene.reshape(-1,)}
        self.cache['left_echoes'] = left_echoes
        self.cache['right_echoes']= right_echoes 
        return scene


    def _envelope(self, scene):
        envelope = {}
        for key, val in zip(scene.keys(), scene.values()):
            envelope[key] = self.cochlear.transform(val)
        return envelope


    def _compress(self, envelope):
        compressed_input = {}
        for key, val in zip(envelope.keys(), envelope.values()):
            compressed_input[key] = self.compressor.transform(val)
        return compressed_input
    

    def _rad2deg(self,objects):
        if len(objects) == 0: return np.asarray(objects).reshape(-1,3)
        rad = objects[:,1]
        deg = np.degrees(rad)
        deg[deg>=180] = deg[deg>=180] - 360
        deg[deg<-180] = deg[deg>-180] + 360
        objects[:,1] = deg
        return objects

    
    def _reset(self):
        self.cache.clear()

