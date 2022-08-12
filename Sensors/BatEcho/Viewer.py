import numpy as np


class FoV:
    def __init__(self, pose=None, linear=3., angular=np.pi/2):
        self.linear = linear
        self.angular_range = [-0.5*angular, 0.5*angular]
        self.objects_diameter={'pole': 0.1, 'plant': 0.3}
        self.objects_dict={'pole':1, 'plant':2}
        if pose is None: pose = np.zeros(3)

        
    def _in_view_filter(self, objects, pose=None, input_format='cartesian'):
        if input_format=='cartesian': objects = self._polar_objects(objects,pose)
        conditions = (objects[:,0]<self.linear) & \
            (objects[:,1]>=self.angular_range[0]) & \
            (objects[:,1]<=self.angular_range[1])
        inview = objects[conditions]
        return inview


    def _update_pose(self, pose):
        if type(pose) is list: pose=np.asarray(pose)
        self.pose = pose.reshape(3,)


    def _polar_objects(self, objects, pose=None):
        if pose is not None: self._update_pose(pose)
        A = objects[:,:2] - self.pose[:2]
        B = np.empty(A.shape)
        B[:,0] = np.linalg.norm(A, axis=1)
        B[:,1] = self._wrap2pi(np.arctan2(A[:,1],A[:,0]) - self.pose[2])
        return np.hstack((B, objects[:,2].reshape(-1,1)))


    def _wrap2pi(self, angles):
        angles[angles>=np.pi] = angles[angles>=np.pi] - 2*np.pi
        angles[angles<-np.pi] = angles[angles<-np.pi] + 2*np.pi
        return angles


    def _obscure(self, objects):
        sortarg = np.argsort(objects[:,0])
        distances = objects[sortarg][:,0]
        angles = objects[sortarg][:,1]
        klasses = objects[sortarg][:,2]
        diameters = []
        for k in klass:
            dia_temp=self.objects_diameter[list(self.objects_dict.keys())[list(self.objects_dict.values()).index(k)]]
            diameters.append(dia_temp)
        diameters = np.asarray(diameters)
        da = np.arcsin(np.divide(diameters, 2*distances))
        removed = []
        for i, (a, delta) in enumerate(zip(angles, da)):
            if i == len(angles) - 1: break
            for a2, delta2, idx2 in zip(angles[i:], da[i:], sortarg[i:]):
                condition =((a2-delta2)>=(a-delta))and((a2+delta2)<=(a-delta))
                if condition: removed.append(idx2)
        return np.delete(objects, removed, axis=0)
