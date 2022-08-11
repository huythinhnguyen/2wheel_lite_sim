import numpy as np


class FoV:
    def __init__(self, pose=None, linear=3., angular=np.pi/2):
        self.linear = linear
        self.angular_range = [-0.5*angular, 0.5*angular]
        self.objects_diameter{'pole': 0.1, 'plant': 0.3}
        if pose is None: pose = np.zeros(3)

        
    def _in_view_filter(self, objects, pose=None):
        objects = self._polar_objects(objects,pose)
        conditions = (objects[:,0]<self.linear) & \
            (objects[:,1]>=self.angular_range[0]) & \
            (objects[:,1]<=self.angular_range[1])
        inview = objects[conditions]
        return inview


    def _update_pose(self, pose):
        if type(pose) is list: pose=np.asarray(pose)
        self.pose = pose.reshape(3,)


    def _polar_objects(self, objects, pose=None):
        if pose not None: self._update_pose(pose)
        A = objects[:,:2] - self.pose[:2]
        B = np.empty(A.shape)
        B[:,0] = np.linalg.norm(A, axis=1)
        B[:,1] = self._wrap2pi(np.arctan2(A[:,1],A[:,0]) - self.pose[2])
        return np.hstack((B, objects[:,2].reshape(-1,1)))


    def _wrap2pi(self, angles):
        angles[angles>=np.pi] = angles[angles>=np.pi] - 2*np.pi
        angles[angles<-np.pi] = angles[angles<-np.pi] + 2*np.pi
        return angles
