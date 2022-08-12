import numpy as np
from .Spatializer import DISTANCE_ENCODING

class Subsample:
    def __init__(self, n_sample=125, quiet_threshold=0.7, truncate=0): # n_sample=140
        self.n_sample = n_sample
        self.quiet_threshold = quiet_threshold
        self.quiet = True
        self.truncate = truncate
        self.DISTANCE_REFERENCE = np.mean(DISTANCE_ENCODING.reshape(-1,self.n_sample),
                                          axis=1)[:int(self.n_sample-self.truncate)]


    def transform(self, data):
        if type(data) != np.ndarray: data=np.asarray(data)
        data = np.mean(data.reshape(-1,self.n_sample), axis=1)
        if self.quiet: data[data<self.quiet_threshold] = 0.
        return data[:int(self.n_sample-self.truncate)]


    def _reset(self, n_sample=None):
        if n_sample is not None: self.n_sample=n_sample
        self.DISTANCE_REFERENCE = np.mean(DISTANCE_ENCODING.reshape(-1,self.n_sample),
                                          axis=1)[:int(self.n_sample-self.truncate)]
