import numpy as np


class Subsample:
    def __init__(self, n_sample=125, quiet_threshold=0.7): # n_sample=140
        self.n_sample = n_sample
        self.quiet_threshold = quiet_threshold
        self.quiet = True


    def transform(self, data):
        if type(data) != np.ndarray: data=np.asarray(data)
        data = np.mean(data.reshape(-1,self.n_sample), axis=1)
        if self.quiet: data[data<self.quiet_threshold] = 0.
        return data
