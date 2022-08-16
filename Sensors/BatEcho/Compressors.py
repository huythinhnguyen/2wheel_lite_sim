import numpy as np
from . import Setting


class Subsample:
    def __init__(self, n_sample=Setting.N_SAMPLE, quiet_threshold=Setting.QUIET_THRESHOLD, truncate=0): # n_sample=140
        self.n_sample = n_sample
        self.quiet_threshold = quiet_threshold
        self.quiet = True
        self.truncate = truncate
        

    def transform(self, data):
        if type(data) != np.ndarray: data=np.asarray(data)
        data = np.mean(data.reshape(-1,self.n_sample), axis=1)
        if self.quiet: data[data<self.quiet_threshold] = 0.
        return data[:int(self.n_sample-self.truncate)]
