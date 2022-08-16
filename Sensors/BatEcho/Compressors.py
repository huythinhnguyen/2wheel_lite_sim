import numpy as np
from . import Setting


class Subsample:
    def __init__(self, n_sample=Setting.N_SAMPLE, quiet_threshold=Setting.QUIET_THRESHOLD,
                 truncate=0, normalized=Setting.COMPRESSOR_NORMALIZED): # n_sample=140
        self.normalized = normalized
        self.n_sample = n_sample
        self.quiet_threshold = quiet_threshold
        self.quiet = True
        self.truncate = truncate
        self.quiet_normalizer = Setting.QUIET_NORMALIZER
        

    def transform(self, data):
        if type(data) != np.ndarray: data=np.asarray(data)
        data = np.mean(data.reshape(-1,self.n_sample), axis=1)
        quiet = self.quiet_threshold
        if self.normalized:
            data = data/np.max(data)
            quiet = self.quiet_normalizer
        if self.quiet: data[data<quiet] = 0.
        return data[:int(self.n_sample-self.truncate)]
