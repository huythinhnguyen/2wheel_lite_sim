import os
import sys
import pathlib

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(pathlib.Path(os.path.abspath(__file__)).parents[2])

import numpy as np
import tensorflow as tf
from .Environment import DiscreteAction

from tf_agents.policies import random_tf_policy
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform, Constant, VarianceScaling
from tensorflow.keras.activations import relu, linear


### Build some Function building model here!
### Build some convenience saver if needed. :D
### ADD Setting if needed


if __name__=='__main__':
    pass