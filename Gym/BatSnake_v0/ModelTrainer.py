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


HIDDEN_LAYER_PARAMS = (128, 128, 128, 64)

TRAINING_STEPS = 100_000
INITIAL_COLLECTION_STEPS = 1000
COLLECT_STEPS_PER_ITERATION = 1
RELAY_BUFFER_MAX_LENGTH = 500_000

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
LOG_STEPS_INTERVAL = 2_000
NUMBER_OF_EVAL_EPISODES = 10
EVAL_STEPS_INTERVAL = 10_000

STARTING_EPSILON = 0.8


### Build some Function building model here!
### Build some convenience saver if needed. :D
### ADD Setting if needed

def hidden_layer(number_of_units):
    return Dense(number_of_units, activation=relu,
                  kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))

def value_layer(number_of_actions):
    return Dense(number_of_actions, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05), bias_initializer=Constant(-0.2))

def q_network(hidden_layer_params, number_of_actions):
    hidden_net = [hidden_layer(number_of_units) for number_of_units in hidden_layer_params]
    q_layer = value_layer(number_of_actions)
    return sequential.Sequential(hidden_net + [q_layer])

def relay_buffer(agent, environment, relay_buffer_max_length):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = agent.collect_data_spec, batch_size= environment.batch_size, 
        max_length=relay_buffer_max_length)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))

def collect_data(environment, policy, buffer, steps):
    for _ in range(steps):
        collect_step(environment, policy, buffer)

def compute_average_return(environment, policy, number_of_episodes, cache=False):
    total_return = 0
    cache = []
    for episode in range(number_of_episodes):
        time_step = environment._reset()
        episode_return = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment._step(action_step.action)
            episode_return += time_step.reward
        else:
            total_return += episode_return
            cache.append(episode_return.numpy()[0])
    if cache: return (total_return/number_of_episodes).numpy()[0], cache
    else: return (total_return/number_of_episodes).numpy()[0]



if __name__=='__main__':
    pass