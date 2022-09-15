import os
import sys
import pathlib

if pathlib.Path(os.path.abspath(__file__)).parents[2] not in sys.path:
    sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parents[2]))

import time

from Gym.BatSnake_v0.Environment import DiscreteAction

import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(2)

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment

from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform, Constant, VarianceScaling
from tensorflow.keras.activations import relu, linear


HIDDEN_LAYER_PARAMS = (128, 128, 128, 64)

TRAINING_STEPS = 2_000_000
INITIAL_COLLECTION_STEPS = 1_000
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 500_000

PARALLEL_CALLS = 32
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
LOG_STEPS_INTERVAL = 20_000
NUMBER_OF_EVAL_EPISODES = 10
EVAL_STEPS_INTERVAL = 20_000

STARTING_EPSILON = 0.8
EPSILON_DECAY_COUNT = 2_000_000
ENDING_EPSILON = 0.1
DISCOUNT_FACTOR = 0.999
TD_ERROR_LOSS_FUNCTION = common.element_wise_squared_loss
TRAIN_STEP_COUNTER = 0

DATE = '09.15.22'
NOTES =''
CHECKPOINT_DIRECTORY = 'TrainedAgents'
TIME_LIMIT = 250

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

def summon_agent(environment, q_network, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                starting_epsilon=STARTING_EPSILON, epsilon_delay_count=EPSILON_DECAY_COUNT, ending_epsilon=ENDING_EPSILON,
                gamma = DISCOUNT_FACTOR, td_loss_fn = TD_ERROR_LOSS_FUNCTION):
    return dqn_agent.DqnAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        q_network=q_network,
        optimizer=optimizer,
        epsilon_greedy=starting_epsilon,
        epsilon_decay_end_count=epsilon_delay_count,
        epsilon_decay_end_value=ending_epsilon,
        gamma=gamma, td_errors_loss_fn=td_loss_fn, train_step_counter= tf.Variable(TRAIN_STEP_COUNTER))

def summon_replay_buffer(agent, environment, max_length):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = agent.collect_data_spec, batch_size= environment.batch_size, 
        max_length=max_length)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))

def collect_data(environment, policy, buffer, steps):
    for _ in range(steps):
        collect_step(environment, policy, buffer)

def compute_average_return(environment, policy, number_of_episodes, getcache=False):
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
    if getcache: return (total_return/number_of_episodes).numpy()[0], cache
    else: return (total_return/number_of_episodes).numpy()[0]


def train_v1(init_policy=None):
    phase = 0
    py_env = DiscreteAction(time_limit = TIME_LIMIT, phase=phase, log=True)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    #action_tensor_spec = tensor_spec.from_spec(py_env.action_spec())
    num_actions = 2 #action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    q_net = q_network(hidden_layer_params=HIDDEN_LAYER_PARAMS, number_of_actions=num_actions)
    
    eval_py_env = DiscreteAction(time_limit = TIME_LIMIT, phase=phase, log=True)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = summon_agent(tf_env, q_net)
    agent.initialize()
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    
    ### COLLECT INIT REPLAY BUFFER

    replay_buffer = summon_replay_buffer(agent=agent, environment=tf_env, max_length=REPLAY_BUFFER_MAX_LENGTH)
    if init_policy is None: init_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    collect_data(tf_env, init_policy, replay_buffer, INITIAL_COLLECTION_STEPS)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=PARALLEL_CALLS,
        sample_batch_size=BATCH_SIZE,
        num_steps=2,
        single_deterministic_pass=False).prefetch(PARALLEL_CALLS)
    iterator = iter(dataset)

    ### TRAINING START HERE.
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(TRAIN_STEP_COUNTER)
    
    #average_return = compute_average_return(eval_tf_env, eval_policy, NUMBER_OF_EVAL_EPISODES, getcache=False)
    returns = []
    losses = []
    training_episodes = []
    training_steps = []
    phases = []
    tic = time.time()
    times = []

    print('Current Dir:', os.getcwd())
    if CHECKPOINT_DIRECTORY=='':
        prompt = input('ENTER SAVE PATH FOR POLICY')
        save_dir = os.path.join(os.getcwd(), prompt+'/'+DATE+NOTES)
    else: save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIRECTORY+'/'+DATE+NOTES)
    print('Policy will be save to:\n', save_dir)
    policy_dir = os.path.join(save_dir, 'policy')
    if not os.path.exists(policy_dir): os.makedirs(policy_dir)
    tf_policy_saver = policy_saver.PolicySaver(eval_policy)
    

    for _ in range(TRAINING_STEPS):
        collect_data(tf_env, collect_policy, replay_buffer, COLLECT_STEPS_PER_ITERATION)
        experience, used_info = next(iterator)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()

        if step % LOG_STEPS_INTERVAL == 0:
            print('step={0}: loss={1}'.format(step, train_loss))
            losses.append(train_loss)
        if step % EVAL_STEPS_INTERVAL == 0:
            # save policy:
            if step%1e6==0: tf_policy_saver.save( os.path.join(policy_dir,str(int(step/1e6))+'M_steps'))
            print('--- Evaluation ---')
            eval_py_env.episode = 0
            average_return = compute_average_return(eval_tf_env, eval_policy, NUMBER_OF_EVAL_EPISODES, getcache=False)
            time_elapse = np.round(time.time() - tic)
            print('step={0}: return={1}. Time elapse: {2}h{3}m'.format(step, average_return, int(np.floor(time_elapse/3600)), int(np.round(time_elapse%3600/60))))
            returns.append(average_return)
            training_episodes.append(py_env.episode)
            training_steps.append(step)
            phases.append(phase)
            times.append(time_elapse)
            if np.mean(returns[-10:])>0.8 and phase<2:
                phase += 0
                print('TRAINING IN PHASE {0}'.format(phase))
                py_env.cache['phase']=phase
                eval_py_env.cache['phase']=phase
            np.savez(save_dir+'/return_loss_log.npz', losses=np.asarray(losses), returns=np.asarray(returns),
                    episodes=np.asarray(training_episodes), steps=np.asarray(training_steps), phases=np.asarray(phases), times=np.asarray(times))
            np.savez(save_dir+'/training_log.npz', episode=np.asarray(py_env.records['episode']), steps=np.asarray(py_env.records['steps']),
                    hit=np.asarray(py_env.records['hit']), success=np.asarray(py_env.records['success']), 
                    timeout=np.asarray(py_env.records['timeout']), outbound=np.asarray(py_env.records['outbound']))



    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2,1)
    ax[0].plot(training_steps,returns)
    ax[0].set_ylabel('average episode return')
    ax[0].set_xlabel('training steps')
    ax[1].plot(losses)
    ax[1].set_ylabel(np.linspace(np.min(training_steps), np.max(training_steps), len(losses) ),'loss')
    ax[1].set_xlabel('training steps')
    plt.show()

    return None

if __name__=='__main__':
    train_v1()