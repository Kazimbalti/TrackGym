# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Ali Darwish
"""

import numpy as np
import gym
import gym.spaces
from gym import envs
#import gym_pull
#gym_pull.pull('')
#print(envs.registry.all())
import pickle
import gym_trackairsim.envs
import gym_trackairsim


import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Concatenate
from keras.optimizers import Adam
import keras.backend as K
 
from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
#from callbacks import *

from keras.callbacks import History, TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='TrackSimEnv-v1')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
INPUT_SHAPE = (30, 100)
WINDOW_LENGTH = 1
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


model = Sequential()
model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape, data_format = "channels_first"))
model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
model.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


train = True
'''
class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()
def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
def save_model(self):
            self.brain.save(self.weight_backup)
def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])
def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
'''
tb = TensorBoard(log_dir='logs')
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
try:
    memory = pickle.load(open("memory.pkl", "rb"))
except (FileNotFoundError, EOFError):
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)                        #reduce memmory


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)
# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=True, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])

load_pre_trained = True
if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    #dqn.load_weights('dqn_{}_weights.h5f'.format(args.env_name))
    if load_pre_trained:
        try:
            dqn.load_weights('dqn_{}_weights.h5f'.format(args.env_name))
        except (OSError):
            logger.warning ("File not found")

    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
     
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=10000)]
    callbacks += [tb]
    dqn.fit(env, callbacks=callbacks, nb_steps=100002, visualize=False, verbose=2, log_interval=10000)#250000
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)
    pickle.dump(memory, open("memory.pkl", "wb"))

else:

    dqn.load_weights('dqn_{}_weights.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=10, visualize=False)
