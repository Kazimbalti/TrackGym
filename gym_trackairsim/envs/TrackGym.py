#env def as described by openAI
#http://rllab.readthedocs.io/en/latest/user/implement_env.html

import gym 
from gym import spaces
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box
from gym.utils import seeding

import random
import logging
import numpy as np
from TrackSimClient import *
from gym_trackairsim.envs.myTrackGymClient import *
logger = logging.getLogger(__name__)

class TrackSimEnv(gym.Env):
    trackgym = None

    def __init__(self):
        '''
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        '''
        # left depth, center depth, right depth, yaw
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100), dtype=np.float32)
        self.state = np.zeros((30, 100), dtype=np.uint8) 
        self.action_space = spaces.Discrete(3)
        self.goal =[100, 2]# 	[221.0, -9.0] # global xy coordinates
        self.episodeN = 0
        self.stepN = 0 
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [10]# [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1] 
        self._seed()
        global trackgym
        trackgym = myTrackGymClient()
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def computeReward(self, now, track_now):
		
		#get exact coordiantes of the tip
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        distance_before = self.allLogs['distance'][-1] 
        r = -1 
        """
        if abs(distance_now - distance_before) < 0.0001:
            r = r - 2.0
            #Check if last 4 positions are the same. Is the copter actually moving?
            if self.stepN > 5 and len(set(self.allLogs['distance'][len(self.allLogs['distance']):len(self.allLogs['distance'])-5:-1])) == 1: 
                r = r - 50
        """  
            
        r = r + (distance_before - distance_now)
            
        return r, distance_now
    '''
    def compute_reward(quad_state, quad_vel, collision_info):
        thresh_dist = 3.5#7
        beta = 1

        z = -3
        #pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]
        pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

        if collision_info.has_collided:
            reward = -100
        else:    
            dist = 10000000
            for i in range(0, len(pts)-1):
                dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

            #print(dist)
            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = (math.exp(-beta*dist) - 0.5) 
                reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
                reward = reward_dist + reward_speed

        return reward
    '''
    def _step(self, action, vehicle_name=''):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided = trackgym.take_action(action, vehicle_name)
        
        now = trackgym.simGetGroundTruthKinematics(vehicle_name).position
        track = trackgym.goal_direction(self.goal, now, vehicle_name) #self, goal, pos, vehicle_name=''

        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        elif collided == 99:
            done = True
            reward = 0.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        else: 
            done = False
            reward, distance = self.computeReward(now, track)
        
        # Youuuuu made it
        if distance < 3:
            done = True
            reward = 100.0
        
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance', distance)
        self.addToLog('track', track)      
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -100:
            done = True
        
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t {:.0f}  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, track, action))
        sys.stdout.flush()
        
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        self.state = trackgym.getScreenDepthVis(track, vehicle_name)

        return self.state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self,vehicle_name=''):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        trackgym.AirSim_reset(vehicle_name)
        
        self.stepN = 0
        self.episodeN += 1
        
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [10]# [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        
        print("")
        
        now = trackgym.simGetGroundTruthKinematics(vehicle_name).position
        track = trackgym.goal_direction(self.goal, now, vehicle_name)
        self.state = trackgym.getScreenDepthVis(track,vehicle_name)
        
        return self.state