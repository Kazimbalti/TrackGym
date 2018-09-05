#env def as described by openAI
#http://rllab.readthedocs.io/en/latest/user/implement_env.html
import re
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

        global trackgym
        trackgym = myTrackGymClient()
        #self.client = myTrackGymClient()
        # left depth, center depth, right depth, yaw
        self.drone1_vehicle_name = "Drone1"
        self.target1_vehicle_name = "Target1"
        self.z =-2
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100), dtype=np.float32)
        self.state = np.zeros((30, 100), dtype=np.uint8) 
        self.action_space = spaces.Discrete(3)
        self.drone1_init=[0,0,-2]
        self.target1_init=[6,-2,-2]
        self.goal1 =np.subtract (self.target1_init,self.drone1_init)# [x,y,z]  the first target drone obtained by flying the drone around , should be d = 12.106197 
        print('initial goal', self.goal1)
        self.episodeN = 0
        self.stepN = 0 
        self.allLogs = { 'reward':[0] }
        self.drone1_position = trackgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        self.target1_position = trackgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position
        self.allLogs['distance1'] = [np.sqrt(np.power((self.goal1[0]-self.drone1_position.x_val),2) + np.power((self.goal1[1]-self.drone1_position.y_val),2)+ np.power((self.goal1[2]-self.drone1_position.z_val),2))] 

        self.allLogs['track1'] = [trackgym.goal_direction(self.goal1, self.drone1_position, self.drone1_vehicle_name)]

        self.allLogs['action'] = [1] 
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def computeReward(self, current_position, heading_to_target,velocity):
		#track_current_position has the heading to goal 
		#compute the current distance to the target in 3d
        distance1_current_position = np.sqrt(np.power((self.goal1[0]-current_position.x_val),2) + np.power((self.goal1[1]-current_position.y_val),2)+ np.power((self.goal1[2]-current_position.z_val),2))
        #get the previous distance info
        distance1_before = self.allLogs['distance1'][-1] 
        reward_speed = (np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val]) - 0.5)
        r = -1 
        r = r + (distance1_before - distance1_current_position) + reward_speed# - abs(heading_to_target/10)


        # make sure the drone is moving and isn't stuck in the env
        # Check if the drone is stuck in the last 10 positions using the velocity
        #if self.stepN > 11 and abs(sum(velocity.x_val, velocity.y_val)) < 1:
            #r = r - 60
        print('spying on rewards r', r)
        print('spying on heading abs(heading_to_target/10)',abs(heading_to_target/10))
        print('spying on reward_speed ',reward_speed)
        return r, distance1_current_position
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
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided, collided_with = trackgym.take_action(action, self.drone1_vehicle_name)
        
        #print('#############################################')
        print('passed collided_with value', collided_with)
        #print('passed collision value', collided)
        #print('#############################################')
        current_position = trackgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        velocity = trackgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).linear_velocity
        print('current_position',current_position)
        #traget1_current_position = trackgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position
        #print('traget1_current_position',traget1_current_position)
        track1 = trackgym.goal_direction(self.goal1, current_position, self.drone1_vehicle_name) #self, goal1, pos, vehicle_name=''

        if collided == True:
            done = True
            if re.match(r'Target\d',collided_with):
                reward = +100.0
                distance1 = np.sqrt(np.power((self.goal1[0]-current_position.x_val),2) + np.power((self.goal1[1]-current_position.y_val),2))
                print('got drone', collided_with)
            else:
                reward = -100.0
                distance1 = np.sqrt(np.power((self.goal1[0]-current_position.x_val),2) + np.power((self.goal1[1]-current_position.y_val),2))
        else: 
            done = False
            reward, distance1 = self.computeReward(current_position, track1, velocity)
        
        # intercepted target
        if distance1 < 2:
            done = True
            reward = 100.0
        
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance1', distance1)
        self.addToLog('track1', track1)      
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -50:
            done = True
        
        sys.stdout.write("\r\x1b[K{}/{}==>reward/rsum: {:.1f}/{:.1f}   \t track1:{:.0f}  action:{:.0f} distance1:{:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, track1, action, distance1))
        sys.stdout.flush()
        
        info = {"x_pos" : current_position.x_val, "y_pos" : current_position.y_val}
        self.state =trackgym.getScreenDepthVis(track1, self.drone1_vehicle_name)

        return self.state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        trackgym.AirSim_reset()
        trackgym.take_initial_action(self.target1_vehicle_name)
        trackgym.take_initial_action(self.drone1_vehicle_name)
        
         
        if  1 < self.episodeN < 6:
            print('333333333333333333333 moving level 1 3333333333333333333333333')
            target1_new_position=[1,-3,self.z]
            trackgym.moveToPositionAsync(target1_new_position[0],target1_new_position[1],target1_new_position[2], 5, vehicle_name="Target1").join()
            #trackgym.moveToPositionAsync(4,1,-2, 5, vehicle_name="Target1").join()
            print('Done moving!!!!!!!!!')
            target1_position = trackgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position
            #self.goal1 =np.add (self.target1_init,[target1_position.x_val,target1_position.y_val,target1_position.z_val])# [x,y,z]  the first target drone obtained by flying the drone around , should be d = 12.106197 
            self.goal1 = [self.target1_init[0]+target1_position.x_val,self.target1_init[1]+target1_position.y_val,self.z ]
            print('new goal level 1', self.goal1)
            
        if  5 < self.episodeN < 10:
            print('333333333333333333333 moving level 2 3333333333333333333333333')
            target1_new_position=[-2,3,self.z]
            trackgym.moveToPositionAsync(target1_new_position[0],target1_new_position[1],target1_new_position[2], 5, vehicle_name="Target1").join()
            #trackgym.moveToPositionAsync(4,1,-2, 5, vehicle_name="Target1").join()
            print('Done moving!!!!!!!!!')
            target1_position = trackgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position
            #self.goal1 =np.add (self.target1_init,[target1_position.x_val,target1_position.y_val,target1_position.z_val])# [x,y,z]  the first target drone obtained by flying the drone around , should be d = 12.106197 
            self.goal1 = [self.target1_init[0]+target1_position.x_val,self.target1_init[1]+target1_position.y_val,self.z ]
            print('new goal level 2', self.goal1) 
 
        if  9 < self.episodeN < 15:
            print('333333333333333333333 moving level 3 3333333333333333333333333')
            target1_new_position=[-4,1,self.z]
            trackgym.moveToPositionAsync(target1_new_position[0],target1_new_position[1],target1_new_position[2], 5, vehicle_name="Target1").join()
            #trackgym.moveToPositionAsync(4,1,-2, 5, vehicle_name="Target1").join()
            print('Done moving!!!!!!!!!')
            target1_position = trackgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position
            #self.goal1 =np.add (self.target1_init,[target1_position.x_val,target1_position.y_val,target1_position.z_val])# [x,y,z]  the first target drone obtained by flying the drone around , should be d = 12.106197 
            self.goal1 = [self.target1_init[0]+target1_position.x_val,self.target1_init[1]+target1_position.y_val,self.z ]
            print('new goal level 3', self.goal1)
        
        self.stepN = 0
        self.episodeN += 1
        
        self.allLogs = { 'reward': [0] }
        self.drone1_position = trackgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        self.allLogs['distance1'] = [np.sqrt(np.power((self.goal1[0]-self.drone1_position.x_val),2) + np.power((self.goal1[1]-self.drone1_position.y_val),2)+ np.power((self.goal1[2]-self.drone1_position.z_val),2))] 
        self.allLogs['track1'] = [trackgym.goal_direction(self.goal1, self.drone1_position, self.drone1_vehicle_name)]

        self.allLogs['action'] = [1]
        
        print("")
        
        #current_position = self.initial_position
        track1 = trackgym.goal_direction(self.goal1, self.drone1_position, self.drone1_vehicle_name)
        self.state = trackgym.getScreenDepthVis(track1,self.drone1_vehicle_name)
        


        
        return self.state

    #def _render(self, mode='human', close=False):
		#return