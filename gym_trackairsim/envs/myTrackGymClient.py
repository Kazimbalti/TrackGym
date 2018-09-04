import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
import pprint
 
from TrackSimClient import *


class myTrackGymClient(MultirotorClient):

    def __init__(self):        
        self.drone1_vehicle_name = "Drone1"
        self.target1_vehicle_name = "Target1"
        self.z =-2
        self.max_z= self.z - 1  #maximum allowable z value as the drone goofs up sometimes.
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True,self.drone1_vehicle_name)
        self.enableApiControl(True,self.target1_vehicle_name)################
        self.armDisarm(True,self.drone1_vehicle_name)
        self.armDisarm(True,self.target1_vehicle_name)################
        #f1 = self.takeoffAsync(self.drone1_vehicle_name)
        #f2 = self.takeoffAsync(self.target1_vehicle_name)
        #f1.join()
        #f2.join()
        state1 = self.getMultirotorState(self.drone1_vehicle_name)
        s = pprint.pformat(state1)
        print("state: %s" % s)
        state2 = self.getMultirotorState(self.target1_vehicle_name)
        s = pprint.pformat(state2)
        print("state: %s" % s)
        self.home_pos = self.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        print('home_pos',self.home_pos)
        #self.target1_home_pos = self.simGetGroundTruthKinematics(self.target1_vehicle_name).position###########
        #print('target1_home_pos',self.target1_home_pos)#####################
        self.home_ori = self.simGetGroundTruthKinematics(self.drone1_vehicle_name).orientation
        print('home_ori',self.home_ori)
        #self.target1_home_ori = self.simGetGroundTruthKinematics(self.target1_vehicle_name).orientation################
        #print('target1_home_ori',self.target1_home_ori)
        #f1 = self.moveToPositionAsync(-5, 5, -10, 5, self.drone1_vehicle_name)
        #f2 = self.moveToPositionAsync(5, -5, -10, 5, self.target1_vehicle_name)
        #f1.join()
        #f2.join()
        
   
    def straight(self, duration,speed, vehicle_name=''):
        #pitch, roll, yaw  = client.getRollPitchYaw()
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        #quad_state = client.getMultirotorState().kinematics_estimated.position
    #quad_state.z_val
        self.moveByVelocityZAsync(vx, vy,self.z, duration, DrivetrainType.ForwardOnly, YawMode(False, 0),  vehicle_name).join()
        start = time.time()
        return start, duration
    def yaw_right(self, duration,vehicle_name=''):
        self.rotateByYawRateAsync(15, duration,vehicle_name)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration, vehicle_name=''):
        self.rotateByYawRateAsync(-15, duration,vehicle_name)
        start = time.time()
        return start, duration
    
    
    def take_action(self, action, vehicle_name):
        #f1 = self.takeoffAsync(vehicle_name)
        #f1.join()
        #check if copter is on level cause sometimes it goes up without a reason
        self.enableApiControl(True,vehicle_name)################
        self.armDisarm(True,vehicle_name)
        x = 0
        while self.simGetGroundTruthKinematics(vehicle_name).position.z_val < self.max_z:
            self. moveToZAsync(self.max_z, 3, vehicle_name)
            time.sleep(1)
            print(self.simGetGroundTruthKinematics(vehicle_name).position.z_val, "and", x)
            x = x + 1
            if x > 10:
                return True , 'None'       
        
    
        start = time.time()
        duration = 0 
        
        collided = False
        collided_with = ''
        if action == 0:

            start, duration = self.straight(1, 4, vehicle_name)
        
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    print('collision_info',self.simGetCollisionInfo(vehicle_name).object_name)
                    collided_with = self.simGetCollisionInfo(vehicle_name).object_name 
                    return True, collided_with 
                    
                
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
            
        if action == 1:
         
            start, duration = self.yaw_right(0.8,vehicle_name)
            
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    print('collision_info',self.simGetCollisionInfo(vehicle_name).object_name)
                    collided_with = self.simGetCollisionInfo(vehicle_name).object_name 
                    return True, collided_with 
            
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
        if action == 2:
            
            start, duration = self.yaw_left(1,vehicle_name)
            
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    print('collision_info',self.simGetCollisionInfo(vehicle_name).object_name)
                    collided_with = self.simGetCollisionInfo(vehicle_name).object_name 
                    return True, collided_with 
                
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
        return collided, collided_with

    def take_initial_action(self, vehicle_name=''):
        self.takeoffAsync(vehicle_name=vehicle_name).join()
        self. moveToZAsync(self.max_z, 3, vehicle_name)
        self.hoverAsync(vehicle_name).join()
 

    def goal_direction(self, goal, pos, vehicle_name=''):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
    def getScreenDepthVis(self, track, vehicle_name=''):

        responses = self.simGetImages([ImageRequest(0, ImageType.DepthPerspective, True, False)],vehicle_name)
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        
        
        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
                
        cut = small[20:40,:]
        
        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) + 255
        info_section[9,:] = 0
        
        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-1:line+2]  = 0
        elif line == 0:
            info_section[:,0:3]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 0
            
        total = np.concatenate((info_section, cut), axis=0)
            
        #cv2.imshow("Test", total)
        #cv2.waitKey(0)
        
        return total
    
    def AirSim_reset(self):
        #self.armDisarm(False, vehicle_name)
        
        self.reset()
        self.enableApiControl(True,self.drone1_vehicle_name)################
        self.armDisarm(True,self.drone1_vehicle_name)
        self.enableApiControl(True,self.target1_vehicle_name)################
        self.armDisarm(True,self.target1_vehicle_name)
        # that's enough fun for now. let's quit cleanly
        #self.enableApiControl(False, vehicle_name)
"""
    def AirSim_reset(self,vehicle_name=''):
        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True,vehicle_name )
         
        self.armDisarm(True,vehicle_name)
         
        time.sleep(1)
        self.moveToZAsync(self.z, 3,vehicle_name) 
         
        time.sleep(3)
"""     

