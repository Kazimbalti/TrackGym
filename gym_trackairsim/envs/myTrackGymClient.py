import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
# Change the path below to point to the directoy where you installed the AirSim PythonClient
#sys.path.append('C:/Users/Kjell/Google Drive/MASTER-THESIS/AirSimpy')

from TrackSimClient import *
vehicle_name = "Drone1"

class myTrackGymClient(MultirotorClient):

    def __init__(self,vehicle_name=''):        
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self, vehicle_name)
        self.enableApiControl(True,vehicle_name)
        self.armDisarm(True,vehicle_name)
    
        self.home_pos = self.simGetGroundTruthKinematics(vehicle_name).position
        print('home_pos',self.home_pos)
        self.home_ori = self.simGetGroundTruthKinematics(vehicle_name).orientation
        
        self.z = -1
   
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
        self.rotateByYawRateAsync(30, duration,vehicle_name)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration, vehicle_name=''):
        self.rotateByYawRateAsync(-30, duration,vehicle_name)
        start = time.time()
        return start, duration
    
    
    def take_action(self, action, vehicle_name='' ):
		
        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        while self.simGetGroundTruthKinematics(vehicle_name).position.z_val < -7.0:
            self. moveToZAsync(-6, 3, vehicle_name)
            time.sleep(1)
            print(self.simGetGroundTruthKinematics(vehicle_name).position.z_val, "and", x)
            x = x + 1
            if x > 10:
                return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False

        if action == 0:

            start, duration = self.straight(1, 4, vehicle_name)
        
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    return True    
                
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
            
        if action == 1:
         
            start, duration = self.yaw_right(0.8,vehicle_name)
            
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    return True
            
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
        if action == 2:
            
            start, duration = self.yaw_left(1,vehicle_name)
            
            while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    return True
                
            self.moveByVelocityAsync(0, 0, 0, 1, vehicle_name)
            self.rotateByYawRateAsync(0, 1, vehicle_name)
            
        return collided
    
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


    def AirSim_reset(self,vehicle_name=''):
        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True,vehicle_name )
        self.armDisarm(True,vehicle_name)
        time.sleep(0.5)
        self.moveToZAsync(self.z, 3, vehicle_name) 
        time.sleep(0.5)
        
     