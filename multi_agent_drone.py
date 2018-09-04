import setup_path 
#import airsim
import TrackSimClient
from TrackSimClient import *
import numpy as np
import os
import tempfile
import pprint
from gym_trackairsim.envs.myTrackGymClient import *
# Use below in settings.json with Blocks environment
"""
{
	"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
	"SettingsVersion": 1.2,
	"SimMode": "Multirotor",
	"ClockSpeed": 1,
	
	"Vehicles": {
		"Drone1": {
		  "VehicleType": "SimpleFlight",
		  "X": 4, "Y": 0, "Z": -2
		},
		"Drone2": {
		  "VehicleType": "SimpleFlight",
		  "X": 8, "Y": 0, "Z": -2
		}

    }
}
"""
#global trackgym
#client = TrackSimClient
# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Target1")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Target1")

#airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Target1")
f1.join()
f2.join()

state1 = client.getMultirotorState(vehicle_name="Drone1")
s = pprint.pformat(state1)
print("state: %s" % s)
state2 = client.getMultirotorState(vehicle_name="Target1")
s = pprint.pformat(state2)
print("state: %s" % s)

#airsim.wait_key('Press any key to move vehicles')
f1 = client.moveToPositionAsync(-5, 5, -10, 5, vehicle_name="Drone1")
f2 = client.moveToPositionAsync(5, -5, -10, 5, vehicle_name="Target1")
f1.join()
f2.join()
f2 = client.moveToPositionAsync(11, -5, -13, 5, vehicle_name="Drone2")
f2.join()
f2 = client.moveToPositionAsync(5, -5, -10, 5, vehicle_name="Target1")
f2.join()
f2 = client.moveToPositionAsync(15, -15, -15, 5, vehicle_name="Drone2")
f2.join()
f2 = client.moveToPositionAsync(5, -5, -10, 5, vehicle_name="Target1")
f2.join()

#airsim.wait_key('Press any key to take images')
# get camera images from the drone
responses1 = client.simGetImages([
    TrackSimClient.ImageRequest("0", TrackSimClient.ImageType.DepthVis),  #depth visualization image
    TrackSimClient.ImageRequest("1", TrackSimClient.ImageType.Scene, False, False)], vehicle_name="Drone1")  #scene vision image in uncompressed RGBA array
print('Drone1: Retrieved images: %d' % len(responses1))
responses2 = client.simGetImages([
    TrackSimClient.ImageRequest("0", TrackSimClient.ImageType.DepthVis),  #depth visualization image
    TrackSimClient.ImageRequest("1", TrackSimClient.ImageType.Scene, False, False)], vehicle_name="Target1")  #scene vision image in uncompressed RGBA array
print('Target1: Retrieved images: %d' % len(responses2))

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses1 + responses2):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        #TrackSimClient.write_pfm(os.path.normpath(filename + '.pfm'), TrackSimClient.get_pfm_array(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        #TrackSimClient.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
        img_rgba = np.flipud(img_rgba) #original image is flipped vertically
        img_rgba[:,:,1:2] = 100 #just for fun add little bit of green in all pixels
        #TrackSimClient.write_png(os.path.normpath(filename + '.greener.png'), img_rgba) #write to png

#TrackSimClient.wait_key('Press any key to reset to original state')

client.armDisarm(False, "Drone1")
client.armDisarm(False, "Target1")
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, "Drone1")
client.enableApiControl(False, "Target1")


