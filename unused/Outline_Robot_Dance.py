
import time
import threading
import cv2
import argparse
import numpy as np
import hashlib
import sys
from gretchen.robot import Robot

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY = 159.3340, 146.2192, 300.8006, 250.6516
CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY = 97.2448, 166.3934, 326.2788, 240.9489

CAMERA_B_R = np.array([[0.9727336636758184, -0.11933485621410486, 0.19886782455736485], [0.06261490710602888, 0.9607525094791584, 0.2702480137532986], [-0.2233127693615466, -0.2504272501583216, 0.9420284493677556]])
CAMERA_B_T = np.array([-0.25774806772504666, -0.22067519618419015, 0.14534646663239795])

ACCEL_NOISE = 500.0
MEASUREMENT_NOISE_STD = 0.02  # metres
PRINT_INTERVAL_S = 0.2

def main():
    # Initialize Robot and Camera
    #Device path to motor, camera
    #   Ubuntu/Linux  - motor: '/dev/grt_motor', camera: '/dev/grt_cam'
    #   Mac - motor: /dev/tty.usbserial-FT5WJ4JS', camera: '/dev/cu.usbserial-FT5WJ4JS' or 0
    #   Windows - motor: 'COM4', camera: 0
    robot = Robot('/dev/grt_motor', '/dev/grt_cam') #EDIT ACCORDING TO DEVICE
    camera = robot.camera
    robot.start()

    VELOCITY= #enter variable here

    if VELOCITY<X #Replace X with "sad" threshold:
    	#Speed= 15rev/sec
    	#Motor 0:  150->90, Motor 1:  150->110
    	#Sleep 
    	#if time_elapsed=2:
    		#Motor1=150, Motor 2=150

    elif VELOCITY≤X≤Y #Replace Y with "satisfied" threshold:
    	#Speed= 15rev/sec
    	#Motor 1:  150->110->150
    	#Sleep 
    	#if time_elapsed=2:
    		#Motor1=150, Motor 2=150

    elif VELOCITY≤y≤Z #Replace Z with "Happy" threshold:
    	#Speed= 20rev/sec
    	#Motor 1:  150->110->150->110->150
    	#Sleep 
    	#if time_elapsed=2:
    		#Motor1=150, Motor 2=150

    else:
    	#Speed= 15rev/sec
    	#Motor 0: 60-> 240-> 150 Motor 1: 100-> 190->100-> 190 -> 110 -> 150
    	#Sleep 
    	#if time_elapsed=2:
    		#Motor1=150, Motor 2=150





