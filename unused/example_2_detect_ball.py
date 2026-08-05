#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Detect a ball and draw a tracking circle around it
# Press a key to exit program (with camera window focused)
#

# Import required modules
import cv2
from example_ball_detector import BallDetector
from gretchen.camera import Camera


def main():
    # Initialize ROS environment and camera
    # Device path to motor, camera
    #   Ubuntu/Linux  - motor: '/dev/grt_motor', camera: '/dev/grt_cam'
    #   Mac - motor: /dev/tty.usbserial-FT5WJ4JS', camera: '/dev/cu.usbserial-FT5WJ4JS' or 0
    #   Windows - motor: 'COM4', camera: 0
    camera = Camera(1)
    camera.start()

    # Initialize ball detector
    ball_detector = BallDetector()

    print("Camera started. Focus on the 'Frame' window and press any key to exit.")

    # Loop
    while True:        
        ret, img, timestamp = camera.getImage()

        # Run ball detector on image
        (img, center) = ball_detector.detect(img)
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        # ----------------------------------------------------------

        # Display image with the overlays
        cv2.imshow("Frame", img)

        # Exit loop if key was pressed
        key = cv2.waitKey(1)
        if key > 0:
            break


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
