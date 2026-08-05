#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# First tries moving Gretchen.
# Combine the camera with the robot to have Gretchen
# look at the point clicked in the image.
#

# Import required modules
import cv2
from gretchen.robot import Robot

# Create camera & robot
#Device path to motor, camera
#   Ubuntu/Linux  - motor: '/dev/grt_motor', camera: '/dev/grt_cam'
#   Mac - motor: /dev/tty.usbserial-FT5WJ4JS', camera: '/dev/cu.usbserial-FT5WJ4JS' or 0
#   Windows - motor: 'COM4', camera: 0
#robot = Robot('/dev/tty.usbserial-FT94EO3U', 0)

robot = Robot('/dev/tty.usbserial-FT6RW7EI', 1)

camera = robot.camera

# Image point
p = (320, 240)

# Method executed on click in camera image
def onMouse(event, u, v, flags, param):
    global p

    # If left button is clicked...
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print 2d coordinates of clicked pixel (y/x on image)
        p = (u, v)
        print("Image:         ({}, {})".format(u,v))

        # Convert the 2d coordinates to 3d coordinates in camera frame
        # and print the result
        (x,y,z) = robot.convert2d_3d(u,v)
        print("Camera frame: ({}, {}, {})".format(x,y,z))

        # The robot homography is need for convert3d_3d.
        # We use the current robot position to compute the robot homography
        c_theta0, c_theta1 = robot.getPosition()
        H0, H1, H2 = robot.forwardKinematics(c_theta0, c_theta1, delta0 = 0.15, delta1 = 0, delta2 = 0.5)
        robot_H = H0 @ H1 @ H2


        # Convert the 3d coordinates from the camera frame into
        # Gretchen's frame using a transformatio matrix, then
        # print the result and have Gretchen look at that point
        (x,y,z) = robot.convert3d_3d(x,y,z, robot_H)
        print ("Robot frame:  ({}, {}, {})".format(x,y,z))
        robot.lookatpoint(x,y,z)


def main():
    global p

    # Initalize ROS environment, start robot and camera
    robot.start()

    # Create a window called "Frame" and install a mouse handler
    cv2.namedWindow("Frame")
#    cv2.setMouseCallback("Frame", onMouse)
    robot.move(0,0)
    # Loop
    while True:
        # Get image from camera
        ret, img, timestamp = camera.getImage()

        # Use OpenCV to draw a circle around clicked point
        # Change the color of the circle
        cv2.circle(img, p, 10, (0, 0, 255), 3)

        # Use OpenCV to show image in a window named "Frame"
        cv2.imshow("Frame", img[...,::])
        p = (320, 240)

        # Exit loop if key was pressed
        key = cv2.waitKey(1)
        if key > 0:
            break


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
