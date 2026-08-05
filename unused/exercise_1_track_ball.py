#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Detect and track a ball
# Press a key to exit program (with camera window focused)
#
# Currently, this exercise is a copy of example 2. Modify it such that
# * it allows the user to click on the image and select the color of the filter
# * it moves Gretchen's head to follow the largest detected circle.
#

# Import required modules
import cv2
import sys
from example_ball_detector import BallDetector
from gretchen.robot import Robot


# convert a value in RGB into HSV
# see https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
def rgb2hsv(r, g, b):
    r = float(r) / 255
    g = float(g) / 255
    b = float(b) / 255

    print(r, g, b)

    v = xmax = max(r, g, b)
    xmin = min(r, g, b)
    c = xmax - xmin

    print(v, xmin, c)

    # hue (0..360)
    if c == 0:
        h = 0
    elif v == r:
        h = int(60 * (0 + (g-b)/c))
    elif v == g:
        h = int(60 * (2 + (b-r)/c))
    else:
        h = int(60 * (4 + (r-g)/c))

    if (h < 0):
        h = 360 + h

    # saturation on an integer scale from 0..100
    if v > 0:
        s = int(255 * c / v)
    else:
        s = 0

    # value an integer scale from 0..100
    v = int(255 * v)

    print(h, s, v)

    return h, s, v


# Method executed on mouse event in camera image
def onMouse(event, u, v, flags, param):
    # access global variable 'point'
    global point
    global H, S, V

    # If left button is clicked...
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get image from camera
        ret, img, timestamp = camera.getImage()

        point = (u,v)
        (R, G, B) = img[v,u]

        # Convert RBG to HSV
        (H, S, V) = rgb2hsv(R, G, B)

        # print result
        print("Point clicked:   ({}/{})\n"
              "  RGB value:     ({:3d}, {:3d}, {:3d})\n"
              "  HSV value:     ({:3d}, {:3d}, {:3d})"
              .format(v, u,
                      R, G, B,
                      H, S, V))


        # double check using OpenCV
        # convert entire image to HSV
        # refer to https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
        # Note that OpenCV uses a range of 0..179 for hue, so we scale it back to 0..359 when printing
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        (H, S, V) = hsv[v,u]
        H, S, V = int(H), int(S), int(V)

        print("  HSV (OpenCV):  ({:3d}, {:3d}, {:3d})\n"
              .format(H*2, S, V))



H, S, V = 0, 0, 0
point = (0,0)
robot = Robot('/dev/tty.usbserial-FT94EO3U', 0)
camera = robot.camera
tolerance = 50



def main():
    # Initialize Robot and Camera
    #Device path to motor, camera
    #   Ubuntu/Linux  - motor: '/dev/grt_motor', camera: '/dev/grt_cam'
    #   Mac - motor: /dev/tty.usbserial-FT5WJ4JS', camera: '/dev/cu.usbserial-FT5WJ4JS' or 0
    #   Windows - motor: 'COM4', camera: 0

    robot.start()
    
    # Announce frame and set mouse handler
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", onMouse)

    # Initalize ball detector
    ball_detector = BallDetector()

    # Smoothed ball center, updated with a decaying (exponential moving) average
    smoothed_center = None
    alpha = 0.2  # weight given to the new detection each frame; lower = smoother/slower, higher = snappier

    # Loop
    while True:
        # Get image from camera
        ret, img, timestamp = camera.getImage()

        # Draw circle
        cv2.circle(img, point, 10, (0, 0, 255), 3)



        ball_detector.colorLower = (max(0, H - tolerance), 80, 30)
        ball_detector.colorUpper = (min(179, H + tolerance), 255, 255)

        # Run ball detector on image
        (img, center) = ball_detector.detect(img)

        # Only smooth and move when the ball is actually detected this frame.
        # Without this guard, losing the ball would keep re-issuing the last
        # stale target every frame, causing the head to keep drifting.
        if center is not None:
            if smoothed_center is None:
                smoothed_center = center
            else:
                smoothed_center = (
                    alpha * center[0] + (1 - alpha) * smoothed_center[0],
                    alpha * center[1] + (1 - alpha) * smoothed_center[1],
                )

            # Move Gretchen's head to look at the smoothed ball position
            u, v = smoothed_center
            (x, y, z) = robot.convert2d_3d(u, v)
            c_theta0, c_theta1 = robot.getPosition()
            H0, H1, H2 = robot.forwardKinematics(c_theta0, c_theta1, robot.d0, robot.d1, robot.d2)
            robot_H = H0 @ H1 @ H2
            (x, y, z) = robot.convert3d_3d(x, y, z, robot_H)
            robot.lookatpoint(x, y, z)

        # Display image
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
