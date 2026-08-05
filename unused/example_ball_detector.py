#
# First Steps in Programming a Humanoid AI Robot
#
# Ball detector class
# Used by example_2_detect_ball.py
#

# Import required modules
import numpy as np
import cv2
import imutils

class BallDetector:
    PI = 3.141592

    # Class constructor
    def __init__(self):
        # Lower and upper limits for detected color (H, S, V)
        # This example tracks green balls.
        # Also remember that OpenCVs hue range is scaled to 0..179
        self.colorLower = ( 80, 50,  80) #30 80 30
        self.colorUpper = ( 95, 150, 255)

    # Class method that detects a ball and marks it on the frame
    def detect(self, frame):
        # Convert frame from RGB to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Apply a bilateral filter to remove unwanted noise while preserving edges
        # For details on bilateral filtering, see
        #   https://learnopencv.com/image-filtering-using-convolution-in-opencv/#bilateral-filter-opencv
        # A radius of 10 gives good results while not being too computationally intensive
        hsv2 = cv2.bilateralFilter(hsv, 15, 35, 75)
##        cv2.imshow("bilateral filter2", hsv2)


        # Create a mask from the frame that only contains values falling in between colorLower...colorUpper
        colormask = cv2.inRange(hsv2, self.colorLower, self.colorUpper)
##        cv2.imshow("in range filter", mask)

        # Apply some more filters to get rid of noise
        # A good discussion on erode and dilate filters can be found here
        #  https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
        erodemask1 = cv2.erode(colormask, None, iterations=2)
##        cv2.imshow("erosion filter", erodemask1)
        
        mask = cv2.dilate(erodemask1, None, iterations=4)

        cv2.imshow("final filter", mask)

        # Ask OpenCV to find all contours in the mask, then use IMutils' grab_contours() function
        # to extract all contours in a uniform format (independent of OpenCV library version).
        # Details can be found here:
        #   https://www.educba.com/opencv-findcontours/
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Find all circles
        centor = None
        circles = []
        max_radius = 0
        max_center = None
        for cnt in cnts:
            # Calculate the area of the contour
            contour_area = cv2.contourArea(cnt)

            # Get the minimum enclosing circle and compute its area
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            circle_area = self.PI*radius*radius

            # Ignore very small circles
            if radius < 1: #FROM 10
                continue

            # If the area of the contour makes up for least 75% of the enclosing circle,
            # then the contour resembles a circle and we include it
            if contour_area / circle_area > 0.75: #FROM 0.75
                center = (int(x), int(y))
                circles.append((center, int(radius)))

                # Did we find a new biggest circle?
                if radius > max_radius:
                    max_radius = radius
                    max_center = center

        # Mark all identified circles
        for (center, radius) in circles:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

        # Return frame and center of largest circle
        return [frame, max_center]
