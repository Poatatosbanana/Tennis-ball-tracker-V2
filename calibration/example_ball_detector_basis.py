#
# 9-tennis
#
# Ball detector class (copy of 2-tracking-a-ball/example_ball_detector.py,
# kept local so this folder's scripts are self-contained)
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
        # Tuned from color_picker_for_the_final_final_filter.py for this
        # ball/lighting - see unused/final_final_color_filter.py.
        # Remember OpenCV's hue range is scaled to 0..179.
        self.colorLower = (73, 40, 152)
        self.colorUpper = (97, 255, 255)

    # Class method that detects a ball and marks it on the frame
    def detect(self, frame):
        # Convert frame from RGB to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Apply a bilateral filter to remove unwanted noise while preserving edges
        hsv = cv2.bilateralFilter(hsv, 15, 100, 100)

        # Create a mask from the frame that only contains values falling in between colorLower...colorUpper
        mask = cv2.inRange(hsv, self.colorLower, self.colorUpper)

        # Apply some more filters to get rid of noise
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find all contours in the mask
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Find all circles
        circles = []
        max_radius = 0
        max_center = None
        for cnt in cnts:
            contour_area = cv2.contourArea(cnt)

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            circle_area = self.PI*radius*radius

            # Ignore very small circles
            if radius < 4:
                continue

            # If the area of the contour makes up for least 60% of the enclosing circle,
            # then the contour resembles a circle and we include it
            if contour_area / circle_area > 0.60:
                center = (int(x), int(y))
                circles.append((center, int(radius)))

                if radius > max_radius:
                    max_radius = radius
                    max_center = center

        # Mark all identified circles
        for (center, radius) in circles:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

        # Return frame, center of largest circle, and the mask (for callers
        # that want to display it themselves - imshow must run on the main
        # thread, so this class no longer calls it directly)
        return [frame, max_center, mask]
