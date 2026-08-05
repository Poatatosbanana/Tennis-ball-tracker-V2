#!/usr/bin/env python3
#
# 9-tennis
#
# Single-camera ball tracking with a manually-set HSV color filter.
# Set the HSV_LOWER / HSV_UPPER values below (e.g. from your own color-picker
# script) and the detector will use them directly. No clicking, no motor
# control, no stereo, no speed math -- just detect-and-mark.
#
# Controls:
#   * 'q' or ESC -> quit
#
# Why the color space matters (do not "fix" this):
#   BallDetector.detect() builds its mask with cv2.cvtColor(frame,
#   COLOR_RGB2HSV) even though the camera delivers BGR. Whatever tool you use
#   to pick your HSV values needs to sample in that same color space
#   (RGB2HSV, not BGR2HSV) or the numbers you paste in below won't match what
#   the detector actually filters on.
#

import cv2

from example_ball_detector_basis import BallDetector
from gretchen.camera import Camera

# Camera device index. ROBOT's stereo scripts use 0 (camera A) and 1 (camera B);
# the first-steps course examples used 2. Change this to match your setup.
CAMERA_INDEX = 1

FRAME_WIN = "Ball tracking"

# ======================================================================
# >>> SET YOUR BALL COLOR HERE (H, S, V) -- from your color-picker script <<<
# ======================================================================
HSV_LOWER = (73, 40, 152)    # (H_low, S_low, V_low)
HSV_UPPER = (97, 255, 255)  # (H_high, S_high, V_high)
# ======================================================================


def main():
    camera = Camera(CAMERA_INDEX)
    camera.start()

    detector = BallDetector()
    detector.colorLower = HSV_LOWER
    detector.colorUpper = HSV_UPPER

    cv2.namedWindow(FRAME_WIN)

    while True:
        ret, img, timestamp = camera.getImage()
        if not ret or img is None:
            continue

        # detect() marks all detected circles on `img` and returns the largest.
        img, center = detector.detect(img)

        # HUD
        cv2.putText(img, "q/ESC to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "filter {} .. {}".format(tuple(detector.colorLower),
                                                  tuple(detector.colorUpper)),
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if center is not None:
            cv2.putText(img, "ball @ {}".format(center),
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(FRAME_WIN, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
