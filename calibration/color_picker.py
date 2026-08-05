#!/usr/bin/env python3
#
# color_picker.py
#
# Standalone HSV color picker. Left-click any pixel in the camera window and
# it prints the sampled HSV value plus a suggested colorLower/colorUpper
# range to the terminal -- copy those straight into HSV_LOWER / HSV_UPPER in
# 9-tennis.py.
#
# Controls:
#   * Left-click a pixel -> print HSV + suggested range to terminal
#   * 'q' or ESC          -> quit
#
# Why the color space matters (do not "fix" this):
#   BallDetector.detect() (used by 9-tennis.py) builds its mask with
#   cv2.cvtColor(frame, COLOR_RGB2HSV) even though the camera delivers BGR.
#   This picker uses the *same* COLOR_RGB2HSV call so the numbers it prints
#   land in the exact same HSV space the detector thresholds against.
#

import cv2
import numpy as np

from gretchen.camera import Camera

# Camera device index. ROBOT's stereo scripts use 0 (camera A) and 1 (camera B);
# the first-steps course examples used 2. Change this to match your setup.
CAMERA_INDEX = 0

FRAME_WIN = "Color picker (left-click a pixel; q/ESC to quit)"

# Sampling patch size and margins -- same defaults as the old in-app picker.
PATCH = 50
H_MARGIN = 12
S_MARGIN = 70
V_MARGIN = 70
S_FLOOR = 40
V_FLOOR = 80

_last_frame = None
_last_sample_rect = None


def sample_hsv(frame, x, y):
    """Sample a PATCH x PATCH window around (x, y), return median HSV plus
    a suggested (lower, upper) filter range built from it."""
    h_img, w_img = frame.shape[:2]

    half = PATCH // 2
    x0 = max(x - half, 0)
    y0 = max(y - half, 0)
    x1 = min(x + half + 1, w_img)
    y1 = min(y + half + 1, h_img)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)   # match BallDetector.detect()
    region = hsv[y0:y1, x0:x1].reshape(-1, 3)

    h, s, v = [int(c) for c in np.median(region, axis=0)]

    lower = (int(np.clip(h - H_MARGIN, 0, 179)),
             max(s - S_MARGIN, S_FLOOR),
             max(v - V_MARGIN, V_FLOOR))
    upper = (int(np.clip(h + H_MARGIN, 0, 179)), 255, 255)

    return h, s, v, lower, upper


def on_mouse(event, x, y, flags, param):
    global _last_sample_rect

    if event == cv2.EVENT_LBUTTONDOWN and _last_frame is not None:
        h, s, v, lower, upper = sample_hsv(_last_frame, x, y)
        print("Clicked ({}, {}) -> HSV median ({}, {}, {})".format(x, y, h, s, v))
        print("  HSV_LOWER = {}".format(lower))
        print("  HSV_UPPER = {}".format(upper))
        print()

        h_img, w_img = _last_frame.shape[:2]
        half = PATCH // 2
        x0 = max(x - half, 0)
        y0 = max(y - half, 0)
        x1 = min(x + half + 1, w_img)
        y1 = min(y + half + 1, h_img)
        _last_sample_rect = (x0, y0, x1, y1)


def main():
    global _last_frame

    camera = Camera(CAMERA_INDEX)
    camera.start()

    cv2.namedWindow(FRAME_WIN)
    cv2.setMouseCallback(FRAME_WIN, on_mouse)

    print("Left-click the ball to print its HSV filter range. q/ESC to quit.\n")

    while True:
        ret, img, timestamp = camera.getImage()
        if not ret or img is None:
            continue

        _last_frame = img.copy()

        if _last_sample_rect is not None:
            x0, y0, x1, y1 = _last_sample_rect
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
            cv2.putText(img, "sample region", (x0, max(y0 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(img, "left-click a pixel; q/ESC to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(FRAME_WIN, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
