#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Camera class
#
#
#

import cv2
from datetime import datetime
################################################################################
#   Camera class
#
#   Input:   (String) path to device
#
#
################################################################################

class Camera:
    def __init__(self, path):

        #Camera parameters
        self.fx = 570.3422241210938*2
        self.fy = 570.3422241210938*2
        self.cx = 319.5
        self.cy = 239.5

        #VideoCapture
        self.vc = None

        #timestamp
        self.dt = datetime.now()
        self.ts = datetime.timestamp(self.dt)

        #Path to camera device
        self.path = path

        #Throttles the "frame read failed" print so a stuck camera doesn't
        #flood stdout and add its own lag on top of the read failures
        self._failed_read_count = 0

    ################################################################################
    #   Starts camera inputs
    #
    #
    #
    #
    ################################################################################
    def start(self):
        self.vc = cv2.VideoCapture(self.path)
        if not self.vc.isOpened():
            print(f"Camera failed to open: path={self.path!r}")
            return

        # Request the lower resolution directly from the camera instead of
        # capturing full-size and resizing after the fact - cuts USB
        # bandwidth at the source, which matters on a bandwidth-limited
        # port/hub. Not all cameras honor this, but it's free to try.
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.vc.set(cv2.CAP_PROP_FPS, 15)
        actual_w = self.vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.vc.get(cv2.CAP_PROP_FPS)
        print(f"Camera {self.path!r} capturing at {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f}fps")

    ################################################################################
    #   Gets image frame
    #
    #   Returns:   (Boolean) ret - True, if read properly
    #              (np.array) frame - image frame
    #              (float) self.ts - timestamp
    #
    #
    ################################################################################
    def getImage(self):
        ret, frame = self.vc.read()
        self.dt = datetime.now()
        self.ts = datetime.timestamp(self.dt)

        if not ret or frame is None:
            self._failed_read_count += 1
            if self._failed_read_count % 30 == 1:
                print(f"Frame read failed ({self._failed_read_count} so far): path={self.path!r}")
            return False, None, self.ts

        #Resize
        frame = cv2.resize(frame, (640, 480))
        return ret, frame, self.ts

def main():
    print("Hello World")
    cv2.namedWindow("Camera")
    camera = Camera('/dev/grt_cam')
    vc = camera.start()
    while True:
        ret, frame, ts = camera.getImage()
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key > 0:
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
