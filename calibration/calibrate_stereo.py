#!/usr/bin/env python3
#
# 9-tennis
#
# One-time stereo extrinsic calibration for the two stationary cameras used
# in measure_ball_speed_stationary.py. Show a checkerboard to both cameras
# from several different positions/angles; this computes the relative
# rotation/translation (R, T) of camera B with respect to camera A - the
# values to paste into CAMERA_B_R / CAMERA_B_T there.
#
# BEFORE RUNNING:
#   - Verify CAMERA_A_INDEX / CAMERA_B_INDEX with test_camera_indices.py.
#   - Set BOARD_SIZE to your checkerboard's internal corner count
#     (columns, rows) and SQUARE_SIZE_M to its real square size in metres.
#
# Controls (with a camera window focused):
#   space - capture the current frame pair (only registers if a
#           checkerboard was found in both images)
#   c     - finish capturing and run calibration
#   q     - quit without calibrating
#

import cv2
import numpy as np
from gretchen.camera import Camera

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

BOARD_SIZE = (3, 4)       # internal corners (columns, rows)
SQUARE_SIZE_M = 0.0188     # real-world size of one checkerboard square, metres

MIN_CAPTURES = 20


def make_object_points():
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def find_corners(gray):
    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def main():
    camera_a = Camera(CAMERA_A_INDEX)
    camera_b = Camera(CAMERA_B_INDEX)
    camera_a.start()
    camera_b.start()

    cv2.namedWindow("Camera A")
    cv2.namedWindow("Camera B")

    objp = make_object_points()
    objpoints = []
    imgpoints_a = []
    imgpoints_b = []
    image_size = None

    print(f"Show the checkerboard ({BOARD_SIZE[0]}x{BOARD_SIZE[1]} corners) to "
          f"both cameras. Press SPACE to capture a pair, C to calibrate "
          f"(need >= {MIN_CAPTURES} captures), Q to quit.")

    while True:
        ret_a, img_a, _ = camera_a.getImage()
        ret_b, img_b, _ = camera_b.getImage()

        if not ret_a or not ret_b or img_a is None or img_b is None:
            key = cv2.waitKey(1)
            if key in (ord('q'), ord('Q')):
                return
            continue

        image_size = (img_a.shape[1], img_a.shape[0])

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        found_a, corners_a = find_corners(gray_a)
        found_b, corners_b = find_corners(gray_b)

        disp_a = img_a.copy()
        disp_b = img_b.copy()
        if found_a:
            cv2.drawChessboardCorners(disp_a, BOARD_SIZE, corners_a, found_a)
        if found_b:
            cv2.drawChessboardCorners(disp_b, BOARD_SIZE, corners_b, found_b)

        status = f"captures: {len(objpoints)}  found: A={found_a} B={found_b}"
        cv2.putText(disp_a, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camera A", disp_a)
        cv2.imshow("Camera B", disp_b)

        key = cv2.waitKey(1)
        if key in (ord('q'), ord('Q')):
            print("Quit without calibrating.")
            return
        elif key == ord(' '):
            if found_a and found_b:
                objpoints.append(objp)
                imgpoints_a.append(corners_a)
                imgpoints_b.append(corners_b)
                print(f"Captured pair {len(objpoints)}")
            else:
                print("Checkerboard not found in both cameras - reposition and try again.")
        elif key in (ord('c'), ord('C')):
            if len(objpoints) < MIN_CAPTURES:
                print(f"Need at least {MIN_CAPTURES} captures, have {len(objpoints)}.")
                continue
            break

    print("Calibrating each camera individually...")
    ret_a, mtx_a, dist_a, _, _ = cv2.calibrateCamera(objpoints, imgpoints_a, image_size, None, None)
    ret_b, mtx_b, dist_b, _, _ = cv2.calibrateCamera(objpoints, imgpoints_b, image_size, None, None)
    print(f"Camera A reprojection error: {ret_a:.4f}")
    print(f"Camera B reprojection error: {ret_b:.4f}")

    print("Running stereo calibration...")
    ret_stereo, mtx_a, dist_a, mtx_b, dist_b, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_a, imgpoints_b,
        mtx_a, dist_a, mtx_b, dist_b,
        image_size, flags=cv2.CALIB_FIX_INTRINSIC)

    print(f"Stereo reprojection error: {ret_stereo:.4f}")
    print()
    print("Camera A intrinsics (fx, fy, cx, cy):")
    print(f"  fx={mtx_a[0,0]:.4f}  fy={mtx_a[1,1]:.4f}  cx={mtx_a[0,2]:.4f}  cy={mtx_a[1,2]:.4f}")
    print("Camera B intrinsics (fx, fy, cx, cy):")
    print(f"  fx={mtx_b[0,0]:.4f}  fy={mtx_b[1,1]:.4f}  cx={mtx_b[0,2]:.4f}  cy={mtx_b[1,2]:.4f}")
    print()
    print("Paste these into measure_ball_speed_stationary.py:")
    print(f"CAMERA_B_R = np.array({R.tolist()})")
    print(f"CAMERA_B_T = np.array({T.flatten().tolist()})")

    capture_ground_reference(camera_a, objp, mtx_a, dist_a)
    cv2.destroyAllWindows()


def capture_ground_reference(camera_a, objp, mtx_a, dist_a):
    # Camera A's own frame (used as the triangulation "world" frame) is
    # tilted however camera A physically happens to be mounted - real
    # vertical motion ends up smeared across all three estimated axes.
    # Fix: lay the checkerboard FLAT on a known-level surface (floor/court)
    # so the board's own out-of-plane axis is true vertical, solve its pose
    # relative to camera A, and hand back the rotation that un-tilts camera
    # A's frame into a gravity-aligned one (z = true up).
    print()
    print("Ground reference: lay the checkerboard FLAT on a level surface "
          "(floor/court), fully visible to Camera A. Press SPACE to "
          "capture, or Q to skip this step.")

    while True:
        ret_a, img_a, _ = camera_a.getImage()
        if not ret_a or img_a is None:
            key = cv2.waitKey(1)
            if key in (ord('q'), ord('Q')):
                return
            continue

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        found_a, corners_a = find_corners(gray_a)

        disp_a = img_a.copy()
        if found_a:
            cv2.drawChessboardCorners(disp_a, BOARD_SIZE, corners_a, found_a)
        cv2.imshow("Camera A", disp_a)

        key = cv2.waitKey(1)
        if key in (ord('q'), ord('Q')):
            print("Skipped ground reference capture.")
            return
        elif key == ord(' '):
            if not found_a:
                print("Checkerboard not found - reposition and try again.")
                continue
            ok, rvec, _ = cv2.solvePnP(objp, corners_a, mtx_a, dist_a)
            if not ok:
                print("solvePnP failed - try again.")
                continue
            r_ground, _ = cv2.Rodrigues(rvec)
            world_r = r_ground.T
            print()
            print("Paste this into measure_ball_speed_threaded.py to align "
                  "triangulated points to a gravity-up frame (z = vertical):")
            print(f"WORLD_R = np.array({world_r.tolist()})")
            return


if __name__ == '__main__':
    main()
